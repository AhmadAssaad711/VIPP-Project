$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$artifactRoot = Join-Path $repo 'artifacts\final_Results'
$cbfStudyRelative = 'artifacts\final_Results\ppo200k_cbf5'
$noSafetyStudyRelative = 'artifacts\final_Results\ppo200k_nosafety5'
$cbfStudy = Join-Path $repo $cbfStudyRelative
$noSafetyStudy = Join-Path $repo $noSafetyStudyRelative
$logRoot = Join-Path $artifactRoot '_runlogs_ppo200k'
$statusLog = Join-Path $artifactRoot 'ppo200k_resilient_status.log'
$successMarker = Join-Path $cbfStudy 'study_success.marker'
$maxAttempts = 6
$staleSeconds = 600
$expectedWorkerCount = 20
$workerStartupGraceSeconds = 180
$minimumAvailableMemoryMB = 8192
$maximumCommitPercent = 85.0

Add-Type -AssemblyName System.IO.Compression.FileSystem

# Keep native numerical libraries deterministic and prevent each spawned
# worker from creating a second thread pool.  The learner is explicitly CPU,
# so hiding CUDA also avoids loading an unused native runtime in every child.
$nativeProcessEnvironment = [ordered]@{
    OMP_NUM_THREADS = '1'
    OMP_THREAD_LIMIT = '1'
    OMP_MAX_ACTIVE_LEVELS = '1'
    OPENBLAS_NUM_THREADS = '1'
    BLIS_NUM_THREADS = '1'
    MKL_NUM_THREADS = '1'
    NUMEXPR_NUM_THREADS = '1'
    VECLIB_MAXIMUM_THREADS = '1'
    TORCH_NUM_THREADS = '1'
    OMP_DYNAMIC = 'FALSE'
    MKL_DYNAMIC = 'FALSE'
    OMP_WAIT_POLICY = 'PASSIVE'
    KMP_BLOCKTIME = '0'
    PYTHONFAULTHANDLER = '1'
    CUDA_VISIBLE_DEVICES = '-1'
    MPLBACKEND = 'Agg'
}
foreach ($entry in $nativeProcessEnvironment.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable(
        [string]$entry.Key,
        [string]$entry.Value,
        [EnvironmentVariableTarget]::Process
    )
}

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Set-Content -LiteralPath $statusLog -Value ("started_at=" + (Get-Date).ToString('o')) -Encoding UTF8
if (Test-Path -LiteralPath $successMarker) {
    Remove-Item -LiteralPath $successMarker -Force
}

function Write-Status([string]$Message) {
    $line = (Get-Date).ToString('o') + ' ' + $Message
    Add-Content -LiteralPath $statusLog -Value $line -Encoding UTF8
    # Do not emit a success-stream object here.  Invoke-TrackedPython returns a
    # Boolean, and status strings in that pipeline would make a failed process
    # look truthy to its caller.
}

function Get-RunDirectory([string]$StudyRelativePath, [string]$Variant, [int]$Seed) {
    return Join-Path (Join-Path (Join-Path $repo $StudyRelativePath) $Variant) ('seed_' + $Seed)
}

function Test-ZipArchiveReadable([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $archive = $null
    try {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
        return [bool]($archive.Entries.Count -gt 0)
    } catch {
        return $false
    } finally {
        if ($null -ne $archive) {
            $archive.Dispose()
        }
    }
}

function Test-RunComplete([string]$RunDirectory) {
    $modelPath = Join-Path $RunDirectory 'model_final.zip'
    $completionPath = Join-Path $RunDirectory 'training_complete.json'
    if (
        -not (Test-Path -LiteralPath $modelPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $completionPath -PathType Leaf) -or
        -not (Test-ZipArchiveReadable $modelPath)
    ) {
        return $false
    }
    try {
        $completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
        if (
            [string]$completion.model_file -ne 'model_final.zip' -or
            [int64]$completion.num_timesteps -ne 200000 -or
            [string]::IsNullOrWhiteSpace([string]$completion.model_sha256)
        ) {
            return $false
        }
        $observedHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash
        return [string]::Equals(
            [string]$completion.model_sha256,
            [string]$observedHash,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    } catch {
        return $false
    }
}

function Get-LatestCheckpoint([string]$RunDirectory) {
    $checkpointDirectory = Join-Path $RunDirectory 'checkpoints'
    if (-not (Test-Path -LiteralPath $checkpointDirectory)) {
        return $null
    }
    $candidates = @()
    foreach ($path in (Get-ChildItem -LiteralPath $checkpointDirectory -Filter 'rollout_*_steps.zip' -File -ErrorAction SilentlyContinue)) {
        if ($path.Name -match '^rollout_(\d+)_steps\.zip$') {
            if (Test-ZipArchiveReadable $path.FullName) {
                $candidates += [pscustomobject]@{ step = [int64]$Matches[1]; path = $path.FullName }
            } else {
                Write-Status ("ignored_unreadable_checkpoint path=" + $path.FullName)
            }
        }
    }
    if ($candidates.Count -eq 0) {
        return $null
    }
    return ($candidates | Sort-Object step | Select-Object -Last 1).path
}

function Move-PartialRun(
    [string]$StudyRelativePath,
    [string]$Variant,
    [int]$Seed,
    [string]$ArchiveLabel
) {
    $source = Get-RunDirectory $StudyRelativePath $Variant $Seed
    if (-not (Test-Path -LiteralPath $source)) {
        return
    }
    if (Test-RunComplete $source) {
        return
    }
    $archiveRoot = Join-Path $artifactRoot ('recovery_' + $ArchiveLabel)
    New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
    $archiveName = $Variant + '_seed_' + $Seed + '_' + (Get-Date).ToString('yyyyMMdd_HHmmss')
    $destination = Join-Path $archiveRoot $archiveName
    $resolvedRoot = (Resolve-Path $artifactRoot).Path
    $resolvedSource = (Resolve-Path $source).Path
    if (-not $resolvedSource.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to archive a run outside the artifact root: $resolvedSource"
    }
    if (-not $destination.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to archive to a destination outside the artifact root: $destination"
    }
    Move-Item -LiteralPath $source -Destination $destination
    Write-Status ("archived_partial_run source=" + $source + " destination=" + $destination)
}

function Stop-ProcessTree([int]$ParentProcessId) {
    $allIds = @($ParentProcessId)
    $frontier = @($ParentProcessId)
    while ($frontier.Count -gt 0) {
        $children = @(
            Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object { $frontier -contains $_.ParentProcessId } |
                Select-Object -ExpandProperty ProcessId
        )
        $newChildren = @($children | Where-Object { $allIds -notcontains $_ -and $_ -ne $PID })
        if ($newChildren.Count -eq 0) {
            break
        }
        $allIds += $newChildren
        $frontier = $newChildren
    }
    foreach ($processId in ($allIds | Sort-Object -Descending)) {
        if ($processId -ne $PID) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
    return $allIds
}

function Get-DescendantProcesses([int]$ParentProcessId) {
    $snapshot = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $descendantIds = @()
    $frontier = @($ParentProcessId)
    while ($frontier.Count -gt 0) {
        $children = @(
            $snapshot |
                Where-Object { $frontier -contains [int]$_.ParentProcessId } |
                Select-Object -ExpandProperty ProcessId
        )
        $newChildren = @(
            $children | Where-Object {
                $descendantIds -notcontains [int]$_ -and [int]$_ -ne $PID
            }
        )
        if ($newChildren.Count -eq 0) {
            break
        }
        foreach ($child in $newChildren) {
            $descendantIds += [int]$child
        }
        $frontier = @($newChildren | ForEach-Object { [int]$_ })
    }
    return @(
        $snapshot | Where-Object {
            $descendantIds -contains [int]$_.ProcessId
        }
    )
}

function Stop-ProcessDescendants([int]$ParentProcessId) {
    $descendants = @(Get-DescendantProcesses $ParentProcessId)
    foreach ($processId in ($descendants.ProcessId | Sort-Object -Descending)) {
        if ([int]$processId -ne $PID) {
            Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
        }
    }
    return @($descendants.ProcessId)
}

function Get-SystemHeadroom {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
    $availableMB = if ($null -ne $memory -and $null -ne $memory.AvailableMBytes) {
        [double]$memory.AvailableMBytes
    } elseif ($null -ne $os -and $null -ne $os.FreePhysicalMemory) {
        [double]$os.FreePhysicalMemory / 1024.0
    } else {
        [double]::PositiveInfinity
    }
    $committedBytes = if ($null -ne $memory) { [double]$memory.CommittedBytes } else { 0.0 }
    $commitLimit = if ($null -ne $memory) { [double]$memory.CommitLimit } else { 0.0 }
    $commitPercent = if ($commitLimit -gt 0.0) {
        100.0 * $committedBytes / $commitLimit
    } else {
        0.0
    }
    return [pscustomobject]@{
        available_mb = [math]::Round($availableMB, 2)
        commit_percent = [math]::Round($commitPercent, 2)
    }
}

function Wait-ForSystemHeadroom([string]$Label) {
    while ($true) {
        $headroom = Get-SystemHeadroom
        if (
            [double]$headroom.available_mb -ge $minimumAvailableMemoryMB -and
            [double]$headroom.commit_percent -lt $maximumCommitPercent
        ) {
            return
        }
        Write-Status (
            "waiting_for_memory_headroom label=" + $Label +
            " available_mb=" + $headroom.available_mb +
            " commit_percent=" + $headroom.commit_percent
        )
        Start-Sleep -Seconds 30
    }
}

function New-StudyArguments(
    [string]$StudyRelativePath,
    [string]$Variant,
    [int]$Seed,
    [ValidateSet('cbf', 'nosafety')]
    [string]$Mode,
    [bool]$ForceRetrain,
    [bool]$EvaluationOnly
) {
    $tensorboardLabel = if ($Mode -eq 'cbf') { 'ppo200k_cbf5' } else { 'ppo200k_nosafety_baseline5' }
    $arguments = @(
        '-u', 'scripts\run_ppo_cbf_progression.py',
        '--project-root', '.',
        '--output-dir', $StudyRelativePath,
        # CPU avoids the CUDA MLP path and reduces the memory pressure that
        # preceded the Windows worker access violations; collection remains 20-worker.
        '--device', 'cpu',
        '--timesteps', '200000',
        '--n-envs', '20',
        '--seeds', ([string]$Seed),
        '--variants', $Variant,
        '--ppo-config', 'Q1_stable',
        '--n-steps', '1000',
        '--batch-size', '100',
        '--n-epochs', '10',
        # Additional rollout-aligned checkpoints do not change the learned model;
        # they make a native worker failure recoverable before 100k steps.
        '--checkpoint-freq', '25000',
        '--reward-mode', 'reciprocal',
        '--lambda-delta', '0.05',
        '--lambda-intervention', '0.10',
        '--lambda-mean', '0.10',
        '--lambda-detached-actor', '0.10',
        '--lambda-sample', '0.0',
        '--correction-epsilon', '0.03',
        '--traffic-model', 'mtm',
        '--remove-vehicle-dimensions',
        '--expose-target-y',
        '--task-distance-m', '600',
        '--task-max-policy-steps', '3000',
        '--post-train-eval-episodes', '200',
        '--post-train-eval-workers', '20',
        '--post-train-eval-seed-start', '1100000',
        '--skip-evaluation',
        '--skip-counterfactual',
        '--tensorboard-run-label', $tensorboardLabel
    )
    if ($Mode -eq 'cbf') {
        $arguments += @(
            '--safety-potential-formulation', 'cbf_violation',
            '--safety-cbf-alpha', '1.0',
            '--safety-cbf-psi-scale', '1.0',
            '--safety-potential-weight', '0.0'
        )
    } else {
        $arguments += '--disable-safety-shaping'
    }
    if ($ForceRetrain) {
        $arguments += '--force-retrain'
    }
    if ($EvaluationOnly) {
        $arguments += @('--skip-training', '--post-train-evaluate-reused')
    }
    return $arguments
}

function Invoke-TrackedPython(
    [string]$Label,
    [string[]]$Arguments,
    [string]$RunDirectory,
    [bool]$AllowStaleAfterCompletion
) {
    $stdout = Join-Path $logRoot ($Label + '_stdout.log')
    $stderr = Join-Path $logRoot ($Label + '_stderr.log')
    Wait-ForSystemHeadroom $Label
    $process = Start-Process -FilePath 'python.exe' -ArgumentList $Arguments -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Write-Status ("started label=" + $Label + " pid=" + $process.Id)
    $processStartedAt = Get-Date
    $completionSeen = $false
    $stale = $false
    $failureReason = ''
    while ($true) {
        $alive = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        if ($null -eq $alive) {
            break
        }
        if (-not $completionSeen -and (Test-RunComplete $RunDirectory)) {
            $completionSeen = $true
        }
        if (-not $completionSeen -or -not $AllowStaleAfterCompletion) {
            $progressPath = Join-Path $RunDirectory 'training_episodes.csv'
            $lastProgress = if (Test-Path -LiteralPath $progressPath) {
                (Get-Item -LiteralPath $progressPath).LastWriteTime
            } else {
                $processStartedAt
            }
            # A resumed run has an old CSV from before this process started.
            # That file is valid evidence of prior progress, but must not cause
            # the fresh process to be killed immediately.
            if ($lastProgress -lt $processStartedAt) {
                $lastProgress = $processStartedAt
            }
            if (((Get-Date) - $lastProgress).TotalSeconds -gt $staleSeconds) {
                Write-Status ("stale_process label=" + $Label + " pid=" + $process.Id + " last_progress=" + $lastProgress.ToString('o'))
                Stop-ProcessTree $process.Id | Out-Null
                $stale = $true
                $failureReason = 'no_progress'
                break
            }
        }
        if (-not $completionSeen) {
            $descendants = @(Get-DescendantProcesses $process.Id)
            $pythonWorkerCount = @(
                $descendants | Where-Object { $_.Name -ieq 'python.exe' }
            ).Count
            $startupAgeSeconds = ((Get-Date) - $processStartedAt).TotalSeconds
            if (
                $startupAgeSeconds -gt $workerStartupGraceSeconds -and
                $pythonWorkerCount -lt $expectedWorkerCount
            ) {
                Write-Status (
                    "worker_pool_drop label=" + $Label +
                    " pid=" + $process.Id +
                    " workers=" + $pythonWorkerCount +
                    " expected=" + $expectedWorkerCount
                )
                Stop-ProcessTree $process.Id | Out-Null
                $stale = $true
                $failureReason = 'worker_pool_drop'
                break
            }
        }
        Start-Sleep -Seconds 30
    }
    $exitCode = $null
    try {
        $process.WaitForExit()
        $exitCode = $process.ExitCode
    } catch {}
    $orphanedIds = @(Stop-ProcessDescendants $process.Id)
    if ($orphanedIds.Count -gt 0) {
        Write-Status (
            "cleaned_orphaned_descendants label=" + $Label +
            " parent_pid=" + $process.Id +
            " count=" + $orphanedIds.Count
        )
    }
    $exitCodeText = if ($null -eq $exitCode) { 'unknown' } else { [string]$exitCode }
    if ($exitCode -eq -1073741819 -or $exitCode -eq 3221225477) {
        Write-Status (
            "native_access_violation label=" + $Label +
            " pid=" + $process.Id +
            " exit_code=" + $exitCodeText
        )
    }
    if ($stale) {
        Write-Status (
            "attempt_failed label=" + $Label +
            " reason=" + $failureReason +
            " exit_code=" + $exitCodeText
        )
        return [bool]$false
    }
    if (Test-RunComplete $RunDirectory) {
        Write-Status ("completed label=" + $Label + " exit_code=" + $exitCodeText)
        return [bool]$true
    }
    Write-Status ("exited_without_completion label=" + $Label + " exit_code=" + $exitCodeText)
    return [bool]$false
}

function Invoke-Slot(
    [string]$StudyRelativePath,
    [string]$Variant,
    [int]$Seed,
    [ValidateSet('cbf', 'nosafety')]
    [string]$Mode
) {
    $runDirectory = Get-RunDirectory $StudyRelativePath $Variant $Seed
    if (Test-RunComplete $runDirectory) {
        Write-Status ("skip_complete mode=" + $Mode + " variant=" + $Variant + " seed=" + $Seed)
        return
    }
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        if (Test-RunComplete $runDirectory) {
            return
        }
        $force = $false
        $completionArtifactsPresent = (
            (Test-Path -LiteralPath (Join-Path $runDirectory 'model_final.zip')) -or
            (Test-Path -LiteralPath (Join-Path $runDirectory 'training_complete.json'))
        )
        if ($completionArtifactsPresent) {
            Write-Status ("invalid_completion_artifacts mode=" + $Mode + " variant=" + $Variant + " seed=" + $Seed)
            Move-PartialRun $StudyRelativePath $Variant $Seed $(if ($Mode -eq 'cbf') { 'ppo200k_cbf5' } else { 'ppo200k_nosafety5' })
            $force = $true
        }
        $checkpoint = Get-LatestCheckpoint $runDirectory
        if (-not $force -and (Test-Path -LiteralPath $runDirectory)) {
            if ($null -eq $checkpoint) {
                Move-PartialRun $StudyRelativePath $Variant $Seed $(if ($Mode -eq 'cbf') { 'ppo200k_cbf5' } else { 'ppo200k_nosafety5' })
                $force = $true
            } else {
                Write-Status ("resumable_checkpoint mode=" + $Mode + " variant=" + $Variant + " seed=" + $Seed + " checkpoint=" + $checkpoint)
            }
        }
        $label = $Mode + '_' + $Variant + '_seed' + $Seed + '_attempt' + $attempt
        $arguments = New-StudyArguments $StudyRelativePath $Variant $Seed $Mode $force $false
        $succeeded = Invoke-TrackedPython $label $arguments $runDirectory $true
        if ($succeeded) {
            $kpiFiles = @(Get-ChildItem -LiteralPath $runDirectory -Recurse -Filter 'kpi.csv' -File -ErrorAction SilentlyContinue)
            if ($kpiFiles.Count -eq 0) {
                Write-Status ("evaluation_missing mode=" + $Mode + " variant=" + $Variant + " seed=" + $Seed)
                $evalLabel = $label + '_evaluation'
                $evalArgs = New-StudyArguments $StudyRelativePath $Variant $Seed $Mode $false $true
                # The completion marker predates the 400-episode evaluation;
                # do not interpret a quiet training CSV as a hung evaluator.
                $evalSucceeded = Invoke-TrackedPython $evalLabel $evalArgs $runDirectory $true
                if (-not $evalSucceeded) {
                    throw "Evaluation failed for $Mode/$Variant/seed_$Seed; inspect $logRoot"
                }
            }
            return
        }
        Write-Status ("retrying mode=" + $Mode + " variant=" + $Variant + " seed=" + $Seed + " next_attempt=" + ($attempt + 1))
        # Let Windows release terminated workers' private pages and handles
        # before a fresh 20-process pool is created.
        Start-Sleep -Seconds 15
    }
    throw "Exhausted $maxAttempts attempts for $Mode/$Variant/seed_$Seed; inspect $logRoot"
}

function Invoke-Aggregate(
    [string]$StudyRelativePath,
    [string[]]$Variants,
    [int[]]$Seeds,
    [ValidateSet('cbf', 'nosafety')]
    [string]$Mode
) {
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            if (-not (Test-RunComplete (Get-RunDirectory $StudyRelativePath $variant $seed))) {
                throw "Cannot aggregate incomplete run $Mode/$variant/seed_$seed"
            }
        }
    }
    # This pass writes the complete study manifest/config without retraining or
    # duplicating the 200-episode evaluations already saved beside each slot.
    $seedArguments = @()
    foreach ($seed in $Seeds) { $seedArguments += [string]$seed }
    $variantArguments = @()
    foreach ($variant in $Variants) { $variantArguments += $variant }
    $aggregateLabel = if ($Mode -eq 'cbf') { 'ppo200k_cbf5' } else { 'ppo200k_nosafety_baseline5' }
    $arguments = @(
        '-u', 'scripts\run_ppo_cbf_progression.py',
        '--project-root', '.',
        '--output-dir', $StudyRelativePath,
        '--device', 'cpu',
        '--timesteps', '200000',
        '--n-envs', '20',
        '--seeds'
    ) + $seedArguments + @('--variants') + $variantArguments + @(
        '--ppo-config', 'Q1_stable',
        '--n-steps', '1000',
        '--batch-size', '100',
        '--n-epochs', '10',
        '--checkpoint-freq', '25000',
        '--reward-mode', 'reciprocal',
        '--traffic-model', 'mtm',
        '--remove-vehicle-dimensions',
        '--expose-target-y',
        '--task-distance-m', '600',
        '--task-max-policy-steps', '3000',
        '--post-train-eval-episodes', '200',
        '--post-train-eval-workers', '20',
        '--skip-training',
        '--skip-post-train-evaluation',
        '--skip-evaluation',
        '--skip-counterfactual',
        '--tensorboard-run-label', $aggregateLabel
    )
    if ($Mode -eq 'cbf') {
        $arguments += @(
            '--safety-potential-formulation', 'cbf_violation',
            '--safety-cbf-alpha', '1.0',
            '--safety-cbf-psi-scale', '1.0',
            '--safety-potential-weight', '0.0'
        )
    } else {
        $arguments += '--disable-safety-shaping'
    }
    $stdout = Join-Path $logRoot ($Mode + '_aggregate_stdout.log')
    $stderr = Join-Path $logRoot ($Mode + '_aggregate_stderr.log')
    & python.exe @arguments 1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Aggregation failed for $Mode; inspect $stderr"
    }
    Write-Status ("aggregated mode=" + $Mode)
}

$seeds = @(307, 308, 309, 310, 311)
$cbfVariants = @(
    'ppo_nominal',
    'ppo_cbf_reward',
    'ppo_cbf_nd_reward_actor',
    'ppo_cbf_nd_actor_only',
    'ppo_cbf_diff_reward_only',
    'ppo_cbf_integrated_actor_only',
    'ppo_cbf_projected_reward_off'
)

foreach ($seed in $seeds) {
    foreach ($variant in $cbfVariants) {
        Invoke-Slot $cbfStudyRelative $variant $seed 'cbf'
    }
}
Invoke-Aggregate $cbfStudyRelative $cbfVariants $seeds 'cbf'
Set-Content -LiteralPath $successMarker -Value ("completed_at=" + (Get-Date).ToString('o')) -Encoding UTF8
Write-Status ("cbf_success_marker=" + $successMarker)

foreach ($seed in $seeds) {
    Invoke-Slot $noSafetyStudyRelative 'ppo_nominal' $seed 'nosafety'
}
Invoke-Aggregate $noSafetyStudyRelative @('ppo_nominal') $seeds 'nosafety'
Write-Status 'all_studies_complete'
