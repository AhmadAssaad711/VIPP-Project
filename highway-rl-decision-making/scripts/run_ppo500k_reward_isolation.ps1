$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$archiveStudyRoot = Join-Path $repo 'artifacts\final_Results\ppo500k_reward_isolation'
$stageStudyRoot = if ($env:HIGHWAY_RL_STAGE_ROOT) {
    [IO.Path]::GetFullPath($env:HIGHWAY_RL_STAGE_ROOT)
} else {
    'C:\agv_runs\ppo500k_reward_isolation'
}
$studyRoot = $stageStudyRoot
$sessionId = (Get-Date).ToString('yyyyMMdd_HHmmss') + '_' + $PID
$logRoot = Join-Path $studyRoot ('_runlogs\' + $sessionId)
$statusLog = Join-Path $studyRoot 'study_status.log'
$targetTimesteps = 500000
$trainingSeed = 307
$expectedWorkerCount = 20
$workerStartupTimeoutSeconds = 600
$workerDropGraceSeconds = 180
$minimumAvailableMemoryMB = 8192
$maximumCommitPercent = 85.0
$maxAttempts = 6
$staleSeconds = 900
$requiredEvaluationProtocolVersion = 2
$pythonExe = if ($env:HIGHWAY_RL_PYTHON) {
    [IO.Path]::GetFullPath($env:HIGHWAY_RL_PYTHON)
} else {
    'C:\Program Files\Python39\python.exe'
}
if ($studyRoot.Contains(' ')) {
    throw "Local staging root must not contain spaces because Start-Process passes the short training argv directly: $studyRoot"
}

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
    [Environment]::SetEnvironmentVariable([string]$entry.Key, [string]$entry.Value, [EnvironmentVariableTarget]::Process)
}

New-Item -ItemType Directory -Path $studyRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType Directory -Path $archiveStudyRoot -Force | Out-Null

function Write-Status([string]$Message) {
    $line = (Get-Date).ToString('o') + ' ' + $Message
    Add-Content -LiteralPath $statusLog -Value $line -Encoding UTF8
    Write-Host $line
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

function Stop-ProcessTree([int]$ParentProcessId) {
    $allIds = @($ParentProcessId)
    $frontier = @($ParentProcessId)
    $snapshot = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    while ($frontier.Count -gt 0) {
        $children = @(
            $snapshot |
                Where-Object { $frontier -contains [int]$_.ParentProcessId } |
                Select-Object -ExpandProperty ProcessId
        )
        $newChildren = @(
            $children | Where-Object {
                $allIds -notcontains [int]$_ -and [int]$_ -ne $PID
            }
        )
        if ($newChildren.Count -eq 0) {
            break
        }
        $allIds += $newChildren
        $frontier = @($newChildren | ForEach-Object { [int]$_ })
    }
    foreach ($processId in ($allIds | Sort-Object -Descending)) {
        if ([int]$processId -ne $PID) {
            Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
        }
    }
}

function Stop-ProcessDescendants([int]$ParentProcessId) {
    $descendants = @(Get-DescendantProcesses $ParentProcessId)
    foreach ($processId in ($descendants.ProcessId | Sort-Object -Descending)) {
        if ([int]$processId -ne $PID) {
            Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
        }
    }
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
    $commitPercent = if ($commitLimit -gt 0.0) { 100.0 * $committedBytes / $commitLimit } else { 0.0 }
    return [pscustomobject]@{
        available_mb = [math]::Round($availableMB, 2)
        commit_percent = [math]::Round($commitPercent, 2)
    }
}

function Get-ProcessResourceSnapshot([int]$ProcessId) {
    $ids = @($ProcessId) + @(
        (Get-DescendantProcesses $ProcessId).ProcessId |
            ForEach-Object { [int]$_ }
    )
    $processes = @($ids | ForEach-Object {
        Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue
    })
    $privateMB = ($processes | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1MB
    $workingSetMB = ($processes | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB
    $headroom = Get-SystemHeadroom
    return [pscustomobject]@{
        process_count = $processes.Count
        private_mb = [math]::Round([double]$privateMB, 1)
        working_set_mb = [math]::Round([double]$workingSetMB, 1)
        available_mb = $headroom.available_mb
        commit_percent = $headroom.commit_percent
    }
}

function Wait-ForSystemHeadroom([string]$Label) {
    while ($true) {
        $headroom = Get-SystemHeadroom
        if ([double]$headroom.available_mb -ge $minimumAvailableMemoryMB -and [double]$headroom.commit_percent -lt $maximumCommitPercent) {
            return
        }
        Write-Status ("waiting_for_memory_headroom label=" + $Label + " available_mb=" + $headroom.available_mb + " commit_percent=" + $headroom.commit_percent)
        Start-Sleep -Seconds 30
    }
}

function Get-RunDirectory([string]$OutputDir) {
    return Join-Path (Join-Path $OutputDir 'ppo_nominal') ('seed_' + $trainingSeed)
}

function Get-RunLastProgress([string]$OutputDir, [datetime]$Fallback) {
    $runDir = Get-RunDirectory $OutputDir
    if (-not (Test-Path -LiteralPath $runDir -PathType Container)) {
        return $Fallback
    }
    $files = @(Get-ChildItem -LiteralPath $runDir -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match '^(training_progress.*\.json|training_episodes.*\.csv|r_.*_steps\.zip|model_final\.zip|training_complete\.json)$'
    })
    if ($files.Count -eq 0) {
        return $Fallback
    }
    return ($files | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime
}

function Get-LiveWorkerCount([int]$ParentProcessId) {
    $descendants = @(Get-DescendantProcesses $ParentProcessId)
    $liveWorkers = @($descendants | Where-Object {
        $_.Name -ieq 'python.exe' -and
        $_.CommandLine -match 'multiprocessing\.spawn|spawn_main' -and
        $null -ne (Get-Process -Id ([int]$_.ProcessId) -ErrorAction SilentlyContinue)
    })
    return $liveWorkers.Count
}

function Get-LogTail([string]$Path, [int]$MaximumCharacters = 5000) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return ''
    }
    $tail = ((Get-Content -LiteralPath $Path -Tail 80 -ErrorAction SilentlyContinue) -join ' | ')
    if ($tail.Length -gt $MaximumCharacters) {
        return $tail.Substring($tail.Length - $MaximumCharacters)
    }
    return $tail
}

function Test-RunComplete([string]$OutputDir) {
    $runDir = Get-RunDirectory $OutputDir
    $modelPath = Join-Path $runDir 'model_final.zip'
    $completionPath = Join-Path $runDir 'training_complete.json'
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $completionPath -PathType Leaf)) {
        return $false
    }
    try {
        $completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
        return ([int]$completion.num_timesteps -eq $targetTimesteps -and
            (Get-Item -LiteralPath $modelPath).Length -gt 0)
    } catch {
        return $false
    }
}

function Test-EvaluationComplete([string]$OutputDir) {
    $runDir = Get-RunDirectory $OutputDir
    $summaryPath = Join-Path $OutputDir 'post_train_200ep_kpis.csv'
    $manifestPath = Join-Path $runDir 'pe\m.json'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        return ([bool]$manifest.complete -and
            [int]$manifest.evaluation_protocol_version -ge $requiredEvaluationProtocolVersion -and
            [bool]$manifest.cbf_off_geometry_bypassed -and
            (Get-Item -LiteralPath $summaryPath).Length -gt 0)
    } catch {
        return $false
    }
}

function Copy-DirectoryContents([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        return
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($item in @(Get-ChildItem -LiteralPath $Source -Force)) {
        $target = Join-Path $Destination $item.Name
        Copy-Item -LiteralPath $item.FullName -Destination $target -Recurse -Force
    }
}

function Sync-VariantFromArchive([string]$Name, [string]$StageOutput) {
    if (Test-RunComplete $StageOutput) {
        return
    }
    $archiveVariant = Join-Path $archiveStudyRoot $Name
    if (-not (Test-Path -LiteralPath $archiveVariant -PathType Container)) {
        return
    }
    $stageVariant = Join-Path $StageOutput 'ppo_nominal'
    if (-not (Test-Path -LiteralPath $stageVariant -PathType Container)) {
        Write-Status ("restoring_archive_variant name=" + $Name + " source=" + $archiveVariant)
        # The archive variant already contains the ppo_nominal directory;
        # restore its contents into the experiment root, not into a second
        # nested ppo_nominal directory.
        Copy-DirectoryContents $archiveVariant $StageOutput
    }
}

function Publish-Variant([string]$Name, [string]$StageOutput) {
    if (-not (Test-RunComplete $StageOutput) -or -not (Test-EvaluationComplete $StageOutput)) {
        throw "Refusing to archive incomplete variant $Name"
    }
    $archiveVariant = Join-Path $archiveStudyRoot $Name
    Copy-DirectoryContents (Join-Path $StageOutput 'ppo_nominal') $archiveVariant
    $archiveSummary = Join-Path $archiveStudyRoot 'post_train_200ep_kpis.csv'
    $stageSummary = Join-Path $StageOutput 'post_train_200ep_kpis.csv'
    if (Test-Path -LiteralPath $stageSummary -PathType Leaf) {
        $existingRows = if (Test-Path -LiteralPath $archiveSummary -PathType Leaf) {
            @(Import-Csv -LiteralPath $archiveSummary)
        } else {
            @()
        }
        $newRows = @(Import-Csv -LiteralPath $stageSummary)
        $mergedRows = @($existingRows + $newRows)
        if ($mergedRows.Count -gt 0) {
            $mergedRows = @(
                $mergedRows |
                    Group-Object -Property {
                        "{0}|{1}|{2}|{3}" -f $_.training_seed, $_.variant, $_.mode, $_.KPI
                    } |
                    ForEach-Object { $_.Group | Select-Object -Last 1 }
            )
            $mergedRows | Export-Csv -LiteralPath $archiveSummary -NoTypeInformation -Encoding UTF8
        }
    }
    Write-Status ("published_variant name=" + $Name + " archive=" + $archiveVariant)
}

function Test-PinnedRuntime {
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        throw "Pinned Python interpreter does not exist: $pythonExe"
    }
    $runtimeProbe = @'
import importlib.metadata as md
import json
import sys
import numpy
import torch
import stable_baselines3
print(json.dumps({
    'python': '.'.join(str(x) for x in sys.version_info[:3]),
    'numpy': numpy.__version__,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'stable_baselines3': md.version('stable-baselines3'),
}))
'@
    $runtimeOutput = @(& $pythonExe -c $runtimeProbe 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Pinned runtime probe failed: $($runtimeOutput -join ' ')"
    }
    try {
        $runtime = ($runtimeOutput -join [Environment]::NewLine).Trim() | ConvertFrom-Json
    } catch {
        throw "Pinned runtime probe did not return JSON: $($runtimeOutput -join ' ')"
    }
    $expected = @{
        python = '3.9.13'
        numpy = '1.26.4'
        torch = '2.8.0+cu128'
        stable_baselines3 = '2.7.1'
    }
    foreach ($key in $expected.Keys) {
        if ([string]$runtime.$key -ne $expected[$key]) {
            throw "Pinned runtime mismatch for $key`: expected $($expected[$key]), got $($runtime.$key)"
        }
    }
    $runtimeRecord = [ordered]@{
        schema_version = 1
        session_id = $sessionId
        interpreter = $pythonExe
        working_directory = $repo
        probed_at = (Get-Date).ToString('o')
        packages = $runtime
    }
    ($runtimeRecord | ConvertTo-Json -Depth 10) | Set-Content -LiteralPath (Join-Path $studyRoot 'runtime_manifest.json') -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $studyRoot 'runtime_manifest.json') -Destination (Join-Path $archiveStudyRoot 'runtime_manifest.json') -Force
    Write-Status ("runtime_pinned python=" + $runtime.python + " numpy=" + $runtime.numpy + " torch=" + $runtime.torch + " sb3=" + $runtime.stable_baselines3)
}

function New-StudyArguments(
    [string]$OutputDir,
    [ValidateSet('cbf_reward', 'safety_potential', 'no_safety')]
    [string]$RewardVariant,
    [bool]$EvaluationOnly,
    [string]$AttemptId
) {
    $label = 'ppo500k_reward_isolation_' + $RewardVariant
    $arguments = @(
        '-u', 'scripts\run_ppo_cbf_progression.py',
        '--project-root', '.',
        # The default staging root has no spaces, so each argv remains literal
        # and cannot be split by Start-Process command-line reconstruction.
        '--output-dir', $OutputDir,
        '--run-attempt-id', $AttemptId,
        '--device', 'cpu',
        '--timesteps', ([string]$targetTimesteps),
        '--n-envs', ([string]$expectedWorkerCount),
        '--seeds', ([string]$trainingSeed),
        '--variants', 'ppo_nominal',
        '--ppo-config', 'Q1_stable',
        '--n-steps', '1000',
        '--batch-size', '100',
        '--n-epochs', '10',
        '--checkpoint-freq', '25000',
        '--reward-mode', 'reciprocal',
        '--correction-epsilon', '0.03',
        '--traffic-model', 'mtm',
        '--remove-vehicle-dimensions',
        '--expose-target-y',
        '--task-distance-m', '600',
        '--task-max-policy-steps', '3000',
        '--post-train-eval-episodes', '200',
        '--post-train-eval-workers', '20',
        '--post-train-eval-seed-start', '1200000',
        '--skip-evaluation',
        '--skip-counterfactual',
        '--tensorboard-run-label', $label
    )
    if ($RewardVariant -eq 'cbf_reward') {
        $arguments += @(
            '--safety-potential-formulation', 'cbf_violation',
            '--safety-cbf-alpha', '1.0',
            '--safety-cbf-psi-scale', '1.0',
            '--safety-potential-weight', '0.0'
        )
    } elseif ($RewardVariant -eq 'safety_potential') {
        # In reciprocal mode this is the legacy potential-field slot wf*cf.
        # The direct CBF formulation is explicitly off for this control arm.
        $arguments += @(
            '--safety-potential-formulation', 'none',
            '--safety-potential-weight', '0.0'
        )
    } else {
        $arguments += '--disable-safety-shaping'
    }
    if ($EvaluationOnly) {
        $arguments += @('--skip-training', '--post-train-evaluate-reused')
    }
    return $arguments
}

function Invoke-TrackedPython(
    [string]$Label,
    [string[]]$Arguments,
    [string]$OutputDir
) {
    $stdout = Join-Path $logRoot ($Label + '_stdout.log')
    $stderr = Join-Path $logRoot ($Label + '_stderr.log')
    $resultPath = Join-Path $logRoot ($Label + '_result.json')
    Wait-ForSystemHeadroom $Label
    Write-Status ("launching label=" + $Label + " python=" + $pythonExe + " output=" + $OutputDir)
    $process = Start-Process -FilePath $pythonExe -ArgumentList $Arguments -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $processId = [int]$process.Id
    Write-Status ("started label=" + $Label + " pid=" + $processId)
    $processStartedAt = Get-Date
    $completionSeen = $false
    $failureReason = ''
    $workerPoolSeen = $false
    $workerDropStartedAt = $null
    $lastResourceLogAt = [datetime]::MinValue
    $exitCode = $null
    while ($true) {
        $alive = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $alive) {
            try {
                $process.Refresh()
                $exitCode = $process.ExitCode
            } catch {
                $exitCode = $null
            }
            break
        }
        if (-not $completionSeen -and (Test-RunComplete $OutputDir)) {
            $completionSeen = $true
            Write-Status ("training_completion_seen label=" + $Label + " pid=" + $processId)
        }
        if (-not $completionSeen) {
            $lastProgress = Get-RunLastProgress $OutputDir $processStartedAt
            if (((Get-Date) - $lastProgress).TotalSeconds -gt $staleSeconds) {
                Write-Status ("stale_process label=" + $Label + " pid=" + $processId)
                $failureReason = 'no_progress'
                Stop-ProcessTree $processId
                break
            }
            $workerCount = Get-LiveWorkerCount $processId
            if ($workerCount -ge $expectedWorkerCount) {
                if (-not $workerPoolSeen) {
                    Write-Status ("worker_pool_ready label=" + $Label + " pid=" + $processId + " workers=" + $workerCount)
                }
                $workerPoolSeen = $true
                $workerDropStartedAt = $null
            } elseif (-not $workerPoolSeen) {
                $startupAge = ((Get-Date) - $processStartedAt).TotalSeconds
                if ($startupAge -gt $workerStartupTimeoutSeconds) {
                    Write-Status ("worker_pool_startup_timeout label=" + $Label + " pid=" + $processId + " workers=" + $workerCount + " expected=" + $expectedWorkerCount)
                    $failureReason = 'worker_pool_startup_timeout'
                    Stop-ProcessTree $processId
                    break
                }
                if ([int]$startupAge % 60 -lt 15) {
                    Write-Status ("worker_pool_starting label=" + $Label + " pid=" + $processId + " workers=" + $workerCount + " expected=" + $expectedWorkerCount)
                }
            } else {
                if ($null -eq $workerDropStartedAt) {
                    $workerDropStartedAt = Get-Date
                    Write-Status ("worker_pool_drop_observed label=" + $Label + " pid=" + $processId + " workers=" + $workerCount)
                } elseif (((Get-Date) - $workerDropStartedAt).TotalSeconds -gt $workerDropGraceSeconds) {
                    Write-Status ("worker_pool_drop_timeout label=" + $Label + " pid=" + $processId + " workers=" + $workerCount + " expected=" + $expectedWorkerCount)
                    $failureReason = 'worker_pool_drop_timeout'
                    Stop-ProcessTree $processId
                    break
                }
            }
        }
        if (((Get-Date) - $lastResourceLogAt).TotalSeconds -ge 60) {
            $snapshot = Get-ProcessResourceSnapshot $processId
            Write-Status ("resources label=" + $Label + " pid=" + $processId + " processes=" + $snapshot.process_count + " private_mb=" + $snapshot.private_mb + " working_set_mb=" + $snapshot.working_set_mb + " available_mb=" + $snapshot.available_mb + " commit_percent=" + $snapshot.commit_percent)
            $lastResourceLogAt = Get-Date
        }
        Start-Sleep -Seconds 15
    }
    try {
        if (-not $process.HasExited) {
            $process.WaitForExit(10000)
        }
        $process.Refresh()
        $exitCode = $process.ExitCode
    } catch {
        # A force-terminated process can release its handle before .ExitCode is readable.
        if ($null -eq $exitCode) {
            $exitCode = $null
        }
    }
    Stop-ProcessDescendants $processId
    $exitCodeText = if ($null -eq $exitCode) { 'unknown' } else { [string]$exitCode }
    $stderrTail = Get-LogTail $stderr
    $stdoutTail = Get-LogTail $stdout
    $result = [ordered]@{
        schema_version = 1
        label = $Label
        pid = $processId
        exit_code = $exitCode
        failure_reason = if ($failureReason) { $failureReason } else { $null }
        completion_seen = $completionSeen
        output_dir = $OutputDir
        stdout_path = $stdout
        stderr_path = $stderr
        stdout_tail = $stdoutTail
        stderr_tail = $stderrTail
        finished_at = (Get-Date).ToString('o')
    }
    ($result | ConvertTo-Json -Depth 10) | Set-Content -LiteralPath $resultPath -Encoding UTF8
    if ($failureReason -ne '') {
        Write-Status ("attempt_failed label=" + $Label + " reason=" + $failureReason + " exit_code=" + $exitCodeText + " stderr_tail=" + $stderrTail)
        return [bool]$false
    }
    if (Test-RunComplete $OutputDir) {
        Write-Status ("completed label=" + $Label + " exit_code=" + $exitCodeText)
        return [bool]$true
    }
    Write-Status ("exited_without_completion label=" + $Label + " exit_code=" + $exitCodeText + " stderr_tail=" + $stderrTail)
    return [bool]$false
}

function Invoke-RewardVariant(
    [string]$Name,
    [string]$OutputDir,
    [ValidateSet('cbf_reward', 'safety_potential', 'no_safety')]
    [string]$RewardVariant
) {
    Sync-VariantFromArchive $Name $OutputDir
    if ((Test-RunComplete $OutputDir) -and (Test-EvaluationComplete $OutputDir)) {
        Write-Status ("skip_complete reward_variant=" + $Name)
        return
    }
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        if (-not (Test-RunComplete $OutputDir)) {
            $attemptId = $Name + '_train_a' + $attempt + '_' + $sessionId
            $label = $Name + '_train_attempt' + $attempt
            $arguments = New-StudyArguments $OutputDir $RewardVariant $false $attemptId
            $succeeded = Invoke-TrackedPython $label $arguments $OutputDir
            if (-not $succeeded) {
                if ($attempt -eq $maxAttempts) {
                    throw "Exhausted training attempts for reward variant $Name"
                }
                Write-Status ("retrying reward_variant=" + $Name + " next_attempt=" + ($attempt + 1))
                Start-Sleep -Seconds 15
                continue
            }
        }
        if (-not (Test-EvaluationComplete $OutputDir)) {
            $attemptId = $Name + '_eval_a' + $attempt + '_' + $sessionId
            $label = $Name + '_evaluation_attempt' + $attempt
            $arguments = New-StudyArguments $OutputDir $RewardVariant $true $attemptId
            $evalSucceeded = Invoke-TrackedPython $label $arguments $OutputDir
            if (-not $evalSucceeded -or -not (Test-EvaluationComplete $OutputDir)) {
                if ($attempt -eq $maxAttempts) {
                    throw "Exhausted evaluation attempts for reward variant $Name"
                }
                Write-Status ("evaluation_retrying reward_variant=" + $Name + " next_attempt=" + ($attempt + 1))
                Start-Sleep -Seconds 15
                continue
            }
        }
        Publish-Variant $Name $OutputDir
        Write-Status ("reward_variant_complete name=" + $Name)
        return
    }
    throw "Exhausted attempts for reward variant $Name"
}

$experiments = @(
    [pscustomobject]@{
        name = 'cbf_reward'
        output = Join-Path $studyRoot 'cbf_reward'
        archive_output = Join-Path $archiveStudyRoot 'cbf_reward'
        reward_variant = 'cbf_reward'
    }
    [pscustomobject]@{
        name = 'safety_potential'
        output = Join-Path $studyRoot 'safety_potential'
        archive_output = Join-Path $archiveStudyRoot 'safety_potential'
        reward_variant = 'safety_potential'
    }
    [pscustomobject]@{
        name = 'no_safety'
        output = Join-Path $studyRoot 'no_safety'
        archive_output = Join-Path $archiveStudyRoot 'no_safety'
        reward_variant = 'no_safety'
    }
)

Test-PinnedRuntime
$studySpec = [ordered]@{
    study = 'ppo500k_reward_isolation'
    purpose = 'paired nominal PPO comparison of CBF violation, legacy safety potential, and no safety shaping'
    session_id = $sessionId
    stage_root = $studyRoot
    archive_root = $archiveStudyRoot
    python_interpreter = $pythonExe
    training_seed = $trainingSeed
    timesteps_per_policy = $targetTimesteps
    training_workers = $expectedWorkerCount
    evaluation_episodes_per_mode = 200
    evaluation_workers = 20
    evaluation_seed_start = 1200000
    action_side_cbf_during_training = $false
    cbf_off_geometry_bypassed = $true
    shared_settings = @{
        ppo_variant = 'ppo_nominal'
        ppo_config = 'Q1_stable'
        reward_mode = 'reciprocal'
        traffic_model = 'mtm'
        task_distance_m = 600
        task_max_policy_steps = 3000
        checkpoint_frequency = 25000
        collision_penalty_retained = $true
        jerk_term_retained = $true
        policy_frequency_matches_cbf_frequency = $true
    }
    experiments = $experiments
}
($studySpec | ConvertTo-Json -Depth 10) | Set-Content -LiteralPath (Join-Path $studyRoot 'study_spec.json') -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $studyRoot 'study_spec.json') -Destination (Join-Path $archiveStudyRoot 'study_spec.json') -Force
Set-Content -LiteralPath $statusLog -Value ("started_at=" + (Get-Date).ToString('o') + " session_id=" + $sessionId) -Encoding UTF8
Write-Status ("stage_root=" + $studyRoot + " archive_root=" + $archiveStudyRoot)
Write-Status 'sequential_training_begin'
foreach ($experiment in $experiments) {
    Write-Status ("starting_reward_variant name=" + $experiment.name + " output=" + $experiment.output)
    Invoke-RewardVariant $experiment.name $experiment.output $experiment.reward_variant
    Copy-Item -LiteralPath $statusLog -Destination (Join-Path $archiveStudyRoot 'study_status.log') -Force
}
Set-Content -LiteralPath (Join-Path $studyRoot 'study_success.marker') -Value (Get-Date).ToString('o') -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $studyRoot 'study_success.marker') -Destination (Join-Path $archiveStudyRoot 'study_success.marker') -Force
Copy-Item -LiteralPath $statusLog -Destination (Join-Path $archiveStudyRoot 'study_status.log') -Force
Write-Status 'study_complete'
