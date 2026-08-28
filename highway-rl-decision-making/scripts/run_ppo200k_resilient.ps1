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

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Set-Content -LiteralPath $statusLog -Value ("started_at=" + (Get-Date).ToString('o')) -Encoding UTF8
if (Test-Path -LiteralPath $successMarker) {
    Remove-Item -LiteralPath $successMarker -Force
}

function Write-Status([string]$Message) {
    $line = (Get-Date).ToString('o') + ' ' + $Message
    Add-Content -LiteralPath $statusLog -Value $line -Encoding UTF8
    Write-Output $line
}

function Get-RunDirectory([string]$StudyRelativePath, [string]$Variant, [int]$Seed) {
    return Join-Path (Join-Path (Join-Path $repo $StudyRelativePath) $Variant) ('seed_' + $Seed)
}

function Test-RunComplete([string]$RunDirectory) {
    return (
        (Test-Path -LiteralPath (Join-Path $RunDirectory 'model_final.zip')) -and
        (Test-Path -LiteralPath (Join-Path $RunDirectory 'training_complete.json'))
    )
}

function Get-LatestCheckpoint([string]$RunDirectory) {
    $checkpointDirectory = Join-Path $RunDirectory 'checkpoints'
    if (-not (Test-Path -LiteralPath $checkpointDirectory)) {
        return $null
    }
    $candidates = @()
    foreach ($path in (Get-ChildItem -LiteralPath $checkpointDirectory -Filter 'rollout_*_steps.zip' -File -ErrorAction SilentlyContinue)) {
        if ($path.Name -match '^rollout_(\d+)_steps\.zip$') {
            $candidates += [pscustomobject]@{ step = [int64]$Matches[1]; path = $path.FullName }
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
    $process = Start-Process -FilePath 'python.exe' -ArgumentList $Arguments -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Write-Status ("started label=" + $Label + " pid=" + $process.Id)
    $completionSeen = $false
    $stale = $false
    while ($true) {
        $alive = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        if ($null -eq $alive) {
            break
        }
        if (Test-RunComplete $RunDirectory) {
            $completionSeen = $true
        }
        if (-not $completionSeen -or -not $AllowStaleAfterCompletion) {
            $progressPath = Join-Path $RunDirectory 'training_episodes.csv'
            $lastProgress = if (Test-Path -LiteralPath $progressPath) {
                (Get-Item -LiteralPath $progressPath).LastWriteTime
            } else {
                $process.StartTime
            }
            if (((Get-Date) - $lastProgress).TotalSeconds -gt $staleSeconds) {
                Write-Status ("stale_process label=" + $Label + " pid=" + $process.Id + " last_progress=" + $lastProgress.ToString('o'))
                Stop-ProcessTree $process.Id | Out-Null
                $stale = $true
                break
            }
        }
        Start-Sleep -Seconds 30
    }
    if ($stale) {
        return $false
    }
    if (Test-RunComplete $RunDirectory) {
        Write-Status ("completed label=" + $Label)
        return $true
    }
    Write-Status ("exited_without_completion label=" + $Label)
    return $false
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
        $checkpoint = Get-LatestCheckpoint $runDirectory
        $force = $false
        if (Test-Path -LiteralPath $runDirectory) {
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
