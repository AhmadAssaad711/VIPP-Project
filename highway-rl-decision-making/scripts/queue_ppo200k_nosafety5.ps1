param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPid,
    [string]$StudyRelativePath = 'artifacts\final_Results\ppo200k_nosafety5',
    [string]$SuccessMarkerRelativePath = 'artifacts\final_Results\ppo200k_cbf5\study_success.marker'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$study = Join-Path $repo $StudyRelativePath
if (Test-Path -LiteralPath $study) {
    $allowed_queue_logs = @(
        'queue.log',
        'queue_process_stdout.log',
        'queue_process_stderr.log'
    )
    $existing_items = @(Get-ChildItem -LiteralPath $study -Force)
    $unexpected_items = @(
        $existing_items | Where-Object { $allowed_queue_logs -notcontains $_.Name }
    )
    if ($unexpected_items.Count -gt 0) {
        throw "Refusing to overwrite non-empty study directory: $study"
    }
} else {
    New-Item -ItemType Directory -Path $study -Force | Out-Null
}
$queueLog = Join-Path $study 'queue.log'
Set-Content -LiteralPath $queueLog -Value ("queued_at=" + (Get-Date).ToString('o') + "`nwaiting_for_pid=$WaitForPid") -Encoding UTF8

while ($null -ne (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 15
}

if (-not (Test-Path -LiteralPath (Join-Path $repo $SuccessMarkerRelativePath))) {
    Add-Content -LiteralPath $queueLog -Value ("upstream_study_success_marker_missing=" + (Get-Date).ToString('o')) -Encoding UTF8
    exit 1
}

Add-Content -LiteralPath $queueLog -Value ("started_at=" + (Get-Date).ToString('o')) -Encoding UTF8
$stdout = Join-Path $study 'study_stdout.log'
$stderr = Join-Path $study 'study_stderr.log'
$pythonArgs = @(
    '-u', 'scripts\run_ppo_cbf_progression.py',
    '--project-root', '.',
    '--output-dir', $StudyRelativePath,
    '--device', 'cuda',
    '--timesteps', '200000',
    '--n-envs', '20',
    '--seeds', '307', '308', '309', '310', '311',
    '--variants',
    'ppo_nominal',
    '--ppo-config', 'Q1_stable',
    '--n-steps', '1000',
    '--batch-size', '100',
    '--n-epochs', '10',
    '--checkpoint-freq', '100000',
    '--reward-mode', 'reciprocal',
    '--disable-safety-shaping',
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
    '--tensorboard-run-label', 'ppo200k_nosafety_baseline5',
    '--force-retrain'
)

& python.exe @pythonArgs 1> $stdout 2> $stderr
$exitCode = $LASTEXITCODE
Add-Content -LiteralPath $queueLog -Value ("finished_at=" + (Get-Date).ToString('o') + "`nexit_code=$exitCode") -Encoding UTF8
exit $exitCode
