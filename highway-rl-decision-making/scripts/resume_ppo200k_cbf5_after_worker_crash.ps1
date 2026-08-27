$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$study = Join-Path $repo 'artifacts\final_Results\ppo200k_cbf5'
$statusLog = Join-Path $study 'recovery_orchestrator.log'
$successMarker = Join-Path $study 'study_success.marker'

Set-Content -LiteralPath $statusLog -Value ("started_at=" + (Get-Date).ToString('o')) -Encoding UTF8
if (Test-Path -LiteralPath $successMarker) {
    throw "Refusing to use a stale success marker: $successMarker"
}

$commonArgs = @(
    '-u', 'scripts\run_ppo_cbf_progression.py',
    '--project-root', '.',
    '--output-dir', 'artifacts\final_Results\ppo200k_cbf5',
    '--device', 'cuda',
    '--timesteps', '200000',
    '--n-envs', '20',
    '--ppo-config', 'Q1_stable',
    '--n-steps', '1000',
    '--batch-size', '100',
    '--n-epochs', '10',
    '--checkpoint-freq', '100000',
    '--reward-mode', 'reciprocal',
    '--safety-potential-formulation', 'cbf_violation',
    '--safety-cbf-alpha', '1.0',
    '--safety-cbf-psi-scale', '1.0',
    '--safety-potential-weight', '0.0',
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
    '--tensorboard-run-label', 'ppo200k_cbf5'
)

function Invoke-Study([string]$label, [string[]]$studyArgs) {
    $stdout = Join-Path $study ($label + '_stdout.log')
    $stderr = Join-Path $study ($label + '_stderr.log')
    Add-Content -LiteralPath $statusLog -Value ("begin=" + $label + " at=" + (Get-Date).ToString('o')) -Encoding UTF8
    & python.exe @studyArgs 1> $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    Add-Content -LiteralPath $statusLog -Value ("end=" + $label + " at=" + (Get-Date).ToString('o') + " exit_code=" + $exitCode) -Encoding UTF8
    if ($exitCode -ne 0) {
        throw "$label failed with exit code $exitCode; inspect $stderr"
    }
}

$recoverArgs = @($commonArgs) + @(
    '--seeds', '308',
    '--variants', 'ppo_cbf_nd_reward_actor',
    '--force-retrain'
)
Invoke-Study 'retrain_nd_reward_actor_seed308' $recoverArgs

$resumeArgs = @($commonArgs) + @(
    '--seeds', '307', '308', '309', '310', '311',
    '--variants',
    'ppo_nominal',
    'ppo_cbf_reward',
    'ppo_cbf_nd_reward_actor',
    'ppo_cbf_nd_actor_only',
    'ppo_cbf_diff_reward_only',
    'ppo_cbf_integrated_actor_only',
    'ppo_cbf_projected_reward_off'
)
Invoke-Study 'resume_full_cbf_study' $resumeArgs

Set-Content -LiteralPath $successMarker -Value ("completed_at=" + (Get-Date).ToString('o')) -Encoding UTF8
Add-Content -LiteralPath $statusLog -Value ("success_marker=" + $successMarker) -Encoding UTF8
