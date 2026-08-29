param(
    [int]$IntervalSeconds = 30,
    [int]$FreeMemoryWarningMB = 2048,
    [int]$ProcessPrivateWarningMB = 28672,
    [int]$IndividualPrivateWarningMB = 2048,
    [int]$HandleWarning = 10000
)

$ErrorActionPreference = 'SilentlyContinue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$monitorRoot = Join-Path $repo 'artifacts\final_Results\health_monitor_ppo200k'
New-Item -ItemType Directory -Path $monitorRoot -Force | Out-Null

$processCsv = Join-Path $monitorRoot 'process_memory.csv'
$systemCsv = Join-Path $monitorRoot 'system_health.csv'
$eventsLog = Join-Path $monitorRoot 'health_events.log'

$processHeaders = @(
    'timestamp', 'role', 'pid', 'parent_pid', 'name', 'private_mb',
    'working_set_mb', 'virtual_mb', 'pagefile_mb', 'paged_pool_mb',
    'nonpaged_pool_mb', 'handle_count', 'thread_count', 'cpu_pct',
    'start_time', 'tracked_process_count', 'tree_private_mb',
    'tree_working_set_mb', 'command_line'
)
$systemHeaders = @(
    'timestamp', 'tracked_process_count', 'tree_private_mb',
    'tree_working_set_mb', 'available_memory_mb', 'total_memory_mb',
    'committed_memory_mb', 'commit_limit_mb', 'commit_pct',
    'pool_nonpaged_mb', 'pool_paged_mb', 'system_cpu_pct',
    'c_drive_free_gb'
)

function Add-CsvRecord([string]$Path, [object]$Record) {
    $lines = @($Record | ConvertTo-Csv -NoTypeInformation)
    if (-not (Test-Path -LiteralPath $Path)) {
        $lines | Set-Content -LiteralPath $Path -Encoding UTF8
    } elseif ($lines.Count -gt 1) {
        $lines | Select-Object -Skip 1 | Add-Content -LiteralPath $Path -Encoding UTF8
    }
}

function Log-Event([string]$Kind, [string]$Message) {
    $line = "{0}`t{1}`t{2}" -f (Get-Date).ToString('o'), $Kind, ($Message -replace '[\r\n\t]+', ' ')
    Add-Content -LiteralPath $eventsLog -Value $line -Encoding UTF8
}

$thresholdState = @{}
function Log-Threshold([string]$Key, [bool]$Condition, [string]$Message) {
    $wasSet = $thresholdState.ContainsKey($Key) -and [bool]$thresholdState[$Key]
    if ($Condition -and -not $wasSet) {
        Log-Event 'warning' $Message
    } elseif (-not $Condition -and $wasSet) {
        Log-Event 'recovered' $Key
    }
    $thresholdState[$Key] = $Condition
}

function Get-TrackedProcessSnapshot {
    $all = @(Get-CimInstance Win32_Process)
    $rootCandidates = @(
        $all | Where-Object {
            $_.CommandLine -match 'run_ppo200k_resilient\.ps1' -or
            $_.CommandLine -match 'run_ppo_cbf_progression\.py'
        }
    )
    $ids = @()
    $frontier = @()
    foreach ($candidate in $rootCandidates) {
        $id = [int]$candidate.ProcessId
        if ($ids -notcontains $id) {
            $ids += $id
            $frontier += $id
        }
    }
    while ($frontier.Count -gt 0) {
        $children = @(
            $all | Where-Object {
                $frontier -contains [int]$_.ParentProcessId
            } | Select-Object -ExpandProperty ProcessId
        )
        $newChildren = @($children | Where-Object { $ids -notcontains [int]$_ })
        if ($newChildren.Count -eq 0) {
            break
        }
        foreach ($child in $newChildren) {
            $ids += [int]$child
        }
        $frontier = @($newChildren | ForEach-Object { [int]$_ })
    }
    return @($all | Where-Object { $ids -contains [int]$_.ProcessId })
}

function Get-SystemHealth {
    $os = Get-CimInstance Win32_OperatingSystem
    $computer = Get-CimInstance Win32_ComputerSystem
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $processor = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor |
        Where-Object { $_.Name -eq '_Total' } | Select-Object -First 1
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"

    $totalMemoryMB = if ($computer.TotalPhysicalMemory) {
        [double]$computer.TotalPhysicalMemory / 1MB
    } else { 0.0 }
    $availableMemoryMB = if ($memory.AvailableMBytes) {
        [double]$memory.AvailableMBytes
    } elseif ($os.FreePhysicalMemory) {
        [double]$os.FreePhysicalMemory / 1024.0
    } else { 0.0 }
    $committedMB = if ($memory.CommittedBytes) {
        [double]$memory.CommittedBytes / 1MB
    } else { 0.0 }
    $commitLimitMB = if ($memory.CommitLimit) {
        [double]$memory.CommitLimit / 1MB
    } else { 0.0 }
    $commitPct = if ($commitLimitMB -gt 0) {
        100.0 * $committedMB / $commitLimitMB
    } else { 0.0 }
    [pscustomobject]@{
        available_memory_mb = [math]::Round($availableMemoryMB, 2)
        total_memory_mb = [math]::Round($totalMemoryMB, 2)
        committed_memory_mb = [math]::Round($committedMB, 2)
        commit_limit_mb = [math]::Round($commitLimitMB, 2)
        commit_pct = [math]::Round($commitPct, 2)
        pool_nonpaged_mb = [math]::Round(([double]$memory.PoolNonpagedBytes / 1MB), 2)
        pool_paged_mb = [math]::Round(([double]$memory.PoolPagedBytes / 1MB), 2)
        system_cpu_pct = [math]::Round(([double]$processor.PercentProcessorTime), 2)
        c_drive_free_gb = [math]::Round(([double]$disk.FreeSpace / 1GB), 2)
    }
}

$previousCpu = @{}
$previousSampleAt = Get-Date
$lastApplicationEventId = 0
$lastSystemEventId = 0
Log-Event 'monitor_started' ("interval_seconds=$IntervalSeconds; free_memory_warning_mb=$FreeMemoryWarningMB; tree_private_warning_mb=$ProcessPrivateWarningMB; individual_private_warning_mb=$IndividualPrivateWarningMB")

while ($true) {
    try {
        $now = Get-Date
        $cimProcesses = @(Get-TrackedProcessSnapshot)
        $trackedRows = @()
        $treePrivateMB = 0.0
        $treeWorkingSetMB = 0.0
        $elapsedSeconds = [math]::Max(0.001, ($now - $previousSampleAt).TotalSeconds)

        foreach ($cimProcess in $cimProcesses) {
            try {
                $process = Get-Process -Id ([int]$cimProcess.ProcessId)
                $privateMB = [double]$process.PrivateMemorySize64 / 1MB
                $workingSetMB = [double]$process.WorkingSet64 / 1MB
                $virtualMB = [double]$process.VirtualMemorySize64 / 1MB
                $pagefileMB = [double]$process.PagedMemorySize64 / 1MB
                $pagedPoolMB = [double]$process.PagedSystemMemorySize64 / 1MB
                $nonpagedPoolMB = [double]$process.NonpagedSystemMemorySize64 / 1MB
                $cpuTotalSeconds = [double]$process.TotalProcessorTime.TotalSeconds
                $cpuPct = 0.0
                if ($previousCpu.ContainsKey([int]$process.Id)) {
                    $cpuPct = 100.0 * ($cpuTotalSeconds - [double]$previousCpu[[int]$process.Id]) / $elapsedSeconds
                    $cpuPct = [math]::Max(0.0, $cpuPct / [math]::Max(1, [Environment]::ProcessorCount))
                }
                $previousCpu[[int]$process.Id] = $cpuTotalSeconds
                $startTime = ''
                try { $startTime = $process.StartTime.ToString('o') } catch {}
                $role = 'worker'
                if ([string]$cimProcess.CommandLine -match 'run_ppo200k_resilient\.ps1') {
                    $role = 'supervisor'
                } elseif ([string]$cimProcess.CommandLine -match 'run_ppo_cbf_progression\.py') {
                    $role = 'learner_or_evaluator'
                }
                $treePrivateMB += $privateMB
                $treeWorkingSetMB += $workingSetMB
                $trackedRows += [pscustomobject]@{
                    role = $role
                    pid = [int]$process.Id
                    parent_pid = [int]$cimProcess.ParentProcessId
                    name = [string]$process.ProcessName
                    private_mb = [math]::Round($privateMB, 2)
                    working_set_mb = [math]::Round($workingSetMB, 2)
                    virtual_mb = [math]::Round($virtualMB, 2)
                    pagefile_mb = [math]::Round($pagefileMB, 2)
                    paged_pool_mb = [math]::Round($pagedPoolMB, 2)
                    nonpaged_pool_mb = [math]::Round($nonpagedPoolMB, 2)
                    handle_count = [int]$process.HandleCount
                    thread_count = [int]$process.Threads.Count
                    cpu_pct = [math]::Round($cpuPct, 2)
                    start_time = $startTime
                    command_line = [string]$cimProcess.CommandLine
                }
            } catch {
                Log-Event 'process_read_error' ("pid=$($cimProcess.ProcessId); $($_.Exception.Message)")
            }
        }

        $system = Get-SystemHealth
        $trackedCount = $trackedRows.Count
        foreach ($row in $trackedRows) {
            Add-CsvRecord $processCsv ([pscustomobject]@{
                timestamp = $now.ToString('o')
                role = $row.role
                pid = $row.pid
                parent_pid = $row.parent_pid
                name = $row.name
                private_mb = $row.private_mb
                working_set_mb = $row.working_set_mb
                virtual_mb = $row.virtual_mb
                pagefile_mb = $row.pagefile_mb
                paged_pool_mb = $row.paged_pool_mb
                nonpaged_pool_mb = $row.nonpaged_pool_mb
                handle_count = $row.handle_count
                thread_count = $row.thread_count
                cpu_pct = $row.cpu_pct
                start_time = $row.start_time
                tracked_process_count = $trackedCount
                tree_private_mb = [math]::Round($treePrivateMB, 2)
                tree_working_set_mb = [math]::Round($treeWorkingSetMB, 2)
                command_line = $row.command_line
            })
        }
        Add-CsvRecord $systemCsv ([pscustomobject]@{
            timestamp = $now.ToString('o')
            tracked_process_count = $trackedCount
            tree_private_mb = [math]::Round($treePrivateMB, 2)
            tree_working_set_mb = [math]::Round($treeWorkingSetMB, 2)
            available_memory_mb = $system.available_memory_mb
            total_memory_mb = $system.total_memory_mb
            committed_memory_mb = $system.committed_memory_mb
            commit_limit_mb = $system.commit_limit_mb
            commit_pct = $system.commit_pct
            pool_nonpaged_mb = $system.pool_nonpaged_mb
            pool_paged_mb = $system.pool_paged_mb
            system_cpu_pct = $system.system_cpu_pct
            c_drive_free_gb = $system.c_drive_free_gb
        })

        Log-Threshold 'low_available_memory' ($system.available_memory_mb -lt $FreeMemoryWarningMB) ("available_memory_mb=$($system.available_memory_mb)")
        Log-Threshold 'high_commit' ($system.commit_pct -ge 90.0) ("commit_pct=$($system.commit_pct)")
        Log-Threshold 'large_process_tree_private' ($treePrivateMB -ge $ProcessPrivateWarningMB) ("tree_private_mb=$([math]::Round($treePrivateMB, 2))")
        $largestPrivateMB = @(
            $trackedRows | ForEach-Object { [double]$_.private_mb }
        ) | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum
        if ($null -eq $largestPrivateMB) { $largestPrivateMB = 0.0 }
        Log-Threshold 'large_individual_private' ([double]$largestPrivateMB -ge $IndividualPrivateWarningMB) ("largest_process_private_mb=$([math]::Round([double]$largestPrivateMB, 2))")
        Log-Threshold 'low_disk_space' ($system.c_drive_free_gb -lt 5.0) ("c_drive_free_gb=$($system.c_drive_free_gb)")
        $largeHandleCount = @($trackedRows | Where-Object { $_.handle_count -ge $HandleWarning }).Count -gt 0
        Log-Threshold 'large_handle_count' $largeHandleCount ("one_or_more_processes_handle_count_ge_$HandleWarning")

        $learnerPresent = @($cimProcesses | Where-Object { $_.CommandLine -match 'run_ppo_cbf_progression\.py' }).Count -gt 0
        Log-Threshold 'learner_missing' ((-not $learnerPresent) -and ($trackedCount -gt 0)) 'supervisor_or_workers_present_but_no_run_ppo_cbf_progression_process'
        Log-Threshold 'no_tracked_processes' ($trackedCount -eq 0) 'no_training_supervisor_or_learner_process_found'

        $eventStart = (Get-Date).AddMinutes(-5)
        $applicationEvents = @(Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $eventStart } -MaxEvents 100)
        foreach ($event in ($applicationEvents | Sort-Object RecordId)) {
            if ([int64]$event.RecordId -le $lastApplicationEventId) { continue }
            $lastApplicationEventId = [int64]$event.RecordId
            $message = [string]$event.Message
            if (($event.ProviderName -match 'Application Error|Windows Error Reporting|\.NET Runtime') -and
                ($message -match 'python|access violation|0xc0000005|0xC0000005|powershell|pycharm|idea64|SearchIndexer')) {
                Log-Event 'windows_application_error' ("provider=$($event.ProviderName); id=$($event.Id); $message")
            }
        }

        # Native application faults can be symptoms of a lower-level storage,
        # PCIe, display-driver, or resource-exhaustion problem.  Capture those
        # system warnings next to the heap samples so any future crash can be
        # correlated without relying on PyCharm remaining alive.
        $systemEvents = @(
            Get-WinEvent -FilterHashtable @{
                LogName = 'System'
                StartTime = $eventStart
                Level = @(1, 2, 3)
            } -MaxEvents 200
        )
        foreach ($event in ($systemEvents | Sort-Object RecordId)) {
            if ([int64]$event.RecordId -le $lastSystemEventId) { continue }
            $lastSystemEventId = [int64]$event.RecordId
            $provider = [string]$event.ProviderName
            $relevant = (
                $provider -match 'WHEA|Disk|Ntfs|stornvme|storahci|Resource-Exhaustion|MemoryDiagnostics|nvlddmkm' -or
                ($provider -match 'Display' -and [int]$event.Id -eq 4101) -or
                ($provider -match 'Kernel-Power' -and [int]$event.Id -eq 41) -or
                ($provider -match 'volmgr|WER-SystemErrorReporting')
            )
            if ($relevant) {
                Log-Event 'windows_system_warning' (
                    "provider=$provider; id=$($event.Id); level=$($event.LevelDisplayName); $($event.Message)"
                )
            }
        }
        $previousSampleAt = $now
    } catch {
        Log-Event 'monitor_error' $_.Exception.Message
    }
    Start-Sleep -Seconds ([math]::Max(5, $IntervalSeconds))
}
