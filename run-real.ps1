[CmdletBinding(DefaultParameterSetName='AutoML')]
param(
    [ValidateRange(1, 7200)][int] $Seconds = 600,
    [ValidateRange(1, 7200)][int] $SampleSeconds = 1800,
    [ValidateRange(1, 65535)][int] $Port = 55432,
    [string] $DbUser = 'hack_reader',
    [string] $Task1Database = 'hack_final_task1',
    [string] $Task2Database = 'hack_final_task2',
    [switch] $NonInteractive,
    [Parameter(Mandatory, ParameterSetName='Compare')][switch] $CompareQwen,
    [Parameter(Mandatory, ParameterSetName='QwenCheck')][switch] $QwenCheck,
    [Parameter(Mandatory, ParameterSetName='QwenSmoke')][switch] $QwenSmoke,
    [Parameter(Mandatory, ParameterSetName='QwenTask2')][switch] $QwenTask2
)

$ErrorActionPreference = 'Stop'
$loopback = '127.0.0.1'
$root = $PSScriptRoot
$pythonPath = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$runName = 'run-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), ([Guid]::NewGuid().ToString('N').Substring(0,8))
$tasks = @(
    [pscustomobject]@{ Name='task1'; Database=$Task1Database; Check=$null }
    [pscustomobject]@{ Name='task2'; Database=$Task2Database; Check=$null }
)
if ($QwenTask2) {
    $tasks = @($tasks[1])
    $Seconds = [Math]::Min($Seconds, 600)
}
$batchStartedUtc = [DateTime]::UtcNow
$passwordInput = $null
$qwenNames = @('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL')
function Get-QwenConfiguration([string] $Path) {
    $values = @{}
    $inherited = @{}
    foreach ($name in $qwenNames) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $values[$name] = $value
            $inherited[$name] = $true
        }
    }

    if ($values.Count -lt $qwenNames.Count -and [IO.File]::Exists($Path)) {
        # Treat values as data. In particular, dollar signs and PowerShell syntax remain literal.
        foreach ($line in [IO.File]::ReadLines($Path, [Text.Encoding]::UTF8)) {
            $entry = [regex]::Match($line, '^\s*(?:export\s+)?(OPENAI_API_KEY|OPENAI_BASE_URL|OPENAI_MODEL)\s*=\s*(.*)$')
            if (-not $entry.Success) { continue }
            $name = $entry.Groups[1].Value
            if ($inherited.ContainsKey($name)) { continue }
            $value = $entry.Groups[2].Value.Trim()
            if ($value.StartsWith('"') -or $value.StartsWith("'")) {
                $pattern = if ($value.StartsWith('"')) { '^"((?:[^"\\]|\\.)*)"\s*(?:#.*)?$' }
                           else { "^'((?:[^'\\]|\\.)*)'\s*(?:#.*)?$" }
                $quoted = [regex]::Match($value, $pattern)
                if (-not $quoted.Success) { throw "Некорректные кавычки в .env для $name." }
                $quote = $value.Substring(0, 1)
                $value = $quoted.Groups[1].Value -replace ('\\([' + $quote + '\\])'), '$1'
            } else {
                $value = ($value -replace '\s+#.*$', '').TrimEnd()
            }
            $values[$name] = $value
        }
    }
    foreach ($name in $qwenNames) {
        if ([string]::IsNullOrWhiteSpace($values[$name]) -or $values[$name] -match '[\x00-\x1f\x7f]') {
            throw "Проверьте $name в окружении процесса или в локальном .env."
        }
    }
    return $values
}

$qwenConfig = if ($CompareQwen -or $QwenSmoke -or $QwenCheck -or $QwenTask2) {
    Get-QwenConfiguration (Join-Path $root '.env')
}
$probe = @'
import os, sys
from postgres_loader import pg_error_reason
try:
    import psycopg
    with psycopg.connect(connect_timeout=8) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        database, role, readonly = conn.execute(
            "SELECT current_database(), current_user, current_setting('transaction_read_only')"
        ).fetchone()
        if (database, role, readonly) != (os.environ["PGDATABASE"], os.environ["PGUSER"], "on"):
            raise RuntimeError("Unexpected database, role or read-only transaction setting")
except Exception as exc:
    state = getattr(exc, "sqlstate", None)
    msg = str(exc).strip()
    print(pg_error_reason(exc))
    pgconn = getattr(exc, "pgconn", None)
    try:
        needs_password = bool(pgconn is not None and pgconn.needs_password)
    except Exception:
        needs_password = False
    prompt = needs_password or state == "28P01" or any(s in msg.lower() for s in (
        "no password supplied", "password authentication failed", "password is required",
        "\u043f\u0430\u0440\u043e\u043b"))
    sys.exit(3 if prompt else 2)
print("read-only SELECT succeeded")
'@

function Invoke-TyrellPython([string] $Database, [string] $Arguments, [switch] $ProbeConnection, [switch] $UseQwen, [switch] $ApiOnly) {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $pythonPath
    $start.Arguments = $Arguments
    $start.WorkingDirectory = $root
    $start.UseShellExecute = $false
    $envMap = $start.EnvironmentVariables
    if (-not $ApiOnly) {
        $envMap['PGHOST'] = $loopback
        $envMap['PGHOSTADDR'] = $loopback
        $envMap['PGPORT'] = $Port.ToString()
        $envMap['PGUSER'] = $DbUser
        $envMap['PGDATABASE'] = $Database
    }
    $envMap['TYRELL_MAX_LLM_CALLS'] = '0'
    $envMap['PYTHONIOENCODING'] = 'utf-8'
    $envMap['PYTHONUTF8'] = '1'
    foreach ($name in @($envMap.Keys)) {
        if ($name -like 'OPENAI_*' -or ($ApiOnly -and $name -like 'PG*')) {
            $envMap.Remove($name)
        }
    }
    if ($UseQwen -and -not $ProbeConnection) {
        foreach ($name in $qwenNames) { $envMap[$name] = $qwenConfig[$name] }
        $envMap['TYRELL_MAX_LLM_CALLS'] = if ($QwenTask2) { '2' } elseif ($ApiOnly) { '1' } else { '3' }
    }
    if ($ProbeConnection) {
        $start.RedirectStandardInput = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        $start.StandardOutputEncoding = [Text.Encoding]::UTF8
        $start.StandardErrorEncoding = [Text.Encoding]::UTF8
    }
    $proc = New-Object Diagnostics.Process
    $proc.StartInfo = $start
    try {
        if (-not $ApiOnly -and $null -ne $passwordInput) {
            $secretBuffer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($passwordInput)
            try {
                $envMap['PGPASSWORD'] = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretBuffer)
            } finally {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretBuffer)
            }
        }
        try {
            [void] $proc.Start()
        } finally {
            $envMap.Remove('PGPASSWORD')
            foreach ($name in $qwenNames) { $envMap.Remove($name) }
        }
        $detail = ''
        if ($ProbeConnection) {
            $stdout = $proc.StandardOutput.ReadToEndAsync()
            $stderr = $proc.StandardError.ReadToEndAsync()
            $proc.StandardInput.Write($probe)
            $proc.StandardInput.Close()
            if (-not $proc.WaitForExit(20000)) {
                $proc.Kill()
                $proc.WaitForExit()
                return [pscustomobject]@{ ExitCode=2; Detail='Таймаут проверки подключения PostgreSQL, 20 с.' }
            }
            $detail = ($stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult()).Trim()
        }
        if ($ApiOnly -and -not $proc.WaitForExit(30000)) {
            $proc.Kill()
            $proc.WaitForExit()
            Write-Host 'Истёк лимит времени проверки Qwen (30 с).'
            return [pscustomobject]@{ ExitCode=124; Detail='Qwen check timed out' }
        }
        $proc.WaitForExit()
        [pscustomobject]@{ ExitCode=$proc.ExitCode; Detail=$detail }
    } finally {
        $proc.Dispose()
    }
}

function Test-Tunnel {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $pending = $client.ConnectAsync($loopback, $Port)
        if (-not $pending.Wait(3000)) { throw 'таймаут TCP 3 с' }
        [void] $pending.GetAwaiter().GetResult()
    } finally {
        $client.Dispose()
    }
}

function Save-Failure([string] $Path, [string] $Task, [string] $Status, [string] $Message) {
    $report = [ordered]@{ task=$Task; status=$Status; error=$Message; csv_valid=$false; temporal_cv_score=$null }
    [void] [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
    $utf8 = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, ($report | ConvertTo-Json), $utf8)
    [IO.File]::AppendAllText((Join-Path ([IO.Path]::GetDirectoryName($Path)) 'run.log'),
        ("{0:o} [{1}] {2}{3}" -f [DateTimeOffset]::Now, $Status, $Message, [Environment]::NewLine), $utf8)
}

function Get-ComparisonSnapshot($Report, [string] $RelativePath) {
    [pscustomobject]@{
        report=$RelativePath; metric=$Report.metric; temporal_cv_score=$Report.temporal_cv_score
        baseline_score=$Report.cv_audit.baseline_score; train_rows=$Report.train_rows; test_rows=$Report.test_rows
        experiments=$Report.reported_experiments; training_seconds=$Report.training_seconds_reported
        elapsed_seconds=$Report.elapsed_seconds; peak_rss_mib=$Report.peak_rss_mib; csv_valid=$Report.csv_valid
        time_limit_seconds=$Report.time_limit_seconds; budget_scope=$Report.budget_scope
        solution_budget_seconds=$Report.solution_budget_seconds; sample_seconds=$Report.sample_seconds
        data_load_seconds=$Report.data_load_seconds; membership_signature=$Report.cv_audit.membership_signature
    }
}

function Get-PreviousSuccessfulRun([string] $TaskDir, [DateTime] $BeforeUtc) {
    $files = Get-ChildItem -LiteralPath (Join-Path $root $TaskDir) -Directory -Filter 'run-*' | ForEach-Object {
        $candidate = Join-Path $_.FullName 'report.json'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { Get-Item -LiteralPath $candidate }
    }
    foreach ($file in ($files | Where-Object { $_.LastWriteTimeUtc -le $BeforeUtc } | Sort-Object LastWriteTimeUtc -Descending)) {
        try { $prior = Get-Content -Raw -Encoding UTF8 -LiteralPath $file.FullName | ConvertFrom-Json } catch { continue }
        if ($prior.status -eq 'ok' -and -not $prior.pair_arm -and $prior.research_mode -ne 'qwen_tools' -and $file.Directory.Name -notlike '*-qwen-tools') {
            return Get-ComparisonSnapshot $prior "$TaskDir/$($file.Directory.Name)/report.json"
        }
    }
}

if ($QwenSmoke -or $QwenCheck) {
    $mode = if ($QwenSmoke) { '--smoke' } else { '--check' }
    $result = Invoke-TyrellPython '' "-B -u runs/real-public/qwen_check.py $mode" -UseQwen -ApiOnly
    exit $result.ExitCode
}
if ($CompareQwen -or $QwenTask2) {
    $result = Invoke-TyrellPython '' '-B -u runs/real-public/qwen_check.py --check' -UseQwen -ApiOnly
    if ($result.ExitCode -ne 0) { exit $result.ExitCode }
}

$rows = @()
$comparisons = @()
$paired = @()
$reportPaths = @()
$hasFailure = $false
try {
    $tunnelIssue = $null
    try { Test-Tunnel } catch {
        $tunnelIssue = "TCP ${loopback}:$Port. " + $_.Exception.GetBaseException().Message
    }
    foreach ($task in $tasks) {
        if ($tunnelIssue) {
            $task.Check = [pscustomobject]@{ ExitCode=2; Detail=$tunnelIssue }
        } else {
            $task.Check = Invoke-TyrellPython $task.Database '-B -u -' -ProbeConnection
        }
        Write-Host "[$($task.Name)] $($task.Check.Detail)"
    }
    if (($tasks.Check.ExitCode -contains 3) -and -not $NonInteractive) {
        $passwordInput = Read-Host "Пароль PostgreSQL для $DbUser" -AsSecureString
        foreach ($task in $tasks) {
            $task.Check = Invoke-TyrellPython $task.Database '-B -u -' -ProbeConnection
            Write-Host "[$($task.Name)] $($task.Check.Detail)"
        }
    }
    $jobs = foreach ($task in $tasks) {
        if ($CompareQwen) {
            foreach ($arm in @('automl','qwen')) {
                [pscustomobject]@{ Task=$task; Arm=$arm; Folder="$runName-$arm" }
            }
        } elseif ($QwenTask2) {
            [pscustomobject]@{ Task=$task; Arm='qwen-tools'; Folder="$runName-qwen-tools" }
        } else {
            [pscustomobject]@{ Task=$task; Arm=$null; Folder=$runName }
        }
    }
    foreach ($job in $jobs) {
        $task = $job.Task
        $taskDir = "runs/real-public/$($task.Name)"
        $runDir = "$taskDir/$($job.Folder)"
        $reportPaths += "$runDir/report.json"
        $reportPath = Join-Path $root "$runDir/report.json"
        $result = $null
        try {
            if ($task.Check.ExitCode -ne 0) {
                Save-Failure $reportPath $task.Name 'connection_failed' $task.Check.Detail
            } else {
                Write-Host "[$($task.Name)] $runDir"
                $arguments = "-B -u runs/real-public/diagnose.py $taskDir/task.md --run-dir $runDir --seconds $Seconds --sample-seconds $SampleSeconds"
                if ($QwenTask2) { $arguments += ' --qwen-tools' }
                elseif ($job.Arm) { $arguments += " --pair-arm $($job.Arm)" }
                $result = Invoke-TyrellPython $task.Database $arguments -UseQwen:($job.Arm -in @('qwen','qwen-tools'))
                if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
                    Save-Failure $reportPath $task.Name 'diagnostic_failed' "diagnose.py exit $($result.ExitCode), report.json отсутствует"
                }
            }
            $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $reportPath | ConvertFrom-Json
            $failed = $report.status -ne 'ok' -or ($null -ne $result -and $result.ExitCode -ne 0)
            if ($failed) {
                $hasFailure = $true
                Write-Host "[$($task.Name)] $($report.status) $($report.error)"
                foreach ($issue in @($report.experiment_problems) + @($report.log_findings)) {
                    if ($issue) { Write-Host $issue }
                }
            }
            if (-not $failed -and -not $CompareQwen -and -not $QwenTask2) {
                $before = Get-PreviousSuccessfulRun $taskDir $batchStartedUtc
                $after = Get-ComparisonSnapshot $report "$runDir/report.json"
                $sameMembership = $null
                if ($before.membership_signature -and $after.membership_signature) {
                    $sameMembership = $before.membership_signature -eq $after.membership_signature
                }
                $comparisons += [pscustomobject]@{ task=$task.Name; before=$before; after=$after
                    same_validation_membership=$sameMembership }
            }
            $rows += [pscustomobject]@{
                Task=$task.Name; Arm=$job.Arm; Status=$report.status; Train=$report.train_rows; Test=$report.test_rows
                Sample_s=$report.sample_seconds; Load_s=$report.data_load_seconds; Training_s=$report.training_seconds_reported
                Wall_s=$report.elapsed_seconds; Limit_s=$Seconds; Research_s=$report.research_seconds
                Metric=$report.metric; CV=$report.temporal_cv_score; Baseline=$report.cv_audit.baseline_score; Experiments=$report.reported_experiments
                RSS_MiB=$report.peak_rss_mib; CSV=$report.csv_valid
            }
        } catch {
            $hasFailure = $true
            $detail = $_.Exception.GetBaseException().Message
            Write-Host "[$($task.Name)] $detail"
            if (-not (Test-Path -LiteralPath $reportPath)) {
                Save-Failure $reportPath $task.Name 'runner_failed' $detail
            }
        }
        Write-Host "[$($task.Name)] $reportPath"
    }
    $summaryPath = Join-Path $root "runs/real-public/$runName.json"
    if ($CompareQwen) {
        foreach ($task in $tasks) {
            $taskDir = Join-Path $root "runs/real-public/$($task.Name)"
            $control = Join-Path $taskDir "$runName-automl/report.json"
            $withQwen = Join-Path $taskDir "$runName-qwen/report.json"
            $result = & $pythonPath -B (Join-Path $root 'paired_research.py') $control $withQwen
            if ($LASTEXITCODE -ne 0) { throw "Не удалось сравнить отчёты $($task.Name)" }
            $pair = $result | ConvertFrom-Json
            if (-not $pair.comparable) {
                $hasFailure = $true
                Write-Host "[$($task.Name)] Парное сравнение недоступно: $($pair.reasons -join ', ')"
            } else {
                Write-Host "[$($task.Name)] Изменение качества ($($pair.metric), положительное = улучшение): $($pair.gain)"
            }
            $paired += [pscustomobject]@{ task=$task.Name; comparison=$pair }
        }
    }
    $summary = [ordered]@{ run=$runName; results=$rows; comparisons=$comparisons; reports=$reportPaths }
    if ($CompareQwen) { $summary['paired'] = $paired }
    [IO.File]::WriteAllText($summaryPath, ($summary | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false)))
    Write-Host "Сводка: $summaryPath"
    $columns = @('Task','Status','Train','Test','Sample_s','Load_s','Training_s','Research_s','Wall_s','Limit_s','Metric','CV','Baseline','Experiments','RSS_MiB','CSV')
    if ($CompareQwen -or $QwenTask2) { $columns = @('Task','Arm') + $columns[1..($columns.Count-1)] }
    $rows | Format-Table -Property $columns -AutoSize | Out-String -Width 240 | Write-Host
} finally {
    if ($null -ne $passwordInput) { $passwordInput.Dispose() }
}
if ($hasFailure) { exit 1 }
exit 0
