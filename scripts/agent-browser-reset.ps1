param(
    [int]$TimeoutSeconds = 20,
    [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"

function Invoke-AgentBrowserCommand {
    param([string[]]$AgentArgs)

    $command = Get-Command agent-browser -ErrorAction SilentlyContinue
    if (!$command) {
        Write-Warning "agent-browser is not installed or not on PATH."
        return 127
    }

    $process = Start-Process -FilePath $command.Source -ArgumentList $AgentArgs -WindowStyle Hidden -PassThru
    if (!$process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        return 124
    }
    return $process.ExitCode
}

Write-Host "Closing agent-browser sessions..."
$closeCode = Invoke-AgentBrowserCommand -AgentArgs @("close", "--all")
if ($closeCode -eq 124) {
    Write-Warning "agent-browser close timed out; continuing with helper-process cleanup."
}

Get-CimInstance Win32_Process |
    Where-Object {
        $_.ProcessId -ne $PID -and
        (
            $_.Name -match "^agent-browser" -or
            ($_.CommandLine -and $_.CommandLine -match "agent-browser")
        )
    } |
    ForEach-Object {
        Write-Host "Stopping agent-browser helper process $($_.ProcessId) $($_.Name)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

if (!$SkipDoctor) {
    Write-Host "Running agent-browser doctor --fix..."
    $doctorCode = Invoke-AgentBrowserCommand -AgentArgs @("doctor", "--fix")
    if ($doctorCode -eq 124) {
        Write-Warning "agent-browser doctor timed out."
    } elseif ($doctorCode -ne 0 -and $doctorCode -ne 127) {
        Write-Warning "agent-browser doctor exited with code $doctorCode."
    }
}

Write-Host "agent-browser reset complete."
