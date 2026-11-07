param([int]$TunnelPid,[long]$StartTicks,[int]$Seconds=21600)
$ErrorActionPreference = 'Stop'
$deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
while ([DateTime]::UtcNow -lt $deadline) {
    $targetProcess = Get-Process -Id $TunnelPid -ErrorAction SilentlyContinue
    if ($null -eq $targetProcess -or $targetProcess.StartTime.ToUniversalTime().Ticks -ne $StartTicks) { exit 0 }
    Start-Sleep -Seconds 60
}
$targetProcess = Get-Process -Id $TunnelPid -ErrorAction SilentlyContinue
if ($null -ne $targetProcess -and $targetProcess.StartTime.ToUniversalTime().Ticks -eq $StartTicks) {
    $details = Get-CimInstance Win32_Process -Filter "ProcessId=$TunnelPid"
    if ($targetProcess.ProcessName -eq 'ssh' -and $details.CommandLine.Contains('127.0.0.1:17971:127.0.0.1:7897')) {
        Stop-Process -Id $TunnelPid
    }
}
