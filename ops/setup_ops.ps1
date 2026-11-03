# One-time setup (run elevated): Waitress, startup + nightly-backup scheduled tasks, then reboot.
$ErrorActionPreference = 'Stop'
$app  = 'C:\Apps\CryptoML'
$log  = "$app\logs\setup_ops.log"
Start-Transcript -Path $log -Force | Out-Null
$user = "$env:USERDOMAIN\$env:USERNAME"
$ok = $true
try {
  "== 1. Install Waitress into the app's venv"
  & "$app\.venv\Scripts\python.exe" -m pip install -q "waitress>=3.0" 2>&1 | Select-Object -Last 3
  & "$app\.venv\Scripts\python.exe" -c "import waitress; print('waitress', waitress.__version__ if hasattr(waitress,'__version__') else 'ok')"

  "== 2. Scheduled task: start the app at boot (no sign-in needed)"
  $act = New-ScheduledTaskAction -Execute 'cmd.exe' -WorkingDirectory $app `
         -Argument "/c cd /d $app && .venv\Scripts\python.exe -u -W ignore start_app.py >> logs\app.log 2>&1"
  $trg = New-ScheduledTaskTrigger -AtStartup
  $trg.Delay = 'PT30S'
  $pr  = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Highest
  $set = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
         -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable -AllowStartIfOnBatteries -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName 'CryptoML App' -Action $act -Trigger $trg -Principal $pr -Settings $set `
         -Description 'Idol Crypto ML dashboard (Waitress, port 8080)' -Force | Out-Null
  "Registered 'CryptoML App' (runs as $user at startup)"

  "== 3. Scheduled task: nightly database backup 02:30"
  $bact = New-ScheduledTaskAction -Execute 'powershell.exe' `
          -Argument "-NoProfile -ExecutionPolicy Bypass -File $app\ops\backup_db.ps1"
  $btrg = New-ScheduledTaskTrigger -Daily -At '02:30'
  $bset = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable
  Register-ScheduledTask -TaskName 'CryptoML Backup' -Action $bact -Trigger $btrg -Principal $pr -Settings $bset `
         -Description 'Nightly compressed + verified backup of CryptoML (7-day weekday rotation)' -Force | Out-Null
  "Registered 'CryptoML Backup' (daily 02:30)"

  "== 4. Run a backup now to prove it works"
  Start-ScheduledTask -TaskName 'CryptoML Backup'
  $deadline = (Get-Date).AddMinutes(3)
  do { Start-Sleep 5 } while ((Get-ScheduledTask -TaskName 'CryptoML Backup').State -eq 'Running' -and (Get-Date) -lt $deadline)
  $bi = Get-ScheduledTaskInfo -TaskName 'CryptoML Backup'
  "Backup task result: $($bi.LastTaskResult)"
  Get-Content "$app\logs\backup.log" -Tail 2
  if ($bi.LastTaskResult -ne 0) { $ok = $false }

  "== 5. Replace the sign-in startup entry with the boot task"
  Remove-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'CryptoMLDashboard' -ErrorAction SilentlyContinue
  'Removed HKCU Run key CryptoMLDashboard (if present)'

  "== 6. Hand the running app over to the scheduled task (Waitress)"
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*start_app.py*' |
    ForEach-Object { "Stopping PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
  Start-Sleep 3
  Start-ScheduledTask -TaskName 'CryptoML App'
  $up = $false
  foreach ($i in 1..18) {
    Start-Sleep 5
    try { $s = Invoke-RestMethod http://localhost:8080/api/startup_status -TimeoutSec 5; $up = $true; "App answering: stage = $($s.stage)"; break } catch {}
  }
  if (-not $up) { "App did NOT answer on 8080 within 90s"; $ok = $false }
  Select-String -Path "$app\logs\app.log" -Pattern 'Waitress' | Select-Object -Last 1 | ForEach-Object Line
} catch {
  "FAILED: $_"; $ok = $false
}

if ($ok) {
  "== 7. All checks passed - rebooting in 60 seconds (driver install asked for it)"
  shutdown.exe /r /t 60 /c "CryptoML setup complete - rebooting to finish ODBC Driver 18 install"
} else {
  "== Not rebooting because a step failed - see above."
}
"DONE"
Stop-Transcript | Out-Null
