# One-time setup (run elevated): Waitress, startup + nightly-backup scheduled tasks, then reboot.
# Tasks run as the built-in SYSTEM account (no password, no sign-in needed); SYSTEM is
# granted access to the CryptoML database for Windows authentication.
$ErrorActionPreference = 'Stop'
$app  = 'C:\Apps\CryptoML'
$inst = '.\IDOLML'
Start-Transcript -Path "$app\logs\setup_ops.log" -Force | Out-Null
$ok = $true
function Wait-Task($name, $minutes) {
  $before = (Get-ScheduledTaskInfo -TaskName $name).LastRunTime
  Start-ScheduledTask -TaskName $name
  $deadline = (Get-Date).AddMinutes($minutes)
  do {
    Start-Sleep 5
    $info = Get-ScheduledTaskInfo -TaskName $name
    $state = (Get-ScheduledTask -TaskName $name).State
  } while ((($info.LastRunTime -eq $before) -or ($state -eq 'Running')) -and (Get-Date) -lt $deadline)
  return $info
}
try {
  "== 1. Waitress in the app's venv"
  & "$app\.venv\Scripts\python.exe" -m pip install -q "waitress>=3.0" 2>&1 | Select-Object -Last 3
  & "$app\.venv\Scripts\python.exe" -c "import waitress; print('waitress ok')"

  "== 2. Give SYSTEM access to the CryptoML database"
  & sqlcmd -S $inst -E -C -b -Q @"
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = N'NT AUTHORITY\SYSTEM')
    CREATE LOGIN [NT AUTHORITY\SYSTEM] FROM WINDOWS;
USE [CryptoML];
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'NT AUTHORITY\SYSTEM')
    CREATE USER [NT AUTHORITY\SYSTEM] FOR LOGIN [NT AUTHORITY\SYSTEM];
ALTER ROLE db_owner ADD MEMBER [NT AUTHORITY\SYSTEM];
"@
  if ($LASTEXITCODE) { throw 'SQL grant failed' }
  'SYSTEM is db_owner on CryptoML'

  $pr  = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

  "== 3. Task 'CryptoML App' (at boot)"
  $act = New-ScheduledTaskAction -Execute 'cmd.exe' -WorkingDirectory $app `
         -Argument "/c cd /d $app && .venv\Scripts\python.exe -u -W ignore start_app.py >> logs\app.log 2>&1"
  $trg = New-ScheduledTaskTrigger -AtStartup
  $trg.Delay = 'PT30S'
  $set = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
         -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable -AllowStartIfOnBatteries -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName 'CryptoML App' -Action $act -Trigger $trg -Principal $pr -Settings $set `
         -Description 'Idol Crypto ML dashboard (Waitress, port 8080)' -Force | Out-Null
  "Registered 'CryptoML App' as SYSTEM"

  "== 4. Task 'CryptoML Backup' (daily 02:30)"
  $bact = New-ScheduledTaskAction -Execute 'powershell.exe' `
          -Argument "-NoProfile -ExecutionPolicy Bypass -File $app\ops\backup_db.ps1"
  $btrg = New-ScheduledTaskTrigger -Daily -At '02:30'
  $bset = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable
  Register-ScheduledTask -TaskName 'CryptoML Backup' -Action $bact -Trigger $btrg -Principal $pr -Settings $bset `
         -Description 'Nightly compressed + verified backup of CryptoML (7-day weekday rotation)' -Force | Out-Null
  "Registered 'CryptoML Backup' as SYSTEM"

  "== 5. Test backup"
  $bi = Wait-Task 'CryptoML Backup' 4
  "Backup task result: $($bi.LastTaskResult)"
  Get-Content "$app\logs\backup.log" -Tail 2 -ErrorAction SilentlyContinue
  if ($bi.LastTaskResult -ne 0) { $ok = $false; throw 'Backup task did not succeed' }

  "== 6. Remove the sign-in startup entry"
  Remove-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'CryptoMLDashboard' -ErrorAction SilentlyContinue
  'Removed HKCU Run key CryptoMLDashboard (if present)'

  "== 7. Hand the running app over to the boot task"
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*start_app.py*' |
    ForEach-Object { "Stopping PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep 3
  Start-ScheduledTask -TaskName 'CryptoML App'
  $up = $false
  foreach ($i in 1..24) {
    Start-Sleep 5
    try { $s = Invoke-RestMethod http://localhost:8080/api/startup_status -TimeoutSec 5; $up = $true; "App answering: stage = $($s.stage)"; break } catch {}
  }
  if (-not $up) { $ok = $false; "App did NOT answer on 8080 within 2 minutes"; Get-Content "$app\logs\app.log" -Tail 15 }
  else { Select-String -Path "$app\logs\app.log" -Pattern 'Waitress' | Select-Object -Last 1 | ForEach-Object Line }
} catch {
  "FAILED: $_"; $ok = $false
}

if ($ok) {
  "== 8. All checks passed - rebooting in 60 seconds (ODBC Driver 18 install asked for it)"
  shutdown.exe /r /t 60 /c "CryptoML setup complete - rebooting to finish ODBC Driver 18 install"
} else {
  "== Not rebooting because a step failed - see above."
  if (-not (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*start_app.py*')) {
    "Restarting the app the old way so the dashboard stays up"
    Start-Process wscript.exe -ArgumentList "`"$app\launch_silent.vbs`""
  }
}
"DONE"
Stop-Transcript | Out-Null
