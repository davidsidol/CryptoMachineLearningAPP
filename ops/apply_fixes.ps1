# Stop the app, re-score this week's outcomes with the corrected logic, start the app again.
$app = 'C:\Apps\CryptoML'
Start-Transcript -Path "$app\logs\apply_fixes.log" -Force | Out-Null
"== Stopping app"
Stop-ScheduledTask -TaskName 'CryptoML App' -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*start_app.py*' |
  ForEach-Object { "Stopping PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep 3
"== Re-scoring outcomes"
& "$app\.venv\Scripts\python.exe" "$app\ops\rescore.py" 2>&1
"== Starting app"
Start-ScheduledTask -TaskName 'CryptoML App'
$up = $false
foreach ($i in 1..24) {
  Start-Sleep 5
  try { $s = Invoke-RestMethod http://localhost:8080/api/startup_status -TimeoutSec 5; $up = $true; "App answering: $($s.stage)"; break } catch {}
}
if (-not $up) { "App did NOT answer within 2 minutes"; Get-Content "$app\logs\app.log" -Tail 15 }
"DONE"
Stop-Transcript | Out-Null
