Start-Transcript -Path C:\Apps\CryptoML\logs\restart.log -Force | Out-Null
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*start_app.py*' | ForEach-Object { "Stopping PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 3
if (Get-ScheduledTask -TaskName 'CryptoML App' -ErrorAction SilentlyContinue) {
  # ending the task first lets it start cleanly (MultipleInstances = IgnoreNew)
  Stop-ScheduledTask -TaskName 'CryptoML App' -ErrorAction SilentlyContinue
  Start-ScheduledTask -TaskName 'CryptoML App'; "Started scheduled task 'CryptoML App'"
} else {
  Start-Process wscript.exe -ArgumentList '"C:\Apps\CryptoML\launch_silent.vbs"'; "Started via launch_silent.vbs"
}
Start-Sleep 20
try { Invoke-RestMethod http://localhost:8080/api/startup_status -TimeoutSec 5 | ConvertTo-Json } catch { "not up yet: $_" }
Stop-Transcript | Out-Null
