# Nightly CryptoML backup: one compressed, checksummed file per weekday (7-day rotation)
$log  = 'C:\Apps\CryptoML\logs\backup.log'
$inst = '.\IDOLML'
$day  = (Get-Date).DayOfWeek.ToString().Substring(0,3)
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
try {
  $dir = (& sqlcmd -S $inst -E -C -h -1 -W -Q "SET NOCOUNT ON; SELECT CAST(SERVERPROPERTY('InstanceDefaultBackupPath') AS nvarchar(400))").Trim()
  $file = Join-Path $dir "CryptoML_$day.bak"
  $out = & sqlcmd -S $inst -E -C -b -Q "BACKUP DATABASE [CryptoML] TO DISK=N'$file' WITH INIT, FORMAT, COMPRESSION, CHECKSUM; RESTORE VERIFYONLY FROM DISK=N'$file' WITH CHECKSUM;" 2>&1
  if ($LASTEXITCODE -ne 0) { throw ($out -join ' ') }
  $size = [math]::Round((Get-Item $file).Length / 1MB, 1)
  Log "OK  $file  ${size} MB  (verified)"
} catch {
  Log "FAILED  $_"
  exit 1
}
