$base = $PSScriptRoot
$endpoint = 'http://127.0.0.1:8765'
function Test-ClearServer {
    try {
        return (Invoke-WebRequest -UseBasicParsing "$endpoint/api/system" -TimeoutSec 1).StatusCode -eq 200
    } catch {
        return $false
    }
}
if (-not (Test-ClearServer)) {
    $python = Join-Path (Split-Path -Parent $base) '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { $python = 'Q:\clear-runtime\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $python)) { throw 'CLEAR runtime was not found.' }
    $script = Join-Path $base 'app.py'
    Start-Process -FilePath $python -ArgumentList @('-u', ('"' + $script + '"')) `
        -WorkingDirectory $base -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $base 'server.out.log') `
        -RedirectStandardError (Join-Path $base 'server.err.log')
    for ($i = 0; $i -lt 12; $i++) {
        if (Test-ClearServer) { break }
        Start-Sleep -Milliseconds 300
    }
    if (-not (Test-ClearServer)) { throw 'See CLEAR-web/server.err.log for startup errors.' }
}
if (-not $env:CLEAR_NO_BROWSER) { Start-Process $endpoint }
