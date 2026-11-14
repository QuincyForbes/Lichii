# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

<#
.SYNOPSIS
iOS bench from the Windows PC: installs a build on the Mac's iPhone, runs a matrix of bench runs
(baseline + each toggle set, N repeats, interleaved, cool-down between) and prints a comparison.

.EXAMPLE
.\bench.ps1 -Build 43611345294 -Toggles guest_cores_first, oot_ideal_affinity, guest_idle_spin
.\bench.ps1 -Build installed -Toggles "guest_cores_first+guest_idle_spin", "EDEN_NX2_IDLE_SPIN_US=50" -Repeats 2
.\bench.ps1 -Build C:\ipas\Lichii.ipa -Plan plans\cpu-sched.json

.PARAMETER Build
"installed" (keep what is on the phone), a path to an .ipa, an iOS CI run id, an artifact id, or a
commit sha (its newest green ios.yml run).
.PARAMETER DryRun
Write and print the plan only.
.PARAMETER Toggles
Toggle sets after the implicit baseline. In each, "+" joins items; an item with "=" is an
environment value (EDEN_NX2_...=1), anything else an iOS experiment name.
#>
param(
    [string]$Build = "installed",
    [string]$Profile = "kokiri",
    [string[]]$Toggles = @(),
    [string]$Plan,
    [int]$Repeats = 3,
    [int]$CoolDownSec = 240,
    [ValidateSet("stikdebug", "lldb")] [string]$Jit = "stikdebug",
    [switch]$Trace,
    [switch]$PushStage,
    [switch]$NoBaseline,
    [switch]$DryRun,
    [string]$Mac = "admin@100.93.246.33",
    [string]$Key = "$env:USERPROFILE\.ssh\lichii_mac",
    [string]$Repo = "QuincyForbes/Lichii"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$session = Get-Date -Format "yyyyMMdd-HHmmss"
$localRoot = Join-Path $env:TEMP "lichii-bench\device-runs\$(Get-Date -Format yyyy-MM-dd)\$session"
New-Item -ItemType Directory -Force $localRoot | Out-Null
$sshArgs = @("-i", $Key, "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=accept-new")
$remote = "~/lichii-bench"

function Ssh([string]$command) {
    & ssh @sshArgs $Mac $command
    if ($LASTEXITCODE -ne 0) { throw "ssh failed ($LASTEXITCODE): $command" }
}
function Scp([string[]]$from, [string]$to) {
    & scp @sshArgs -r @from $to
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $from -> $to" }
}
function Say([string]$text) { Write-Host "$(Get-Date -Format HH:mm:ss) $text" }

# The IPA to install, or $null for "installed".
function Resolve-Ipa([string]$build) {
    if ($build -eq "installed") { return $null }
    if (Test-Path $build) { return (Resolve-Path $build).Path }
    $dir = Join-Path $localRoot "ipa"
    New-Item -ItemType Directory -Force $dir | Out-Null
    if ($build -match '^[0-9a-f]{7,40}$' -and $build -notmatch '^\d+$') {
        $runs = gh run list -R $Repo --workflow ios.yml -L 200 --json databaseId,headSha,conclusion | ConvertFrom-Json
        $run = $runs | Where-Object { $_.headSha.StartsWith($build) -and $_.conclusion -eq "success" } | Select-Object -First 1
        if (-not $run) { throw "no green ios.yml run for $build" }
        $build = "$($run.databaseId)"
    }
    Say "downloading the IPA of $build"
    gh run download $build -R $Repo -D $dir 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Not a run id: an artifact id.
        $zip = Join-Path $dir "artifact.zip"
        gh api "repos/$Repo/actions/artifacts/$build/zip" > $zip
        if ($LASTEXITCODE -ne 0) { throw "$build is neither a run nor an artifact of $Repo" }
        Expand-Archive -Force $zip $dir
    }
    $ipa = Get-ChildItem $dir -Recurse -Filter "Lichii.ipa" | Select-Object -First 1
    if (-not $ipa) { throw "no Lichii.ipa in $build" }
    return $ipa.FullName
}

# --- the plan
if ($Plan) {
    $planObject = Get-Content (Resolve-Path $Plan) -Raw | ConvertFrom-Json
    $requestPath = Join-Path (Split-Path (Resolve-Path $Plan)) $planObject.request
    $planObject.request = Get-Content $requestPath -Raw | ConvertFrom-Json
} else {
    $request = Get-Content (Join-Path $here "stage\profiles\$Profile.json") -Raw | ConvertFrom-Json
    $cells = @()
    if (-not $NoBaseline) { $cells += [ordered]@{ name = "baseline" } }
    # powershell -File passes a list as one comma-joined string.
    foreach ($set in ($Toggles | ForEach-Object { $_ -split ',' } | Where-Object { $_ })) {
        $experiments = @(); $envValues = [ordered]@{}
        foreach ($item in $set -split '\+') {
            if ($item -match '^([^=]+)=(.*)$') { $envValues[$Matches[1]] = $Matches[2] }
            elseif ($item) { $experiments += $item }
        }
        $cell = [ordered]@{ name = ($set -replace '[^A-Za-z0-9_+=.-]', '_') }
        if ($experiments) { $cell.experiments = $experiments }
        if ($envValues.Count) { $cell.env = $envValues }
        $cells += $cell
    }
    $planObject = [ordered]@{
        request = $request; repeats = $Repeats; cooldown_s = $CoolDownSec; hot_retries = 3
        jit = $Jit; trace_first = [bool]$Trace; timeout_s = 3600; cells = $cells
    }
}
$planFile = Join-Path $localRoot "plan.json"
$planObject | ConvertTo-Json -Depth 20 | Set-Content -Encoding ascii $planFile
Say "session $session, plan $planFile"
if ($DryRun) { Get-Content $planFile; return }

# --- the Mac
Say "checking the Mac ($Mac)"
Ssh "mkdir -p $remote/tools $remote/sessions/$session"
Scp @("$here\bench_mac.py", "$here\bench_report.py", "$here\jit_lldb.py", "$here\make-profile.sh", "$here\setup-mac.sh", "$here\stage") "${Mac}:$remote/tools/"
Scp @($planFile) "${Mac}:$remote/sessions/$session/plan.json"
$ipa = Resolve-Ipa $Build
if ($ipa) {
    Say "installing $ipa"
    Scp @($ipa) "${Mac}:$remote/sessions/$session/Lichii.ipa"
    Ssh "cd $remote && python3 -I tools/bench_mac.py sign --ipa sessions/$session/Lichii.ipa --out sessions/$session/app && python3 -I tools/bench_mac.py install --app sessions/$session/app/Lichii.app"
}
if ($PushStage -or $ipa) {
    Ssh "cd $remote && python3 -I tools/bench_mac.py push --stage tools/stage"
}

# The matrix runs under nohup on the Mac, so a dropped connection does not stop it.
$matrixPid = (& ssh @sshArgs $Mac "cd $remote && nohup python3 -I tools/bench_mac.py matrix --plan sessions/$session/plan.json --out sessions/$session/runs > sessions/$session/matrix.log 2>&1 & echo `$!").Trim()
Say "matrix running on the Mac (pid $matrixPid); log $remote/sessions/$session/matrix.log"
$lastLine = ""
while ($true) {
    Start-Sleep 60
    $state = & ssh @sshArgs $Mac "kill -0 $matrixPid 2>/dev/null && echo running || echo done; tail -1 $remote/sessions/$session/matrix.log"
    if ($LASTEXITCODE -ne 0) { Say "(the Mac did not answer; retrying)"; continue }
    $lines = @($state)
    if ($lines.Count -gt 1 -and $lines[1] -ne $lastLine) { $lastLine = $lines[1]; Say $lastLine }
    if ($lines[0] -eq "done") { break }
}

Say "copying the results to $localRoot"
Scp @("${Mac}:$remote/sessions/$session/runs", "${Mac}:$remote/sessions/$session/matrix.log") $localRoot
$python = (Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
& $python -I (Join-Path $here "bench_report.py") (Join-Path $localRoot "runs") --markdown (Join-Path $localRoot "summary.md") --json (Join-Path $localRoot "summary.json")
Say "summary: $localRoot\summary.md"
