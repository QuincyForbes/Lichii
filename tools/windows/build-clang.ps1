# SPDX-License-Identifier: GPL-3.0-or-later

# Local Windows release build with clang-cl, lld-link and ThinLTO for AVX2 (x86-64-v3) CPUs, the
# configuration the windows-x86_64 release job builds. -Profile adds -fprofile-use with an
# MSVC-ABI LLVM profile such as lichii-windows.profdata; a profile older than the source only
# warns. Needs the LLVM that install-llvm.ps1 pins (it reads only its own profile format), Visual
# Studio Build Tools for the headers, runtime and SDK, Qt for MSVC and glslangValidator on PATH.
# Without admin rights, install LLVM with $env:__COMPAT_LAYER = 'RunAsInvoker' set.

param(
    [string]$Source = (Resolve-Path "$PSScriptRoot\..\..").Path,
    [string]$Build = "$Source\build\clang",
    [string]$LlvmRoot = $(if ($env:LLVM_ROOT) { $env:LLVM_ROOT } else { 'C:\Program Files\LLVM' }),
    [string]$QtDir = 'C:\Qt\6.8.3\msvc2022_64',
    [string]$VsDir = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools',
    [string]$Profile = '',
    [string]$Arch = '-arch:AVX2 -clang:-march=x86-64-v3',
    [string]$ExtraFlags = '',
    [string]$Tag = 'local',
    [int]$Parallel = [Environment]::ProcessorCount,
    [switch]$NoLto,
    [switch]$ConfigureOnly
)
$ErrorActionPreference = 'Stop'

Import-Module "$VsDir\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $VsDir -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
# Native tools report through exit codes; their stderr (CMake warnings) is not a failure.
$ErrorActionPreference = 'Continue'
$env:PATH = "$QtDir\bin;$VsDir\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$VsDir\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:PATH"

$llvm = $LlvmRoot.Replace('\', '/')
$lto = if ($NoLto) { '' } else { '-flto=thin' }
$flags = "-DWIN32 -D_WINDOWS -DNOMINMAX $lto $Arch $ExtraFlags" -replace '\s+', ' '
if ($Profile) {
    if (-not (Test-Path $Profile)) { throw "profile not found: $Profile" }
    # CMake splits its flags at spaces, so a path with spaces goes in by its 8.3 short name.
    $path = (New-Object -ComObject Scripting.FileSystemObject).GetFile((Resolve-Path $Profile).Path).ShortPath
    if ($path.Contains(' ')) { throw "profile path has spaces and no short name: $path" }
    $flags += " -fprofile-use=$($path.Replace('\', '/'))"
    $flags += ' -Wno-error=profile-instr-out-of-date -Wno-error=profile-instr-unprofiled -Wno-error=backend-plugin'
}

& cmake -S $Source -B $Build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_TESTING=OFF `
    -DYUZU_TESTS=OFF `
    -DYUZU_CMD=ON `
    -DYUZU_ROOM=OFF `
    -DYUZU_ROOM_STANDALONE=OFF `
    -DYUZU_USE_BUNDLED_QT=OFF `
    -DYUZU_USE_BUNDLED_SDL3=ON `
    "-DCMAKE_PREFIX_PATH=$QtDir" `
    -DENABLE_UPDATE_CHECKER=OFF `
    "-DLICHII_BUILD_TAG=$Tag" `
    -DLICHII_BUILD_PLATFORM=windows-x86_64 `
    -DUSE_DISCORD_PRESENCE=OFF `
    "-DCMAKE_C_COMPILER=$llvm/bin/clang-cl.exe" `
    "-DCMAKE_CXX_COMPILER=$llvm/bin/clang-cl.exe" `
    "-DCMAKE_LINKER=$llvm/bin/lld-link.exe" `
    "-DCMAKE_AR=$llvm/bin/llvm-lib.exe" `
    "-DCMAKE_C_FLAGS=$flags" `
    "-DCMAKE_CXX_FLAGS=-EHsc $flags"
if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }
if ($ConfigureOnly) { return }

& cmake --build $Build --target yuzu yuzu-cmd --parallel $Parallel
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
Get-ChildItem "$Build\bin" -Filter '*.exe' | Select-Object Name, Length, LastWriteTime
