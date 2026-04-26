@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SETVARS=%ONEAPI_SETVARS%"
if not defined SETVARS set "SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

rem ---------------------------------------------------------------------------
rem Lab PC has oneAPI 2025.1+ which ships ifx only (classic ifort was retired).
rem Abaqus 2024's win86_64.env has been patched to call ifx; we just need PATH.
rem ---------------------------------------------------------------------------

rem VS 2026 Community is detected via VS2022INSTALLDIR override because Intel's
rem setvars.bat does not yet know about VS 18. Harmless if the path is wrong.
if not defined VS2022INSTALLDIR set "VS2022INSTALLDIR=C:\Program Files\Microsoft Visual Studio\18\Community"

if not exist "%SETVARS%" (
    echo ERROR: Intel oneAPI setvars.bat was not found.
    echo Checked: "%SETVARS%"
    echo Set ONEAPI_SETVARS to the correct path and rerun.
    exit /b 1
)

echo [1/4] Activating Intel compiler environment (ifx)...
rem Guard against repeated activation: each call to setvars.bat appends ~3KB
rem to PATH and the 8191-char limit is reached after 2-3 re-runs, corrupting
rem the shell. If ifx is already on PATH, skip the call entirely.
where ifx >nul 2>nul
if not errorlevel 1 (
    echo       Intel environment already active in this shell ^(ifx found^); skipping setvars.
) else (
    call "%SETVARS%" intel64 vs2022
    if errorlevel 1 (
        echo ERROR: Failed to activate Intel compiler environment.
        exit /b 1
    )
)

rem Activate VS 2026 MSVC toolchain (setvars vs2022 detects VS 18 via the
rem VS2022INSTALLDIR override above, but we also call vcvars64 to be safe).
if exist "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" (
    call "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)

where ifx >nul 2>nul
if errorlevel 1 (
    echo ERROR: ifx is not available in PATH after setvars.bat.
    echo Check that Intel oneAPI HPC Toolkit / Fortran Compiler is installed.
    exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: cl.exe ^(MSVC^) is not available in PATH.
    echo Install Visual Studio with the "Desktop development with C++" workload.
    exit /b 1
)

where abaqus >nul 2>nul
if errorlevel 1 (
    echo ERROR: abaqus is not available in PATH.
    echo Launch this from an Abaqus command prompt or add Abaqus to PATH.
    exit /b 1
)

echo [2/4] Skipping `abaqus verify -user_std` (Abaqus 2024 verify probe is
echo       hard-wired to look for ifort.exe and fails on ifx-only systems even
echo       when real UMAT jobs compile and link correctly).

if "%~1"=="" (
    set DEFAULT_ARGS=--angles 0 45 90 --compare-exp --cleanup-transients
) else (
    set DEFAULT_ARGS=%*
)

echo [3/4] Launching standalone Yld2000 UMAT runner...
abaqus python "%SCRIPT_DIR%run_yld2000_umat.py" %DEFAULT_ARGS%
set EXITCODE=%ERRORLEVEL%

echo [4/4] Finished.
echo Summary: "%SCRIPT_DIR%output\yld2000_umat_run_summary.txt"
exit /b %EXITCODE%