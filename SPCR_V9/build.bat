@echo off
REM ============================================================================
REM  TITAN AEGIS - build script for the ternary upscaler (Windows / MinGW)
REM ============================================================================

set GPP="C:\winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r6\mingw64\bin\g++.exe"

if not exist %GPP% (
    echo ERROR: g++ not found at %GPP%
    echo Edit the GPP path in this script if your compiler is installed elsewhere.
    pause
    exit /b 1
)

echo Compiling aegis_upscaler.cpp ...
%GPP% -O3 -fopenmp -std=c++17 -o aegis_upscaler.exe aegis_upscaler.cpp

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED. Make sure stb_image.h and stb_image_write.h are in this
    echo same folder as aegis_upscaler.cpp.
    pause
    exit /b 1
)

echo.
echo Build succeeded: aegis_upscaler.exe
echo.
echo Usage:
echo   aegis_upscaler.exe aegis_weights.bin input.png output.png [scale1] [scale2^|auto] [sharpen]
echo.
echo   scale2 defaults to "auto" (brightness/contrast calibration).
echo   sharpen defaults to 0.8 (unsharp-mask post-process, 0 = off).
echo.
echo Example (recommended defaults):
echo   aegis_upscaler.exe aegis_weights.bin freefire_360p.jpg freefire_1080p.png
echo.
echo Example (manual scale, stronger sharpen):
echo   aegis_upscaler.exe aegis_weights.bin freefire_360p.jpg freefire_1080p.png 0.08 0.06 1.2
echo.
pause
