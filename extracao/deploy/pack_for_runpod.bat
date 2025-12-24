@echo off
REM =============================================================================
REM Pack for RunPod - Cria arquivo ZIP para upload (Windows)
REM =============================================================================
REM Uso: Abra o terminal na pasta extracao e execute:
REM   deploy\pack_for_runpod.bat
REM =============================================================================

echo Criando pacote para RunPod...

cd /d "%~dp0\.."

REM Nome do arquivo com timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set OUTPUT_FILE=rag-pipeline-%TIMESTAMP%.zip

REM Usar PowerShell para criar ZIP
powershell -Command "& {Compress-Archive -Path 'src', 'scripts', 'data\*.pdf', 'deploy', 'pyproject.toml' -DestinationPath '%OUTPUT_FILE%' -Force}"

echo.
echo ==============================================
echo Pacote criado: %OUTPUT_FILE%
echo ==============================================
echo.
echo Para fazer upload no RunPod:
echo   1. Acesse o RunPod Web Terminal
echo   2. Use o File Browser para upload
echo   3. Ou use: scp %OUTPUT_FILE% root@pod-ip:/workspace/
echo.
pause
