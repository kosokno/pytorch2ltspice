@echo off
choice /c YN /m "Are you sure you want to delete all the simulation artifacts? (Y/N): "
if errorlevel 2 goto end

echo Deleting *.py, *.sp, *.txt, and 'env\__pycache__' and 'tmp' folder...
if exist "env\*.py" del /Q "env\*.py" 
if exist "env\*.sp" del /Q "env\*.sp" 
if exist "env\*.txt" del /Q "env\*.txt" 

if exist "tmp" rmdir /S /Q "tmp"
if exist "env\__pycache__" rmdir /S /Q "env\__pycache__"


echo Deletion complete.
:end
pause
