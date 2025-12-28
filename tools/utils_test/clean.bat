@echo off
choice /c YN /m "Are you sure you want to delete all the simulation artifacts? (Y/N): "
if errorlevel 2 goto end

echo Deleting *.py, *.sp, *.txt, and 'gym\__pycache__' and 'tmp' folder...
if exist "gym\*.py" del /Q "gym\*.py" 
if exist "gym\*.sp" del /Q "gym\*.sp" 
if exist "gym\*.txt" del /Q "gym\*.txt" 
if exist "gym\sig_gen*" del /Q "gym\sig_gen*" 

if exist "tmp" rmdir /S /Q "tmp"
if exist "gym\__pycache__" rmdir /S /Q "gym\__pycache__"


echo Deletion complete.
:end
pause
