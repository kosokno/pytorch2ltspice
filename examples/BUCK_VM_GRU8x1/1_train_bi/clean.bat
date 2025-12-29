@echo off
choice /c YN /m "Are you sure you want to delete all the simulation artifacts? (Y/N): "
if errorlevel 2 goto end

echo Deleting *.pth, *.sp, *.txt, and the 'gym' folder...
if exist "*.pth" del /Q "*.pth"
if exist "*.sp" del /Q "*.sp"
if exist "*.txt" del /Q "*.txt"
if exist "*.net" del /Q "*.net"



:: Delete the 'gym' folder
if exist "gym" attrib -r -h -s gym\*.* /S /D
if exist "gym" rmdir /S /Q gym

echo Deletion complete.
:end
pause
