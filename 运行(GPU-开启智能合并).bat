@echo off
chcp 65001
set cpath=%~dp0
set cpath=%cpath:~0,-1%
"%cpath%\海南鸡饭OPENVINO.exe" --device="GPU" --enable_segment_merge %*
pause