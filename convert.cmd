@ECHO OFF
python.exe unpk3.py input
python.exe rtcwhq.py input 4 2048 2 2.0 0.0 4 90
python.exe makepk3.py input