' DepthConverter launcher - opens NO console window at all.
' Double-click this (or pin it) instead of the .bat.
' First run only: setup shows a console so you can watch the downloads.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = root

py = root & "\.venv\Scripts\pythonw.exe"
ff = root & "\tools\ffmpeg\ffmpeg.exe"
If fso.FileExists(py) And fso.FileExists(ff) Then
    sh.Run """" & py & """ -m app.main", 0, False
Else
    sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & _
           root & "\launch.ps1""", 1, False
End If
