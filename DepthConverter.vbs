' DepthConverter launcher - opens NO console window at all.
' Double-click this (or the DepthConverter shortcut it keeps up to date).
' First run only: setup shows a console so you can watch the downloads.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = root

py = root & "\.venv\Scripts\pythonw.exe"
ff = root & "\tools\ffmpeg\ffmpeg.exe"

If fso.FileExists(py) And fso.FileExists(ff) Then
    RefreshShortcut sh, root, py
    ' pythonw.exe has no console at all; 0 = hidden, False = don't wait.
    sh.Run """" & py & """ -m app.main", 0, False
Else
    sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & _
           root & "\launch.ps1""", 1, False
End If

' Keep DepthConverter.lnk pointing at THIS folder. A shortcut carried over from
' an older location points at a path that no longer exists, so double-clicking
' it does nothing and people fall back to a launcher that shows a console.
Sub RefreshShortcut(sh, root, py)
    Set sc = sh.CreateShortcut(root & "\DepthConverter.lnk")
    If sc.TargetPath <> py Then
        sc.TargetPath = py
        sc.Arguments = "-m app.main"
        sc.WorkingDirectory = root
        sc.IconLocation = root & "\app\assets\icon.ico,0"
        sc.Description = "DepthConverter - local 2D to 3D VR"
        sc.Save
    End If
End Sub
