' 博研助手 · 静默启动器
' 双击本文件即可打开图形界面，不会弹出任何黑色命令行窗口。
' 原理：用 pythonw.exe（无控制台版 Python）启动，且以隐藏窗口方式运行。

Option Explicit

Dim sh, fso, here
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = here

On Error Resume Next
sh.Run "pythonw -m wrc.gui", 0, False
If Err.Number <> 0 Then
    Err.Clear
    ' 有些环境没有把 pythonw 加进 PATH，退回用 python 启动（仍隐藏窗口）
    sh.Run "python -m wrc.gui", 0, False
    If Err.Number <> 0 Then
        MsgBox "启动失败。请确认已安装 Python 3.11+，并在项目目录下执行过：" & vbCrLf & _
               "    pip install -e ." & vbCrLf & vbCrLf & _
               "排查用命令：python -m wrc.cli doctor", 48, "博研助手"
    End If
End If
