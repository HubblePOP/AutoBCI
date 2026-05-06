# Windows + 多 Provider 运行说明

## 结论

Windows 版的正式入口是 Python 包里的 `autobci`，不是 `open_autoresearch_console.sh`。模型厂商通过 provider 配置切换，业务命令不随 provider 改名。

## 安装

在 Windows 11 x64 PowerShell 里执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install_windows.ps1
```

脚本会检查 Python、Node.js、npm，创建 `.venv`，安装 Python 依赖、AutoBci editable 包和 `tools/autoresearch` 的 Node 依赖，然后跑：

```powershell
.\.venv\Scripts\python.exe -m bci_autoresearch.product_shell.cli doctor --json
.\.venv\Scripts\python.exe -m bci_autoresearch.product_shell.cli provider test fake
```

## Provider 配置

默认配置路径：

- Windows：`%APPDATA%\AutoBci\providers.toml`
- macOS / Linux：`~/.config/autobci/providers.toml`

可用环境变量覆盖：

```powershell
$env:AUTOBCI_PROVIDER_CONFIG = "C:\Users\<you>\AppData\Roaming\AutoBci\providers.toml"
$env:AUTOBCI_DEFAULT_PROVIDER = "deepseek"
$env:AUTOBCI_DEFAULT_MODEL = "deepseek-chat"
```

API key 放在环境变量里，不写进配置和运行日志：

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:KIMI_API_KEY = "..."
$env:GLM_API_KEY = "..."
$env:MINIMAX_API_KEY = "..."
```

## 验收路径

第一版 Windows 验收按这条链路：

```powershell
autobci doctor --json
autobci provider list
autobci provider test fake
autobci dashboard
autobci
```

然后在 `autobci` 里完成一次：描述研究问题、生成 ProgramMD 草案、确认冻结、启动 smoke 级运行、查看 status/report、archive/resume。

## 边界

DeepSeek、Kimi、GLM、MiniMax 的第一阶段含义是：能配置、能 provider-test、能跑 JSON/工具循环 smoke。不同厂商在长周期代码编辑、工具调用稳定性和恢复语义上不承诺等价，实际能力由 provider benchmark 决定。
