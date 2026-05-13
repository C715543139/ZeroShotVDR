$ErrorActionPreference = "Stop"

$condaHook = $null
try {
	$condaHook = & conda shell.powershell hook 2>$null | Out-String
}
catch {
	$condaHook = $null
}

if (-not [string]::IsNullOrWhiteSpace($condaHook)) {
	Invoke-Expression $condaHook
	conda activate zeroshotvdr
}
else {
	Write-Warning "无法初始化 conda shell hook，继续尝试仅激活项目 .venv。"
}

$venvActivate = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
& $venvActivate