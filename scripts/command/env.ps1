$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")

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

$venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
	& $venvActivate
}
else {
	Write-Warning "未找到 $venvActivate"
}