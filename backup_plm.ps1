# Path to Git repository
$repoPath = "D:\PLM"
Set-Location $repoPath

# Git executable
$git = "git"

# Stage all changes except ignored files
& $git add .

# Commit only if there are changes
$changes = & $git status --porcelain
if ($changes) {
    & $git commit -m "Daily backup $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    & $git push origin master
} else {
    Write-Output "No changes to commit."
}
