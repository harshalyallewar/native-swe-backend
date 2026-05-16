$outputFile = "project_dump.txt"

$ignoreDirs = @(
    'venv', 'env', '__pycache__', 'node_modules', 'dist', 'build',
    'coverage', 'logs', 'tmp', 'temp', 'out', 'target', 'bin', 'obj', 'tests', 'test', 'static'
)

$ignoreExts = @(
    '.pyc', '.pyo', '.pyd', '.log', '.lock', '.sqlite', '.db',
    '.dll', '.exe', '.so', '.dylib', '.class', '.jar',
    '.zip', '.tar', '.gz', '.7z',
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.mp3', '.wav', '.pdf'
)

$ignoreFiles = @(
    'project_dump.txt', 'uv.lock', 'poetry.lock', 'package-lock.json',
    'yarn.lock', 'pnpm-lock.yaml', '.DS_Store', 'Thumbs.db',
    '.env', '.env.local', '.env.production', '.env.development',
    'coverage.xml', 'pytest.xml'
)

$projectRoot = (Get-Location).Path

"" | Out-File -FilePath $outputFile -Encoding utf8
"=" * 80 | Out-File -FilePath $outputFile -Encoding utf8
"PROJECT DUMP: $projectRoot" | Out-File -FilePath $outputFile -Append -Encoding utf8
"Generated: $(Get-Date)" | Out-File -FilePath $outputFile -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $outputFile -Append -Encoding utf8
"" | Out-File -FilePath $outputFile -Append -Encoding utf8

$files = Get-ChildItem -Path . -Recurse -File | Where-Object {
    $excluded = $false
    $relativePath = $_.FullName.Substring($projectRoot.Length + 1)
    $parts = $relativePath.Split([System.IO.Path]::DirectorySeparatorChar)

    for ($i = 0; $i -lt $parts.Length - 1; $i++) {
        if ($parts[$i].StartsWith('.') -or $ignoreDirs -contains $parts[$i]) {
            $excluded = $true
            break
        }
    }

    if (-not $excluded -and $ignoreExts -contains $_.Extension.ToLower()) {
        $excluded = $true
    }

    if (-not $excluded -and $ignoreFiles -contains $_.Name) {
        $excluded = $true
    }

    -not $excluded
}

"DIRECTORY STRUCTURE:" | Out-File -FilePath $outputFile -Append -Encoding utf8
"-" * 40 | Out-File -FilePath $outputFile -Append -Encoding utf8
foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($projectRoot.Length + 1)
    $relativePath | Out-File -FilePath $outputFile -Append -Encoding utf8
}
"" | Out-File -FilePath $outputFile -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $outputFile -Append -Encoding utf8
"" | Out-File -FilePath $outputFile -Append -Encoding utf8

foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($projectRoot.Length + 1)

    "" | Out-File -FilePath $outputFile -Append -Encoding utf8
    "+" * 80 | Out-File -FilePath $outputFile -Append -Encoding utf8
    "FILE: $relativePath" | Out-File -FilePath $outputFile -Append -Encoding utf8
    "+" * 80 | Out-File -FilePath $outputFile -Append -Encoding utf8
    "" | Out-File -FilePath $outputFile -Append -Encoding utf8

    try {
        $content = Get-Content -Path $file.FullName -Raw -ErrorAction Stop
        $content | Out-File -FilePath $outputFile -Append -Encoding utf8
    } catch {
        "[BINARY OR UNREADABLE FILE]" | Out-File -FilePath $outputFile -Append -Encoding utf8
    }

    "" | Out-File -FilePath $outputFile -Append -Encoding utf8
}

Write-Host "Done! Output saved to: $outputFile" -ForegroundColor Green
Write-Host "Total files included: $($files.Count)" -ForegroundColor Cyan