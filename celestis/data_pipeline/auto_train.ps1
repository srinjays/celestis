# ============================================================
# CELESTIS - Automated Dataset Expansion + PINN Retraining
# Runs every 65 minutes to stay under Space-Track quota
# Usage: .\data_pipeline\auto_train.ps1    Stop: Ctrl+C
# ============================================================
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$PYTHON = "C:\Users\AJINKYA SINGH\AppData\Local\Programs\Python\Python314\python.exe"
$ROOT = Split-Path -Parent $PSScriptRoot
$CSV = "$ROOT\data_pipeline\output\ground_truth_dataset.csv"
$LOG = "$ROOT\data_pipeline\output\auto_train.log"
$RUN = 0

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

function CountRows {
    if (-not (Test-Path $CSV)) { return 0 }
    return ((Get-Content $CSV).Count - 1)
}

Log "======================================================"
Log " CELESTIS Auto-Train Loop Started"
Log " Will run every 65 min to respect Space-Track quota."
Log " Press Ctrl+C to stop."
Log "======================================================"

while ($true) {
    $RUN++
    Log ""
    Log "--- Run #$RUN ---"

    # ---- Step 1: Build dataset ----
    $before = CountRows
    Log "STEP 1 - Building dataset... (was $before rows)"
    & $PYTHON "$ROOT\data_pipeline\build_dataset.py" 2>&1 | ForEach-Object {
        Write-Host $_ -ForegroundColor Gray
        Add-Content -Path $LOG -Value $_ -Encoding UTF8
    }
    $after = CountRows
    $added = $after - $before
    Log "STEP 1 DONE - Dataset: $before -> $after rows (+$added new)"

    # ---- Step 2: Train PINN ----
    Log "STEP 2 - Training PINN on $after samples..."
    & $PYTHON "$ROOT\data_pipeline\train_pinn.py" 2>&1 | ForEach-Object {
        Write-Host $_ -ForegroundColor Gray
        Add-Content -Path $LOG -Value $_ -Encoding UTF8
    }

    # ---- Read + display benchmark results ----
    $benchFile = "$ROOT\data_pipeline\output\pinn_benchmark.json"
    if (Test-Path $benchFile) {
        $bench = Get-Content $benchFile -Raw | ConvertFrom-Json
        foreach ($r in $bench) {
            $label = $r.label
            $mae = [math]::Round($r.mae_total_km, 4)
            $r2 = [math]::Round($r.r2_total, 4)
            $viol = $r.physics_violations
            $n = $r.n_samples
            Log "  RESULT [$label]"
            Log "    MAE        : $mae km"
            Log "    R-squared  : $r2"
            Log "    Violations : $viol / $n"
        }
    }
    Log "STEP 2 DONE - Model saved to output/pinn_model.pt"

    # ---- Check target ----
    if ($after -ge 500) {
        Log ""
        Log "TARGET REACHED: $after rows >= 500. Training complete!"
        Log "======================================================"
        break
    }

    Log ""
    Log "============================================================"
    Log " Run #$RUN complete."
    Log " Total rows : $after / 500"
    Log " Next run   : in 65 minutes (~$(Get-Date -Format 'HH:mm') + 65min)"
    Log "============================================================"
    Start-Sleep -Seconds 3900
}
