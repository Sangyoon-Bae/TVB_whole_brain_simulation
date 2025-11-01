#!/bin/bash
# Run 10,000 subjects for both HarvardOxford and HCP-MMP1

echo "=========================================="
echo "10,000 SUBJECT SIMULATION - BOTH ATLASES"
echo "=========================================="
echo ""

# Check if running in parallel or sequential
MODE=${1:-sequential}

if [ "$MODE" == "parallel" ]; then
    echo "🚀 PARALLEL MODE: Running both atlases simultaneously"
    echo "   - HarvardOxford: 16 workers"
    echo "   - HCP-MMP1: 16 workers"
    echo ""

    # HarvardOxford (16 workers)
    nohup python run_10k_subjects_atlas.py \
        --atlas harvard_oxford \
        --subjects 10000 \
        --workers 16 \
        --output results/10k_harvard_oxford \
        > ho_10k.log 2>&1 &

    HO_PID=$!
    echo "✓ HarvardOxford started (PID: $HO_PID)"

    # HCP-MMP1 (16 workers)
    nohup python run_10k_subjects_atlas.py \
        --atlas hcpmmp1 \
        --subjects 10000 \
        --workers 16 \
        --output results/10k_hcpmmp1 \
        > mmp_10k.log 2>&1 &

    MMP_PID=$!
    echo "✓ HCP-MMP1 started (PID: $MMP_PID)"

    echo ""
    echo "Monitor progress:"
    echo "  HarvardOxford: tail -f ho_10k.log"
    echo "  HCP-MMP1: tail -f mmp_10k.log"

else
    echo "📌 SEQUENTIAL MODE: Running one after another"
    echo "   - HarvardOxford: 32 workers (~26 hours)"
    echo "   - HCP-MMP1: 32 workers (~130 hours)"
    echo ""

    # HarvardOxford first (32 workers)
    echo "Starting HarvardOxford (48 ROIs)..."
    python run_10k_subjects_atlas.py \
        --atlas harvard_oxford \
        --subjects 10000 \
        --workers 32 \
        --output results/10k_harvard_oxford \
        | tee ho_10k.log

    echo ""
    echo "✓ HarvardOxford complete!"
    echo ""
    echo "Starting HCP-MMP1 (360 ROIs)..."

    # HCP-MMP1 second (32 workers)
    python run_10k_subjects_atlas.py \
        --atlas hcpmmp1 \
        --subjects 10000 \
        --workers 32 \
        --output results/10k_hcpmmp1 \
        | tee mmp_10k.log

    echo ""
    echo "✓ HCP-MMP1 complete!"
fi

echo ""
echo "=========================================="
echo "DONE!"
echo "=========================================="
