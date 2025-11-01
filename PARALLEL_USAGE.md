# Parallel Computing Usage Guide

## Overview

기존 코드는 **순차적(sequential)** 실행만 지원했습니다. 이제 **parallel computing**을 사용하여 여러 atlas 시뮬레이션을 동시에 실행할 수 있습니다!

## 🚀 Performance Comparison

### Sequential (기존 방식)
```bash
# 순차적으로 하나씩 실행 - SLOW!
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8   # ~5 min
python src/simulation.py --nodes 68 --timepoints 375 --tr 0.8   # ~8 min
python src/simulation.py --nodes 360 --timepoints 375 --tr 0.8  # ~20 min

# Total: ~33 minutes
```

### Parallel (새로운 방식) ⚡
```bash
# 3개 atlas를 동시에 실행 - FAST!
python run_parallel_atlases.py --timepoints 375 --tr 0.8 --workers 3

# Total: ~20 minutes (가장 오래 걸리는 360-node 시뮬레이션 시간)
# Speedup: ~1.65x faster!
```

## 📊 Available Scripts

### 1. `run_parallel_atlases.py` - Multiple Atlas Parallel Execution

여러 atlas 구성(48, 68, 360 nodes)을 병렬로 실행합니다.

**Basic Usage:**
```bash
# Default: 48, 68, 360 nodes를 모두 병렬 실행
python run_parallel_atlases.py

# 특정 파라미터 지정
python run_parallel_atlases.py --timepoints 375 --tr 0.8 --workers 3

# 다른 모델 사용
python run_parallel_atlases.py --model kuramoto --workers 3

# 특정 atlas만 실행
python run_parallel_atlases.py --atlases 48 68 --workers 2
```

**Arguments:**
- `--timepoints`: 시뮬레이션 timepoints (default: 375)
- `--tr`: Repetition time in seconds (default: 0.8)
- `--model`: Neural mass model (`wong_wang`, `kuramoto`, `generic_2d_oscillator`)
- `--output`: 출력 디렉토리 (default: `results`)
- `--workers`: 병렬 worker 수 (default: 3)
- `--atlases`: 실행할 atlas 구성 (default: 48 68 360)

**Output:**
```
results/
├── sim_48nodes_375tp_TR0.8_neural.npy
├── sim_48nodes_375tp_TR0.8_bold.npy
├── sim_48nodes_375tp_TR0.8_metadata.json
├── sim_68nodes_375tp_TR0.8_neural.npy
├── sim_68nodes_375tp_TR0.8_bold.npy
├── sim_68nodes_375tp_TR0.8_metadata.json
├── sim_360nodes_375tp_TR0.8_neural.npy
├── sim_360nodes_375tp_TR0.8_bold.npy
├── sim_360nodes_375tp_TR0.8_metadata.json
└── parallel_atlas_summary.json  # 병렬 실행 요약
```

### 2. `src/parallel_simulation.py` - Multiple Subject Parallel Execution

여러 subject를 병렬로 시뮬레이션합니다 (대규모 연구용).

**Usage:**
```bash
# 10,000명의 subject 시뮬레이션 (32 CPUs 사용)
python src/parallel_simulation.py --subjects 10000 --nodes 48 --workers 32

# 작은 규모 테스트
python src/parallel_simulation.py --subjects 100 --nodes 48 --workers 8
```

**Arguments:**
- `--subjects`: 시뮬레이션할 subject 수 (default: 10000)
- `--nodes`: Atlas 구성 (48 or 360)
- `--workers`: 병렬 worker 수 (default: 32)
- `--start`: 시작 subject ID (재개용)

**Output:**
```
results/parallel_sims/
├── subject_00000/
│   ├── sim_48nodes_375tp_TR0.8_neural.npy
│   ├── sim_48nodes_375tp_TR0.8_bold.npy
│   └── sim_48nodes_375tp_TR0.8_metadata.json
├── subject_00001/
│   └── ...
├── ...
└── simulation_summary.json  # 전체 batch 요약
```

## 💻 System Requirements

### Recommended Configuration

**For Atlas Parallel Execution (run_parallel_atlases.py):**
- CPU cores: 3+ (각 atlas당 1 core)
- RAM: 8-16 GB
- Disk space: ~5 GB (3 atlases)

**For Subject Parallel Execution (parallel_simulation.py):**
- CPU cores: 8-32 (더 많을수록 빠름)
- RAM: 32-64 GB (32 workers 기준)
- Disk space: 대규모 (10,000 subjects = ~500 GB)

### CPU Core Usage

현재 시스템의 CPU 정보 확인:
```bash
python -c "from multiprocessing import cpu_count; print(f'Available CPU cores: {cpu_count()}')"
```

## 🔧 Python API Usage

### Example 1: Run All Atlases in Parallel

```python
from run_parallel_atlases import run_all_atlases_parallel

# Run 48, 68, 360 nodes in parallel
results = run_all_atlases_parallel(
    num_timepoints=375,
    TR=0.8,
    model_type='wong_wang',
    output_prefix='results',
    num_workers=3
)

# Check results
for r in results:
    if r['status'] == 'success':
        print(f"{r['num_nodes']} nodes: {r['elapsed_time_minutes']:.1f} min")
```

### Example 2: Parameter Sweep

```python
from run_parallel_atlases import run_parameter_sweep

# Test multiple parameters in parallel
results = run_parameter_sweep(
    node_configs=[48, 68],
    timepoints_list=[375, 750],
    TR_list=[0.8, 1.0],
    model_types=['wong_wang', 'kuramoto'],
    output_prefix='results/parameter_sweep',
    num_workers=8
)

# Total combinations: 2 nodes × 2 timepoints × 2 TRs × 2 models = 16 simulations
# With 8 workers: ~2-3x faster than sequential
```

### Example 3: Multiple Subjects

```python
from src.parallel_simulation import run_parallel_simulations

# Simulate 1000 subjects with 48 nodes each
results = run_parallel_simulations(
    num_subjects=1000,
    num_nodes=48,
    num_timepoints=375,
    TR=0.8,
    model_type='wong_wang',
    output_dir='results/1000subjects',
    num_workers=16
)

# Check completion rate
successful = sum(1 for r in results if r['status'] == 'success')
print(f"Completed: {successful}/1000 subjects")
```

## 📈 Performance Tips

### 1. Optimal Worker Count

**Atlas Parallel (`run_parallel_atlases.py`):**
- 3 workers for 3 atlases (48, 68, 360)
- More workers don't help since only 3 simulations

**Subject Parallel (`parallel_simulation.py`):**
- Use `CPU cores - 1` to leave system responsive
- Example: 32-core machine → use 28-30 workers

### 2. Memory Considerations

**Per simulation memory usage:**
- 48 nodes: ~200 MB
- 68 nodes: ~400 MB
- 360 nodes: ~1.5 GB

**For parallel execution:**
- 3 workers (all atlases): ~2 GB total
- 16 workers (48 nodes): ~3-4 GB total
- 32 workers (48 nodes): ~6-8 GB total

### 3. Disk I/O

병렬 실행 시 디스크 쓰기가 병목이 될 수 있습니다:
- SSD 사용 권장
- 각 simulation을 다른 디렉토리에 저장
- 네트워크 드라이브보다 로컬 디스크 사용

## 🐛 Troubleshooting

### Issue: "Too many workers" warning

```
Warning: Requested 32 workers but only 8 CPUs available
Using 8 workers instead
```

**Solution:** 시스템 CPU 수에 맞게 `--workers` 조정

### Issue: Out of memory

```
MemoryError: Unable to allocate array
```

**Solution:**
1. Worker 수 줄이기: `--workers 4`
2. 더 작은 atlas 사용: `--atlases 48 68` (360 제외)
3. Timepoints 줄이기: `--timepoints 200`

### Issue: Simulations hanging

**Solution:**
1. Ctrl+C로 중단
2. Worker 수 줄이기
3. 한 번에 하나씩 테스트:
   ```bash
   python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8
   ```

## 📝 Summary

| Method | Script | Use Case | Speedup |
|--------|--------|----------|---------|
| **Sequential** | `src/simulation.py` | Single atlas | 1x (baseline) |
| **Atlas Parallel** | `run_parallel_atlases.py` | Multiple atlases | ~1.5-2x |
| **Subject Parallel** | `src/parallel_simulation.py` | Large studies | ~8-16x (with 16-32 workers) |

## 🎯 Recommended Workflow

### For Exploratory Analysis (소규모)
```bash
# Test single atlas first
python src/simulation.py --nodes 68 --timepoints 375 --tr 0.8

# Then run all atlases in parallel
python run_parallel_atlases.py --timepoints 375 --tr 0.8 --workers 3
```

### For Large-Scale Studies (대규모)
```bash
# Run many subjects in parallel
python src/parallel_simulation.py --subjects 10000 --nodes 48 --workers 32
```

### For Parameter Exploration (파라미터 탐색)
```python
from run_parallel_atlases import run_parameter_sweep

run_parameter_sweep(
    node_configs=[48, 68, 360],
    timepoints_list=[375],
    TR_list=[0.5, 0.8, 1.0, 1.5],
    model_types=['wong_wang'],
    num_workers=8
)
```

---

**Questions?** Check the main README.md or CLAUDE.md for more details about the simulation framework.
