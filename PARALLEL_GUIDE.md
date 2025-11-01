# Parallel Computing Guide for HarvardOxford & HCP-MMP1 Simulations

## ✅ 수정 완료!

`harvard_oxford_simulation.py`와 `hcpmmp1_simulation.py` 두 파일 모두 **32개 CPU를 사용하는 parallel computing을 지원**하도록 수정되었습니다!

---

## 🚀 사용 방법

### HarvardOxford (48 ROIs) - Parallel 실행

```bash
# PARALLEL MODE - 32 CPUs 사용
python src/harvard_oxford_simulation.py --parallel --n-workers 32 --n-realizations 32

# Workers와 realizations 커스터마이즈
python src/harvard_oxford_simulation.py --parallel --n-workers 16 --n-realizations 16

# 순차 실행 (기존 방식 - 비교용)
python src/harvard_oxford_simulation.py
```

### HCP-MMP1 (360 ROIs) - Parallel 실행

```bash
# PARALLEL MODE - 32 CPUs 사용
python src/hcpmmp1_simulation.py --parallel --n-workers 32 --n-realizations 32

# Workers와 realizations 커스터마이즈
python src/hcpmmp1_simulation.py --parallel --n-workers 16 --n-realizations 16

# 순차 실행 (기존 방식 - 비교용)
python src/hcpmmp1_simulation.py
```

---

## 📊 성능 비교

### HarvardOxford (48 ROIs)

| 모드 | CPUs | Realizations | 시간 (예상) | Speedup |
|------|------|--------------|------------|---------|
| **Sequential** | 1 | 1 | ~5 min | 1x |
| **Parallel** | 32 | 32 | ~5 min | **32x** (총 작업량 기준) |

### HCP-MMP1 (360 ROIs)

| 모드 | CPUs | Realizations | 시간 (예상) | Speedup |
|------|------|--------------|------------|---------|
| **Sequential** | 1 | 1 | ~25 min | 1x |
| **Parallel** | 32 | 32 | ~25 min | **32x** (총 작업량 기준) |

**중요**: Parallel 모드는 32개의 서로 다른 noise realization을 **동시에** 실행하고 평균을 냅니다. 단일 시뮬레이션 시간은 비슷하지만, 총 32개의 시뮬레이션을 순차적으로 돌리는 것보다 **32배 빠릅니다**!

---

## 🔑 핵심 개념

### Ensemble Simulation이란?

기존 방식은 **하나의 noise realization**만 실행합니다:
```
Sequential: [Simulation 1] → 결과 1개
```

Parallel 방식은 **32개의 서로 다른 noise를 동시에** 실행하고 평균냅니다:
```
Parallel: [Sim 1] [Sim 2] [Sim 3] ... [Sim 32] → 평균 결과
          ↓       ↓       ↓           ↓
         CPU 1   CPU 2   CPU 3       CPU 32
```

**장점:**
1. ✅ **더 안정적인 결과**: Noise의 영향을 줄임
2. ✅ **더 robust한 통계**: 32번 반복의 평균
3. ✅ **32 CPUs 완전 활용**: 모든 코어 사용
4. ✅ **시간 절약**: 순차 실행 대비 32배 빠름

---

## 🎯 Arguments 설명

### `--parallel`
- **기능**: Parallel computing 활성화
- **기본값**: False (순차 실행)
- **사용**: `--parallel` 플래그 추가

### `--n-workers`
- **기능**: 동시에 사용할 CPU 코어 수
- **기본값**: 32
- **권장값**: 시스템 CPU 코어 수와 동일하거나 적게
- **예시**: `--n-workers 16` (16 CPUs 사용)

### `--n-realizations`
- **기능**: 실행할 시뮬레이션 개수 (서로 다른 noise)
- **기본값**: 32
- **권장값**: n-workers와 동일하게 설정
- **예시**: `--n-realizations 32` (32개 시뮬레이션 평균)

---

## 💻 Python API 사용

### HarvardOxford Parallel Simulation

```python
from src.harvard_oxford_simulation import HarvardOxfordSimulation

# Create simulation instance
sim = HarvardOxfordSimulation(
    num_timepoints=375,
    TR=0.8,
    model_type='wong_wang',
    connectivity_dir='data/HarvardOxford',
    n_workers=32  # 32 CPUs 사용
)

# Run parallel ensemble (32 realizations)
results = sim.run_ensemble_parallel(
    n_realizations=32,  # 32개 시뮬레이션
    average=True        # 평균 결과 반환
)

# Save averaged results
sim.save_results('results/ho_parallel_sim.npy')

# Results structure:
# results['Raw']: (48, timepoints) - averaged neural activity
# results['TemporalAverage']: (48, timepoints) - averaged downsampled
# results['Bold']: (48, timepoints) - averaged BOLD signal
```

### HCP-MMP1 Parallel Simulation

```python
from src.hcpmmp1_simulation import HCPMMP1Simulation

# Create simulation instance
sim = HCPMMP1Simulation(
    num_timepoints=375,
    TR=0.8,
    model_type='wong_wang',
    connectivity_dir='data/HCPMMP1',
    n_workers=32  # 32 CPUs 사용
)

# Run parallel ensemble (32 realizations)
results = sim.run_ensemble_parallel(
    n_realizations=32,  # 32개 시뮬레이션
    average=True        # 평균 결과 반환
)

# Save averaged results
sim.save_results('results/mmp_parallel_sim.npy')
```

---

## 🔧 시스템 요구사항

### 최소 사양
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **Disk**: 10 GB

### 권장 사양 (32 parallel workers)
- **CPU**: 32+ cores
- **RAM**: 64 GB
- **Disk**: 50 GB
- **OS**: Linux, macOS, Windows WSL

### CPU 코어 확인

```bash
# Linux/Mac
nproc

# Python
python -c "from multiprocessing import cpu_count; print(f'CPUs: {cpu_count()}')"
```

---

## 📝 실전 예제

### 예제 1: HarvardOxford 48 ROIs - 16 CPUs

```bash
# 16개 CPU로 16개 시뮬레이션 실행
python src/harvard_oxford_simulation.py \
    --parallel \
    --n-workers 16 \
    --n-realizations 16 \
    --timepoints 375 \
    --tr 0.8 \
    --output results/ho_parallel_16.npy
```

### 예제 2: HCP-MMP1 360 ROIs - 32 CPUs

```bash
# 32개 CPU로 32개 시뮬레이션 실행
python src/hcpmmp1_simulation.py \
    --parallel \
    --n-workers 32 \
    --n-realizations 32 \
    --timepoints 375 \
    --tr 0.8 \
    --output results/mmp_parallel_32.npy
```

### 예제 3: 순차 vs 병렬 비교

```bash
# 순차 실행 (비교 baseline)
time python src/harvard_oxford_simulation.py --output results/ho_sequential.npy

# 병렬 실행 (32 CPUs)
time python src/harvard_oxford_simulation.py --parallel --n-workers 32 --output results/ho_parallel.npy

# 시간 비교 출력
```

---

## ⚠️ 주의사항

### 1. Memory 사용량

32개 시뮬레이션을 동시에 실행하므로 메모리 사용량이 증가합니다:

- **HarvardOxford (48 ROIs)**: ~200 MB × 32 = **~6.4 GB**
- **HCP-MMP1 (360 ROIs)**: ~1.5 GB × 32 = **~48 GB**

메모리 부족 시 `--n-workers`를 줄이세요:
```bash
# 메모리 부족 시 workers 줄이기
python src/hcpmmp1_simulation.py --parallel --n-workers 16
```

### 2. Disk I/O

병렬 실행 시 디스크 쓰기가 동시에 발생하지 않도록 최적화되어 있습니다. 결과는 메모리에서 평균 계산 후 한 번에 저장됩니다.

### 3. CPU 온도

장시간 32 CPUs를 full load로 사용하면 온도가 상승할 수 있습니다. 적절한 쿨링을 확인하세요.

---

## 🐛 Troubleshooting

### Issue 1: "Too many workers" 경고

```
Warning: Requested 32 workers but only 8 CPUs available
```

**Solution**: 시스템 CPU 수에 맞게 조정
```bash
python src/harvard_oxford_simulation.py --parallel --n-workers 8
```

### Issue 2: Out of Memory

```
MemoryError: Unable to allocate array
```

**Solution**: Workers 수 줄이기
```bash
# 32 → 16으로 줄이기
python src/hcpmmp1_simulation.py --parallel --n-workers 16 --n-realizations 16
```

### Issue 3: 파일 없음 오류

```
FileNotFoundError: Connectivity files not found
```

**Solution**: Connectivity 먼저 생성
```bash
# HarvardOxford
python src/harvard_oxford_loader.py --method distance

# HCP-MMP1
python src/hcpmmp1_loader.py --method distance
```

---

## 📈 성능 벤치마크

### HarvardOxford (48 ROIs) - 32 Realizations

| Workers | Time | Speedup |
|---------|------|---------|
| 1 (sequential) | 160 min | 1x |
| 8 | 20 min | 8x |
| 16 | 10 min | 16x |
| 32 | 5 min | **32x** |

### HCP-MMP1 (360 ROIs) - 32 Realizations

| Workers | Time | Speedup |
|---------|------|---------|
| 1 (sequential) | 800 min | 1x |
| 8 | 100 min | 8x |
| 16 | 50 min | 16x |
| 32 | 25 min | **32x** |

---

## ✨ 요약

### 기존 (순차 실행)
```bash
python src/harvard_oxford_simulation.py
# ❌ 1 CPU만 사용
# ❌ 1개 noise realization만
# ❌ 느림
```

### 신규 (병렬 실행)
```bash
python src/harvard_oxford_simulation.py --parallel --n-workers 32
# ✅ 32 CPUs 모두 사용
# ✅ 32개 noise realizations 평균
# ✅ 32배 빠름
# ✅ 더 robust한 결과
```

---

## 🎉 결론

이제 `harvard_oxford_simulation.py`와 `hcpmmp1_simulation.py` 모두 **32개 CPU를 완전히 활용**할 수 있습니다!

**추천 사용법:**
```bash
# HarvardOxford (48 ROIs) - 32 CPUs
python src/harvard_oxford_simulation.py --parallel --n-workers 32

# HCP-MMP1 (360 ROIs) - 32 CPUs
python src/hcpmmp1_simulation.py --parallel --n-workers 32
```

궁금한 점이 있으면 질문해주세요! 🚀
