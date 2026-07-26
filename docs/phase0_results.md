# Phase 0 결과

아래 내용은 실제로 실행된 것만 반영합니다. 이 저장소 히스토리상 아직
실행해보지 못한 하드웨어(Linux, Compute Capability 7.0 이상 GPU)에 대한
행은 `PASS`/`FAIL`이 아니라 `NOT_YET_TESTED_ON_THIS_MACHINE`으로
표시됩니다. macOS arm64/CPU와 Windows x64/CUDA(GTX 1080, SM 6.1)는 둘 다
end-to-end로 검증을 마쳤습니다 -- 아래 각 섹션 참고.

## 상태 정의

| 상태 | 의미 |
|---|---|
| PASS | 실행되었고 성공 기준을 만족함. |
| FAIL | 실행은 되었으나 성공 기준을 만족하지 못함 (parity 허용오차 초과, 크래시, 0이 아닌 종료 코드). |
| SKIPPED | 이 머신에 필요한 리소스가 없어 실행되지 않음 (CUDA GPU 없음 등). |
| UNSUPPORTED | 현재 공식 배포판에서 해당 구성 자체를 백엔드/API가 지원하지 않음. |
| BLOCKED | 선행 조건(빌드 산출물, 아티팩트)이 없어서 실행할 수 없었음. |
| INVALID_BUILD_CONFIGURATION | Debug/Release 또는 플랫폼 불일치; 백엔드 자체의 실패가 아님. |
| NOT_YET_TESTED_ON_THIS_MACHINE | 이 세션에서 해당 하드웨어/OS를 사용할 수 없어 시도조차 하지 않음. |

## 환경 (macOS 실행분)

| 항목 | 값 |
|---|---|
| OS | macOS-15.5-arm64-arm-64bit (Darwin 24.5.0) |
| 아키텍처 | arm64 (Apple Silicon, M2) |
| 플랫폼 / 빌드 구성 | macOS arm64, Release (Windows x64 Release는 아래 별도 섹션 참고) |
| Python | 3.11.15 (conda 환경 `image-ai-studio`) |
| PyTorch | 2.13.0 |
| PyTorch CUDA 빌드 | 없음 (CPU 전용 wheel; 이 머신엔 CUDA 없음) |
| torch.cuda.is_available() | False |
| torch.backends.mps.is_available() | True (미사용 -- Phase 0의 device enum은 스펙상 cpu\|cuda만 다룸) |
| GPU | 없음 (Apple M2 내장 GPU, NVIDIA/CUDA 아님) |
| NVIDIA 드라이버 | 해당 없음 (NVIDIA GPU 없음) |
| CUDA 툴킷 | 해당 없음 (nvcc 없음) |
| cuDNN | 해당 없음 |
| Visual Studio / MSVC | 해당 없음 (Windows 아님) |
| 컴파일러 | Apple clang 17.0.0 (clang-1700.0.13.5) |
| CMake | 4.0.3 |
| Git | 2.50.1 |
| LibTorch 출처 | `torch` pip 패키지에 번들된 LibTorch (`torch.utils.cmake_prefix_path`), 별도 다운로드 아님 |
| LibTorch 버전 | 2.13.0 (Python torch와 동일 -- 같은 패키지라 버전 드리프트 불가능) |
| LibTorch variant | cpu-release |
| Python/LibTorch 버전 일치 여부 | 예, 구조적으로 보장됨 (동일 설치본) |

원본 덤프: `artifacts/environment_report.json` (`scripts/inspect_environment.py`가 생성).

## AOTInductor C++ 런타임 지원 여부 프로브

| 단계 | 결과 |
|---|---|
| 헤더 (`torch/csrc/inductor/aoti_package/model_package_loader.h`) | **발견됨** at `<torch pkg>/include/torch/csrc/inductor/aoti_package/model_package_loader.h` |
| 라이브러리/심볼 (`torch::inductor::AOTIModelPackageLoader`) | **발견됨**, `libtorch_cpu.dylib`에서 export됨 (`nm -gU`로 확인) |
| 컴파일 프로브 (`cpp/aoti_probe`) | **PASS** -- 독립된 `build-aoti-probe/` 디렉터리에서 `-DBUILD_AOTI_PROBE=ON`으로 구성/빌드 |
| 링크 프로브 | **PASS** -- `probe_aoti`에서 `torch::inductor::AOTIModelPackageLoader` 생성이 링크까지 성공 |
| 패키지 로드 (`artifacts/aoti/tiny_cnn/cpu/model.pt2`) | **PASS** -- `loader.run({input})`이 `[1, 10]` 텐서를 반환, 종료 코드 0 |
| **종합** | **SUPPORTED** (macOS 15.5 arm64, clang 17, PyTorch/LibTorch 2.13.0, CPU 기준) |

이는 macOS/arm64/CPU에 한정된, 실제로 재현 가능한 데이터입니다.
Windows/MSVC나 Linux/CUDA에 대해서는 아무것도 말해주지 않으며, 해당
하드웨어에서 별도로 프로브해야 합니다.

이 LibTorch 버전에서 실제로 발견된 네임스페이스/API (스펙에서 예상한 형태와 정확히 일치):

```cpp
namespace torch::inductor {
class AOTIModelPackageLoader {
 public:
  AOTIModelPackageLoader(const std::string& model_package_path,
                          const std::string& model_name = "model",
                          const bool run_single_threaded = false,
                          const size_t num_runners = 1,
                          const c10::DeviceIndex device_index = -1);
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs, void* stream_handle = nullptr);
  ...
};
}
```

## 빌드 결과 (macOS)

| 타겟 | 결과 |
|---|---|
| `run_torchscript` (독립된 `build-torchscript/`) | **BUILD PASS** |
| `probe_aoti` (독립된 `build-aoti-probe/`) | **BUILD PASS** |
| `run_aoti` (독립된 `build-aoti/`) | **BUILD PASS** |
| 클린 리빌드 (`rm -rf` 후 재구성/재빌드, 두 타겟 모두) | 이 세션에서 각각 별도로 확인, **PASS** |

`run_torchscript`은 `BUILD_AOTI_PROBE=OFF`, `BUILD_AOTI_RUNNER=OFF`로
구성/빌드되며 AOTI 타겟 둘 중 어느 것도 존재하거나 성공할 필요가
없습니다 -- `phase0_common` + `run_torchscript`만 포함한
`build-torchscript/`가 정상 빌드됨으로써 확인됨.

Windows MSVC 빌드: 아래 "Windows x64 Release 결과" 섹션 참고.

## 테스트 결과 (모델 x 백엔드 x 디바이스, macOS)

Repeat = 100, warmup = 10 (별도 명시 없는 한). rtol/atol: CPU FP32
1e-5/1e-6, CUDA FP32 1e-4/1e-5 (스펙 기본값 그대로 -- 이 실행에서
조정 필요 없었음).

| 모델 | 백엔드 | 디바이스 | 상태 | Parity (최대 절대 오차) | allclose |
|---|---|---|---|---|---|
| TinyCNN | TorchScript | CPU | **PASS** | 0.0 (bit-identical) | true |
| TinyResidualCNN | TorchScript | CPU | **PASS** | 0.0 (bit-identical) | true |
| TinyCNN | AOTInductor | CPU | **PASS** | 1.04e-07 | true |
| TinyResidualCNN | AOTInductor | CPU | **PASS** | 5.96e-08 | true |
| TinyCNN | TorchScript | CUDA | SKIPPED (이 머신에 CUDA GPU 없음) | -- | -- |
| TinyResidualCNN | TorchScript | CUDA | SKIPPED | -- | -- |
| TinyCNN | AOTInductor | CUDA | SKIPPED | -- | -- |
| TinyResidualCNN | AOTInductor | CUDA | SKIPPED | -- | -- |
| (CPU 4개 조합 전체) | -- | -- | 100회 반복 안정성 | **PASS** (아래 참고) | -- |
| (Linux CPU/CUDA 4개 조합 전체) | -- | -- | NOT_YET_TESTED_ON_THIS_MACHINE | -- | -- |

전체 머신 판독 가능 로그: `results/test_matrix.json`.

## 성능 (CPU, macOS 머신 기준)

| 모델 | 백엔드 | 모델 로드 (ms) | 첫 추론 (ms) | Warmup 평균 (ms) | 측정 평균 (ms, n=100) | 측정 최소/최대 (ms) | 표준편차 (ms) | 아티팩트 크기 |
|---|---|---|---|---|---|---|---|---|
| TinyCNN | TorchScript | 9.63 | 27.44 | 2.62 | 2.41 | 1.91 / 3.04 | 0.24 | 35.4 KB (`model.pt`) |
| TinyResidualCNN | TorchScript | 17.23 | 33.65 | 6.50 | 5.33 | 3.82 / 15.17 | 1.37 | 89.0 KB (`model.pt`) |
| TinyCNN | AOTInductor | 148.48 | 5.43 | 3.75 | 3.23 | 2.52 / 5.72 | 0.58 | 395.7 KB (`model.pt2`) |
| TinyResidualCNN | AOTInductor | 148.12 | 6.65 | 5.24 | 5.13 | 4.25 / 10.52 | 0.81 | 441.3 KB (`model.pt2`) |

이 머신/실행에 국한된 관찰 (작은 모델, n=100, 단일 프로세스 -- Windows/CUDA
동작에 대한 주장이 아님):

- AOTInductor의 모델 *로드*는 이 머신에서 훨씬 느림 (~148ms vs TorchScript
  ~10-17ms) -- 패키지 압축 해제가 포함되기 때문으로 보이며, 정상 상태
  *추론* 속도는 첫 호출 이후로는 TorchScript와 비슷하거나 약간 더 빠름.
- AOTInductor의 *첫 추론*은 TorchScript보다 눈에 띄게 빠름 (TorchScript
  인터프리터처럼 첫 호출에 별도 JIT 워밍업 비용이 들어가지 않음) --
  로드 시간과는 반대 양상.
- AOTInductor 아티팩트(`.pt2`)는 이 작은 모델들 기준으로 동등한
  TorchScript `.pt`보다 디스크 용량이 대략 5~10배 큼.

## 반복 안정성 (100회 반복, macOS 머신)

CPU 4개 조합 전체 (TinyCNN/TinyResidualCNN x TorchScript/AOTInductor):

- 100회 측정 루프 동안 크래시나 예외 없음.
- 100회 반복 전체에서 출력 shape이 동일함을 러너 자체가 검증함 (다르면 예외 발생).
- 추가로 **완전히 새 프로세스로 3회 재시작**(매번 새 프로세스, 아티팩트
  새로 로드)해도 4개 조합 모두 **bit-identical 출력**을 확인 -- eval()
  모드의 결정론적 동작과 `.pt`/`.pt2` 아티팩트의 반복적인 콜드 리로드
  성공을 모두 확인.

## GPU 메모리 관찰 (macOS)

이 머신엔 NVIDIA GPU가 없음. `scripts/monitor_gpu_memory.py`가
`nvidia-smi` 부재를 감지하고 데이터를 조작하는 대신 `NOT_APPLICABLE`
행 하나만 기록함 (`results/gpu_memory_log.csv`). GPU 메모리 추이 분석
(누수 vs 캐시 후 안정화)은 이 머신에서는 **데이터 부족**으로 CUDA
가능 머신에서의 실행으로 미룸.

## 알려진 한계 (macOS 실행 기준)

- 이 저장소 히스토리상 지금까지 macOS(Apple Silicon, CPU)만 end-to-end로
  검증되었음. Windows x64 Release(1차 타겟)와 Linux/CUDA 경로는 구현 및
  문서화는 되어 있으나 이 실행 시점 기준 **NOT_YET_TESTED_ON_THIS_MACHINE**
  이었음 (Windows는 이후 아래 섹션에서 검증 완료).
- TorchScript: `torch.jit.trace`만 검증했고 `torch.jit.script`는 검증 안 함.
- AOTInductor: compile/package/load API
  (`torch._inductor.aoti_compile_and_package` /
  `torch._inductor.aoti_load_package`)는 PyTorch 2.13.0에서 private
  모듈인 `torch._inductor` 아래에 있음 -- 공개 별칭 없음. PyTorch
  버전이 바뀌면 언제든 변경될 수 있는 API로 취급할 것.
- 중간 레이어 parity는 구현하지 않았고, 최종 출력 parity만 확인함
  (스펙에 따라 범위를 제한하기 위함).
- CUDA 메모리 누수 추이 분석은 아직 어느 머신에서도 검증되지 않음
  (이 세션에서는 CUDA GPU를 사용할 수 없었음).

## Windows x64 Release 결과 (2026-07-26)

머신: Windows 11, MSVC 14.44 (VS2022 Community), PyTorch 2.12.1+cu126,
GPU: NVIDIA GTX 1080 (SM 6.1).

### 스크립트 버그 수정 (백엔드 자체의 결과가 아니라 이 저장소의 버그)

- `run_torchscript_tests.py` / `run_aoti_tests.py`가
  `build-*/cpp/*/run_*.exe` 경로를 직접 찾고 있었는데, MSVC의 멀티
  컨피그 제너레이터는 실제로 `.../Release/` 하위에 결과물을 생성함.
  두 스크립트(및 `run_phase0.py`의 `probe_aoti` 호출)는 이제 두
  위치를 모두 확인하는 공용 `find_runner_binary()` 헬퍼를 사용함.
- `export_models.py`가 AOTInductor export 실패를 삼키고 있었음
  (`metadata.json`에 `status: FAIL`만 기록하고 겉으로는 드러내지
  않음). 이제 각 export 후 그 status를 다시 읽어 PASS/FAIL을
  출력하고, 하나라도 실패하면 0이 아닌 코드로 종료함.
- `image_ai_studio/tools/msvc_env.py` 추가: AOTInductor의 컴파일
  단계는 (MSVC를 알아서 찾는 CMake 빌드와 달리) `cl.exe`를 직접
  셸아웃으로 호출함. Developer Command Prompt 밖에서 그냥
  `python export_models.py`를 실행하면 `cl`이 PATH에 없었음. 이
  헬퍼는 `vswhere.exe`로 `vcvarsall.bat`을 자동으로 찾아 그 환경을
  주입하므로, 수동으로 셸을 설정할 필요가 없어짐.

### 테스트 결과 (모델 x 백엔드 x 디바이스, Windows)

| 모델 | 백엔드 | 디바이스 | 상태 | 비고 |
|---|---|---|---|---|
| TinyCNN / TinyResidualCNN | TorchScript | CPU | **PASS** | bit-identical parity |
| TinyCNN / TinyResidualCNN | TorchScript | CUDA | **PASS** | CUDA FP32 허용오차 이내 |
| TinyCNN / TinyResidualCNN | AOTInductor | CPU | **FAIL** | export/추론은 정상, 프로세스 종료 시 크래시 (아래 참고) |
| TinyCNN / TinyResidualCNN | AOTInductor | CUDA | **FAIL (export 단계)** | 버그 아니라 하드웨어 문제 -- 아래 참고 |

### AOTInductor CUDA: 하드웨어 제약, 버그 아님

Export가 `GPUTooOldForTriton`으로 실패함: AOTInductor의 CUDA 코드젠은
Triton을 필요로 하고, Triton은 CUDA Compute Capability 7.0 이상을
요구함. GTX 1080은 SM 6.1. 이 GPU에서는 해결 불가능.

### AOTInductor CPU: 프로세스 종료 시 크래시 -- 원인은 이 저장소가 아니라 PyTorch 쪽으로 규명됨

`run_aoti.exe` / `probe_aoti.exe`는 CPU AOTI 케이스마다 `0xC0000005`
(access violation)로 종료되는데, 이는 실행이 **성공적으로 끝난 뒤에만**
발생함:

- `probe_aoti`에 단계별 flush 로깅을 추가해서 확인한 결과, 생성자,
  `run()`, 출력까지 전부 성공함 (`result: SUPPORTED`까지 출력됨).
  크래시는 실제 추론 중이 아니라 `AOTIModelPackageLoader`의 소멸자 /
  프로세스 종료(teardown) 단계에서만 발생함.
- 동일한 크래시가 C++ 없이 `torch._inductor.aoti_load_package()` +
  추론만 사용하는 **순수 Python** 스크립트에서도 재현됨 -- 이 프로젝트의
  C++ 러너 코드가 원인일 가능성을 배제함.
- Windows Error Reporting은 크래시 모듈을 `VCOMP140.DLL`로 지목함
  (크래시 시점에 이미 언로드된 상태 -- 종료 순서 문제의 전형적인 증상).
- `dumpbin /imports`로 확인한 결과, 서로 다른 OpenMP 런타임 두 개가 같은
  프로세스에 로드됨: `torch_cpu.dll`은 Intel OpenMP(`libiomp5md.dll`)를
  링크하고, AOTInductor가 컴파일한 모델 코드(`wrapper.pyd`, `cl.exe
  /openmp`로 빌드됨)는 Microsoft OpenMP(`VCOMP140.DLL`)를 링크함.
  `KMP_DUPLICATE_LIB_OK=TRUE`로도 해결되지 않음.

**결론:** PyTorch 2.12.1의 AOTInductor Windows CPU 백엔드 내부에서
발생하는 이중 OpenMP 런타임 충돌 문제이며(libtorch는 Intel OpenMP로
빌드됨, 코드젠된 모델 코드는 MSVC의 OpenMP로 컴파일됨), 이 프로젝트의
export/러너 코드 버그가 아님. CPU에서 실제 로드/추론/출력 결과는
정확하며, 프로세스 종료 시 정리(cleanup) 단계에서만 크래시함. 이번
세션에서는 더 깊이 조사하지 않음.

## 권장 백엔드

2026-07-26에 수행한 Windows x64 Release 검증 결과를 기준으로, 1차
타겟인 Windows에서는 **TorchScript를 Phase 0 배포 백엔드로 권장**함.

검증한 Windows 11 / Visual Studio 2022 / PyTorch 2.12.1+cu126 / GTX 1080
환경에서:

- TorchScript는 CPU/CUDA 추론 및 parity 테스트를 모두 통과함.
- AOTInductor CPU는 export와 추론 자체는 정상적으로 동작했으나,
  LibTorch가 사용하는 Intel OpenMP와 AOTInductor가 생성한 코드가
  사용하는 Microsoft OpenMP 간의 런타임 충돌로 인해 프로세스 종료
  시점에 항상 크래시함.
- AOTInductor CUDA는 GTX 1080의 Compute Capability(6.1)가
  AOTInductor가 사용하는 Triton CUDA 백엔드의 최소 요구 사양보다
  낮아 export 자체가 불가능함.

앞서 진행한 macOS arm64/CPU 검증도 여전히 유효한 참고 자료임:
PyTorch 2.13.0 기준으로 AOTInductor가 그 환경에서는 end-to-end로
정상 동작했으므로, 이 백엔드 자체가 전면적으로 고장난 것은 아님을
보여줌. 다만 이 프로젝트의 1차 타겟은 Windows + NVIDIA CUDA이고,
현재 Windows 검증 결과는 AOTInductor가 아직 기본 배포 백엔드로
쓰기에는 충분히 안정적이지 않음을 보여줌.

따라서:

- **TorchScript**: Phase 0의 현재 기본/호환성 백엔드.
- **AOTInductor CPU**: 프로세스 종료 시점의 런타임 충돌이 업스트림
  또는 이후 PyTorch 버전에서 해결되기 전까지는 Windows에서 실험적
  상태로 취급.
- **AOTInductor CUDA**: 테스트 머신(GTX 1080)에서는 지원 불가;
  Compute Capability 7.0 이상 GPU에서 재평가 필요.

TorchScript는 업스트림에서 deprecated 상태이므로, AOTInductor는
장기 후보로 계속 추적하면서 PyTorch 버전, Windows 런타임 동작, 또는
대상 GPU가 바뀔 때마다 재검증해야 함.
