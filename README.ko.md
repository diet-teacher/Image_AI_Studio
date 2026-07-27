# Image AI Studio

## 장기 목표

Image AI Studio는 사용자가 UI에서 이미지 AI 모델을 설계하고, PyTorch로
학습시키고, C++ 추론 환경으로 내보낸 뒤, Python과 C++ 출력 및 성능을
비교할 수 있도록 하는 것을 목표로 합니다.

```text
모델 설계
    -> PyTorch 모델
    -> 학습/체크포인트
    -> C++ 내보내기(export)
    -> Python 추론
    -> C++ 추론
    -> Python/C++ 출력 및 성능 비교
```

## Phase 0

Phase 0은 최종 제품이 아니라 기술 스파이크(technical spike)입니다.

장기 아키텍처에서 가장 위험하고 불확실한 부분을 검증하는 것이 목적입니다.

```text
Python PyTorch 모델
    -> C++ 내보내기
    -> C++ 프로그램에서 로드
    -> 동일한 입력 텐서로 실행
    -> Python과 C++ 출력 비교
```

두 가지 배포 경로를 독립적으로 평가합니다:

1. **TorchScript** (`torch.jit.trace`)

   * 업스트림에서 이미 지원 중단(deprecated)된 상태.
   * 현재 Phase 0에서는 호환성 및 안정적인 C++ 배포 경로로 사용됨.

2. **torch.export + AOTInductor** (`.pt2`)

   * 새로운 C++ 배포 경로로 평가.
   * 가정하지 않고 실제 빌드 및 런타임 동작을 테스트함.

`run_torchscript`와 `run_aoti`는 별도의 CMake 타겟으로 빌드되는
별개의 실행 파일입니다.

AOTInductor 경로의 실패가 TorchScript 경로의 빌드나 테스트를 막아서는
안 되며, 그 반대의 경우도 마찬가지입니다.

---

## Phase 1

Phase 0 결과(`docs/phase0_results.md`)를 바탕으로, Phase 1부터는 C++
배포/추론 경로로 **TorchScript만 사용**합니다. Phase 0에서 확인된
Windows CPU 런타임 종료 시 크래시와 CUDA Compute Capability 제약
때문에 AOTInductor는 신규 Phase 1 작업에서 제외되며, 기존
AOTInductor 코드는 기록용으로만 유지되고 새 코드는 이를 참조하지
않습니다.

Phase 1에서는 앞으로 Image AI Studio의 중심이 될 **Model Definition
Layer**를 구현합니다:

```text
Model Definition
    -> Shape Inference / Validation
    -> PyTorch Model Builder
    -> torch.nn.Module
    -> TorchScript Export
    -> C++ Inference
```

전체 설계(지원 레이어, shape inference, validation, JSON 포맷, 기존
TorchScript exporter와의 연동 방식)는 `docs/phase1_design.md`를
참고하세요. Phase 1에는 PySide6 UI, 학습, IPC, Detection/Segmentation이
포함되지 않습니다.

---

## 크로스 플랫폼 범위

주요 대상 환경은 다음과 같습니다:

```text
Windows 11
Visual Studio 2022
MSVC
x64 Release
NVIDIA CUDA GPU
```

C++ 코드는 MSVC 전용 API에 의도적으로 의존하지 않는 표준 C++17로
작성되어 있어, 동일한 CMake 프로젝트를 macOS와 Linux에서도 빌드할
수 있습니다.

### Windows

주요 검증 환경입니다.

```text
CPU 추론
CUDA 추론
TorchScript
AOTInductor
C++ 패리티(parity) 테스트
```

현재 검증 결과에서는 TorchScript CPU/CUDA가 정상 동작했으며,
AOTInductor는 Windows CPU 런타임 종료(teardown) 시 발생하는 문제와
테스트에 사용된 GPU의 Compute Capability 제약이 확인되었습니다.
자세한 내용은 `docs/phase0_results.md`를 참고하세요.

### Linux

동일한 C++ 구현으로 CPU와 CUDA를 모두 지원할 것으로 예상됩니다.

Linux CUDA 검증은 아직 완료되지 않았습니다.

### macOS

CPU 전용 검증 환경입니다.

Apple Silicon은 NVIDIA CUDA를 지원하지 않으므로:

```text
--device cuda
```

는 `UNSUPPORTED`를 반환해야 합니다.

러너(runner)는 절대 조용히 CPU로 폴백해서는 안 됩니다.

실제로 실행 및 검증된 환경은 `docs/phase0_results.md`를 참고하세요.

---

## 포함된 내용

* `TinyCNN`
* `TinyResidualCNN`
* 잔차 연결(Residual connection) 테스트
* BatchNorm 실행 통계(running statistics)
* 재현 가능한 테스트 입력 텐서
* 재현 가능한 공유 `state_dict`
* 고정된 랜덤 시드
* SHA-256 아티팩트 체크섬
* Python CPU 참조 출력
* 가능한 경우 Python CUDA 참조 출력
* TorchScript trace 내보내기
* AOTInductor 내보내기
* 내보내기 환경 메타데이터
* 독립적인 C++ TorchScript 러너
* 독립적인 C++ AOTInductor 러너
* CPU FP32 추론
* 가능한 경우 CUDA FP32 추론
* Python/C++ 출력 패리티 비교
* 100회 반복 안정성 테스트
* 추론 타이밍 통계
* GPU 메모리 관측

---

## 제외된 내용

Phase 0은 다음 항목들을 의도적으로 포함하지 않습니다:

* PySide6 UI
* 모델 그래프 편집기
* 학습 루프
* `ImageFolder` 데이터셋 통합
* 공유 메모리 IPC
* 소켓 IPC
* JSON-Lines IPC
* 장기 실행 워커 프로세스
* 동적 shape(Dynamic shapes)
* ONNX Runtime
* TensorRT
* 객체 탐지(Detection)
* 세그멘테이션(Segmentation)
* 실시간 비디오 처리
* 모델 버전 관리 UI
* Git LFS
* 디버그 빌드
* LibTorch 소스 빌드

---

# 설치(Setup)

## 1. Python 환경 생성

Phase 0에서는 현재 Python 3.11을 사용합니다.

conda 사용 시:

```bash
conda create -n ias python=3.11 pip -y
conda activate ias
```

확인:

```bash
python --version
python -m pip --version
```

---

## 2. 공통 Python 의존성 설치

`requirements.txt`에는 플랫폼에 독립적인 Python 의존성만 포함되어
있습니다.

필요한 PyTorch 빌드가 운영체제와 GPU 환경에 따라 달라지므로,
PyTorch는 의도적으로 `requirements.txt`에 **포함되어 있지 않습니다**.

먼저 공통 의존성을 설치하세요:

```bash
python -m pip install -r requirements.txt
```

현재 `requirements.txt`:

```text
filelock==3.32.0
fsspec==2026.6.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
packaging==26.0
sympy==1.14.0
typing_extensions==4.16.0
```

---

## 3. PyTorch 별도 설치

PyTorch는 대상 환경에 맞게 별도로 설치해야 합니다.

이는 의도된 것입니다.

다음 명령이

```bash
pip install torch
```

원하는 CUDA 지원 빌드를 설치해줄 것이라고 가정하지 마세요.

아래 버전 번호는 설치 예시입니다. 실제 검증에 사용된 PyTorch 버전은
`docs/phase0_results.md`를 참고하세요.

### Windows + NVIDIA CUDA

환경에 맞는 CUDA 지원 PyTorch wheel을 설치하세요.

예시:

```bat
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

설치 후 확인:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'None')"
```

CUDA 지원 환경이라면 다음 결과가 중요합니다:

```text
CUDA available: True
```

기본적인 CUDA 연산도 테스트할 수 있습니다:

```bat
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print(y.device); print(y.mean())"
```

### macOS

표준 macOS PyTorch 패키지를 설치하세요:

```bash
python -m pip install torch==2.12.1
```

macOS에서는 CUDA를 사용할 수 없을 것으로 예상됩니다.

### Linux + NVIDIA CUDA

대상 Linux CUDA 환경에 맞는 PyTorch CUDA wheel을 설치하세요.

정확한 wheel은 `requirements.txt`에 하드코딩하는 대신, 목표로 하는
PyTorch 및 CUDA 구성에 따라 선택해야 합니다.

---

## 4. Image AI Studio를 editable 모드로 설치

이 프로젝트는 `src/` Python 패키지 레이아웃을 사용합니다.

프로젝트 자체를 현재 Python 환경에 설치하세요:

```bash
python -m pip install -e .
```

`-e`는 editable(수정 가능) 설치를 의미합니다.

Python이 현재 소스 트리를 직접 참조하므로,

```text
src/image_ai_studio/
```

아래의 변경 사항은 패키지를 매번 재설치하지 않아도 반영됩니다.

확인:

```bash
python -c "import image_ai_studio; print(image_ai_studio.__file__)"
```

경로는 이 저장소의

```text
src/image_ai_studio/
```

디렉터리를 가리켜야 합니다.

---

## 5. 환경 점검

실행:

```bash
python scripts/inspect_environment.py
```

환경 점검 결과에는 다음 정보가 포함되어야 합니다:

```text
Python 버전
PyTorch 버전
PyTorch CUDA 버전
CUDA 사용 가능 여부
GPU 이름
GPU compute capability
CUDA Toolkit
NVIDIA 드라이버
CMake
컴파일러
LibTorch 위치
```

---

# C++ 빌드를 위한 LibTorch

기본 Phase 0 워크플로우에서는 별도의 LibTorch 다운로드가 필요하지
않습니다.

C++ 빌드는 설치된 Python `torch` 패키지에 번들로 포함된 LibTorch
파일을 사용합니다.

CMake 경로는 다음으로 확인할 수 있습니다:

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

이를 통해 Python과 C++ 양쪽이 동일한 PyTorch 설치를 사용하게 되어,

```text
Python PyTorch
C++ LibTorch
```

사이의 버전 불일치를 방지할 수 있습니다.

예를 들어 Windows에 CUDA 지원 PyTorch wheel을 설치하면, C++ 빌드도
동일한 PyTorch 설치에 번들된 LibTorch를 사용하게 됩니다.

---

# Windows C++ 빌드 구성

Phase 0의 주요 C++ 타겟은 다음과 같습니다:

```text
플랫폼: x64
구성: Release
```

디버그 빌드는 Phase 0 범위에서 의도적으로 제외됩니다.

Debug와 Release MSVC CRT/ABI 조합을 혼용해서는 안 됩니다.

호환되지 않는 Debug/Release 조합으로 인한 구성 오류는 TorchScript나
AOTInductor 백엔드 실패가 아니라 다음으로 분류됩니다:

```text
INVALID_BUILD_CONFIGURATION
```

자세한 빌드 지침은 다음을 참고하세요:

```text
docs/build.md
```

---

# AOTInductor 지원 여부 프로브(probe)

전체 AOTInductor C++ 러너를 시도하기 전에, 설치된 LibTorch 배포판에
필요한 AOTInductor C++ 런타임이 포함되어 있는지 확인하세요.

실행:

```bash
python scripts/probe_aoti_support.py
```

그런 다음 독립된 C++ 프로브를 빌드하세요:

```bash
python scripts/build_aoti.py --build-dir build-aoti-probe --probe-only
```

패키지 테스트 예시:

```bash
./build-aoti-probe/cpp/aoti_probe/probe_aoti \
  --package artifacts/aoti/tiny_cnn/cpu/model.pt2 \
  --input-bin artifacts/common/input.bin \
  --input-meta artifacts/common/input.json
```

Windows에서는 실행 파일이 해당하는 `.exe` 경로를 사용합니다.
Visual Studio의 멀티 컨피그 generator를 사용하는 경우 다음과 같이
`Release\` 하위 경로가 될 수 있습니다:

```bat
build-aoti-probe\cpp\aoti_probe\Release\probe_aoti.exe ^
  --package artifacts\aoti\tiny_cnn\cpu\model.pt2 ^
  --input-bin artifacts\common\input.bin ^
  --input-meta artifacts\common\input.json
```

가능한 지원 상태(capability states)는 다음과 같습니다:

```text
HEADER_NOT_FOUND
LIBRARY_OR_SYMBOL_NOT_FOUND
COMPILE_FAILED
LINK_FAILED
PACKAGE_LOAD_FAILED
SUPPORTED
```

AOTInductor 지원 여부 프로브가 실패해도 TorchScript의 빌드나 테스트를
막지 않습니다.

실제 결과는 `docs/phase0_results.md`를 참고하세요.

---

# 테스트 아티팩트 생성

결정론적(deterministic) 모델 가중치와 테스트 입력을 생성하세요:

```bash
python -m image_ai_studio.tools.prepare_test_artifacts
```

Python 참조 출력을 생성하세요:

```bash
python -m image_ai_studio.tools.run_python_reference
```

모델을 내보내세요:

```bash
python scripts/export_models.py
```

이 과정을 통해 C++ 러너에 필요한 아티팩트가 생성됩니다.

---

# C++ 러너 빌드

TorchScript와 AOTInductor는 독립적으로 빌드됩니다.

## TorchScript

```bash
python scripts/build_torchscript.py
```

예상 출력:

```text
build-torchscript/.../run_torchscript
```

Windows에서는:

```text
run_torchscript.exe
```

Visual Studio처럼 멀티 컨피그 generator를 사용하는 Windows에서는
실행 파일이 `Release/` 하위(예: `build-torchscript/.../Release/run_torchscript.exe`)에
생성될 수 있습니다.

## AOTInductor

```bash
python scripts/build_aoti.py
```

예상 출력:

```text
build-aoti/.../run_aoti
```

Windows에서는:

```text
run_aoti.exe
```

마찬가지로 `build-aoti/.../Release/run_aoti.exe` 형태가 될 수 있습니다.

각 백엔드는 자체 빌드 디렉터리를 사용합니다.

AOTInductor 빌드가 깨지더라도 다음에는 영향을 주지 않아야 합니다:

```text
build-torchscript/
```

---

# 테스트 실행

## Phase 1 Model Definition Layer (unit test)

```bash
python -m pip install -r requirements-dev.txt
pytest
```

이 테스트들은 전부 CPU에서 동작하며 빌드된 C++ 러너가 필요 없습니다.
자세한 내용은 `docs/phase1_design.md`를 참고하세요.

## TorchScript (Phase 0 C++ 패리티)

```bash
python scripts/run_torchscript_tests.py
```

## AOTInductor

```bash
python scripts/run_aoti_tests.py
```

CUDA를 사용할 수 있는 경우, 테스트에는 CPU와 CUDA 패리티가 포함됩니다.

CUDA를 사용할 수 없는 경우, CUDA 테스트는 조용히 CPU로 폴백되지 않고
건너뜀(skipped) 또는 미지원(unsupported)으로 보고됩니다.

---

# 전체 Phase 0 워크플로우 실행

실행:

```bash
python scripts/run_phase0.py
```

이 워크플로우는 다음을 수행합니다:

```text
환경 점검
    -> AOTInductor 지원 여부 프로브
    -> 결정론적 모델/입력 생성
    -> Python 참조 추론
    -> TorchScript 내보내기
    -> AOTInductor 내보내기
    -> C++ 빌드
    -> C++ 추론
    -> Python/C++ 패리티 비교
    -> 반복 안정성 테스트
    -> 결과 생성
```

AOTInductor 단계에서의 실패는 TorchScript 경로의 진행을 막지 않습니다.

---

# 알려진 제한 사항

* TorchScript 검증은 현재 정적(static) `torch.jit.trace` 경로만
  다룹니다.

* `torch.jit.script`는 Phase 0 범위 밖입니다.

* AOTInductor API는 PyTorch 버전에 따라 달라질 수 있습니다.

* 일부 AOTInductor compile/package/load API는 내부/비공개 모듈인
  `torch._inductor` 아래에 있을 수 있으며, 버전에 민감한 것으로
  취급해야 합니다.

* 현재 공유 바이너리 + JSON 텐서 포맷은 float32 텐서만 지원합니다.

* macOS 검증은 CPU 전용입니다.

* Windows x64 Release + NVIDIA CUDA가 주요 타겟 환경이며, 현재
  검증 결과에서는 TorchScript가 Phase 0의 기본 배포 백엔드로
  권장됩니다 (자세한 내용은 `docs/phase0_results.md`의 "권장 백엔드"
  참고).

* macOS CPU에서 동작하는 백엔드가 Windows MSVC/CUDA 경로도
  지원됨을 자동으로 의미하지는 않습니다.

---

# 의존성 정책

`requirements.txt`에는 의도적으로 PyTorch가 포함되어 있지 않습니다.

의존성 전략은 다음과 같습니다:

```text
requirements.txt
    -> 공통 Python 의존성

PyTorch
    -> OS/GPU 구성별로 별도 설치

pip install -e .
    -> Image AI Studio 소스 패키지 자체를 설치
```

이를 통해 저장소가 하나의 PyTorch CPU/CUDA wheel에 종속되는 것을
방지하고, 플랫폼별 환경을 명시적으로 유지합니다.
