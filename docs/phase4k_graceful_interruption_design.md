# Phase 4K: Graceful SIGINT and Cooperative Training Stop — 설계안

**상태: 구현, 자동 검증, 수동 SIGINT acceptance 완료.**
`scripts/train_imagefolder.py`의 Ctrl+C(SIGINT)를
`should_stop()` 기반 cooperative stop으로 연결해, 이미 Phase 4I(`should_stop`)와
Phase 4J(epoch checkpoint/final checkpoint)가 완성해 둔 기반을 CLI 운영
측면에서 완결하는 설계다. **Phase 4K production 코드와 자동 테스트는
구현 및 검증되었고, 실제 Windows 터미널(Anaconda Prompt/cmd)에서의 단일
Ctrl+C graceful stop 수동 acceptance도 PASS로 확인됐다.** 두 번째 Ctrl+C
강제 종료 경로는 자동화 테스트로 검증되어 있으나, 실제 터미널에서의
안정적인 수동 재현은 이번 라운드의 fixture 특성상 이뤄지지 못했다 —
구현 결함이 아니라 수동 재현 자체의 타이밍 제약이다(§14 참고).

**전제**: 이 문서는 별도 조사·비교 검토 라운드(Phase 4K 후보 A~G 비교, 후보 E
선정)를 거친 뒤, 채팅으로 확정된 정책을 그대로 문서화한 것이다. 정책 자체에
대한 재검토는 이 문서의 목적이 아니다 — 이미 확정된 정책을 구현 가능한
수준의 설계로 구체화하는 것이 목적이다.

---

## 1. 현재 구조와 SIGINT 동작

### 1-1. `should_stop()` 평가 조건 (`src/image_ai_studio/training/loop.py`)

`run_training()`의 epoch 루프에서 `should_stop()`은 다음 코드로만 평가된다:

```python
has_next_epoch = run_epoch < config.epochs
if (
    not history.stopped_early
    and has_next_epoch
    and should_stop is not None
    and should_stop()
):
    history.stopped_by_user = True

if history.stopped_early or history.stopped_by_user:
    break
```

호출 순서는 `train_one_epoch → evaluate → history 기록 → best/카운터 갱신 →
scheduler.step() → early stopping 판정 → checkpoint_hook → progress_callback →
should_stop 평가 → break 판정`이다(Phase 4I/4J가 확정한 순서, 이번 Phase는
이 순서를 전혀 바꾸지 않는다).

핵심 조건은 `has_next_epoch = run_epoch < config.epochs`다 — **이번
`run_training()` 호출의 마지막 요청 epoch에서는 `should_stop()`이 물리적으로
호출되지 않는다.** early stopping이 이미 발동한 epoch에서도 호출되지 않는다.

### 1-2. `history.stopped_by_user=True`가 설정되는 정확한 위치

위 코드 블록의 `history.stopped_by_user = True`가 유일한 대입 지점이다.
`should_stop()`이 `True`를 반환하고, `has_next_epoch`이 `True`이고,
`early_stopping`이 이미 발동하지 않은 경우에만 설정된다. resume 시에는
`run_training()`이 `resume_state.history.stopped_by_user`를 항상 `False`로
리셋한다(이번 호출은 아직 멈춘 적이 없으므로).

### 1-3. workflow가 `run_training()` 반환 후 저장하는 순서 (`imagefolder_workflow.py`)

```text
cpu_rng_state_after / loader_generator_state_after 캡처
→ save_training_history()                       (training_history.json)
→ save_class_mapping()                           (class_mapping.json)
→ (checkpoint_out이 있으면) ensure_checkpoint_metadata() → save_training_checkpoint()  (원자적)
→ best_model 빌드 → save_state_dict()            (best_model_state_dict.pt)
→ evaluate(test_loader)                          (test_result.json)
→ (export_torchscript면) TorchScriptExporter().export(...)  (model.ts / model_metadata.json)
```

이 구간 전체는 **`should_stop()`을 다시 호출하지 않는다** — `run_training()`이
이미 반환되어 epoch 루프 자체가 끝났기 때문이다. 즉 이 구간에서 SIGINT는
"중단"이 아니라 그저 "지금 실행 중인 파일 저장/평가/export 작업"과 경쟁하는
관계가 된다.

### 1-4. `checkpoint.py`의 원자적 저장과 `BaseException`

```python
def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
```

`except BaseException`이므로 `KeyboardInterrupt`도 포함해서 잡는다.
`os.replace()` 이전에 어떤 예외가 나든(실제 `KeyboardInterrupt` 포함) 목적지
파일은 전혀 건드려지지 않고 임시 파일만 정리된다. **정확한 표현**: 동일
디렉터리에서 수행하는 `os.replace()`는 지원되는 파일시스템에서 원자적
이름 교체를 제공하므로, 애플리케이션 관점에서 대상 checkpoint가 부분적으로
기록된 상태로 노출되지 않는다(`os.replace()` 호출이 반환되기 전이면 기존
파일이 그대로 있고, 반환된 후면 새 파일로 완전히 교체되어 있다 — 그
중간의 "일부만 바뀐" 상태를 다른 프로세스/재시작된 프로세스가 관찰할 수
없다). **이 보호는 Phase 4J에서 이미 완성됐으며, 이번 Phase는 이를 그대로
재사용할 뿐 아무것도 새로 만들지 않는다.** `save_imagefolder_resume_metadata()`가
쓰는 `_atomic_write_text()`(`imagefolder_resume.py`)도 동일한 계약이다.

### 1-5. Phase 4K 구현 전 `scripts/train_imagefolder.py`의 예외 처리/exit code

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("ImageFolder Training")
    ...
    try:
        training_config = TrainingConfig(...)
        request = ImageFolderWorkflowRequest(...)
        result = run_imagefolder_training_workflow(request, progress_callback=_print_progress)
    except (ModelValidationError, TrainingConfigError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    ...
    return 0
```

Phase 4K 구현 전에는 `KeyboardInterrupt`를 잡는 코드가 어디에도 없었다 —
`except` 튜플에도 없었고(애초에 `KeyboardInterrupt`는 `Exception`의
서브클래스가 아니라 이 튜플로는 절대 잡히지 않는다), 별도
`except KeyboardInterrupt`도 없었다. 구현 전 `import` 목록에도
`signal`/`os` 모듈이 없었다(`argparse`, `sys`, `pathlib.Path`만 사용).
이 상태는 §9(구현된 handler 설치/복원)/§12-1(구현된 `_SigintStopController`)에서
설명하는 실제 구현으로 대체되었다.

### 1-6. `main()`이 테스트에서 반복 호출되는지

`tests/scripts/test_train_imagefolder_cli.py`는 `cli.main([...])`을 동일한
pytest 프로세스의 **메인 스레드**에서 **반복 호출**한다(`subprocess`
사용 없음, 정확한 호출 횟수는 테스트가 추가/삭제될 때마다 바뀌므로 이
문서에서 고정 숫자로 명시하지 않는다). 이는 (a) `signal.signal()`을
테스트에서 실제로 호출해도 `ValueError`(비메인스레드 제약)에 걸리지
않는다는 뜻이며, (b) handler 설치 후 복원을 정확히 하지 않으면 이후
호출들과 pytest 프로세스 자체에 영향이 새어나간다는 뜻이기도 하다 —
`try/finally` 복원이 필수인 이유가 실제 테스트 구조에서 직접 확인된다.

### 1-7. Windows에서 Python `signal` API의 적용 범위

Python `signal` 모듈은 Windows에서 `SIGINT`를 포함한 제한된 신호 집합을
지원하며, `signal.signal(signal.SIGINT, handler)`는 콘솔 애플리케이션의
**메인 스레드**에서 Ctrl+C를 가로채는 표준적인 방법이다(Python 표준 문서
기준의 일반 지식 — 이 저장소 코드로 검증된 사실이 아니라 플랫폼/인터프리터
동작이므로 이 문서에서는 그렇게 명시적으로 구분한다). `signal.signal()`은
**메인 인터프리터의 메인 스레드에서만 호출 가능**하며, 그 외에서 호출하면
`ValueError`가 발생한다. `train_imagefolder.py`는 항상 스크립트 진입점
(`if __name__ == "__main__":`)이거나 §1-6에서 확인한 대로 테스트의 직접
함수 호출(메인 스레드)로만 실행되므로 실사용/테스트 양쪽에서 이 제약에
걸리지 않는다.

**신호 전달 시점에 대한 정확한 서술(과장 금지)**: 첫 번째 SIGINT가 OS에서
프로세스로 전달되면, CPython은 이를 "pending"으로 표시해 두고, 인터프리터의
bytecode 평가 루프가 다음 안전 지점에 도달했을 때 등록된 Python 레벨
handler(`controller.handle_signal`)를 실행한다. **handler 자체는 예외를
던지지 않으므로 flag만 설정하고 즉시 반환하며, 실행 중이던 코드는 handler가
반환된 뒤 중단 없이 이어서 실행된다.** 다만 **Python이 오래 걸리는 C/CUDA
호출(예: 하나의 큰 `torch.save()`/CUDA 커널 동기화/블로킹 I/O) 안에 있는
동안은 그 호출이 Python 평가 루프로 제어를 돌려주기 전까지 handler 실행
자체가 지연될 수 있다** — "첫 번째 Ctrl+C가 모든 작업을 즉시 방해하지
않는다"는 것은 handler가 예외를 던지지 않기 때문이지, 신호가 즉시
처리된다는 뜻이 아니다. 이 지연은 사용자에게는 "안내 메시지가 즉시 뜨지
않을 수 있다"는 형태로 관찰될 수 있으며, 이는 문서화된 한계로 §15(위험
요소)에 남긴다.

### 1-8. Phase 4K 구현 전 `KeyboardInterrupt`가 CLI에서 잡혔는지

§1-5에서 확인한 대로 구현 전에는 잡히지 않았다 — 그대로 전파되어
`main()`을 완주하지 못하게 하고, `if __name__ == "__main__": raise
SystemExit(main())`도 실행되지 못한 채 인터프리터가 처리했다(정확한
종료 코드는 플랫폼/셸에 따라 다르며 이 저장소가 그 시점에는 명시적으로
통제하지 않았음). 이 Phase는 `main()` 실행 전체에서 발생하는
`KeyboardInterrupt`를 exit code 130으로 명시적으로 통제하는 것을
목표로 했고(§9-3의 구조, §10의 exit code 정책), 실제 구현도 이 목표를
그대로 따랐다.

---

## 2. 목표와 비목표

### 2-1. 목표

- CLI 실행 중 첫 번째 Ctrl+C를 즉시 `KeyboardInterrupt`로 종료시키지 않고
  stop request로 변환한다.
- 현재 실행 중인 epoch가 정상 완료되고, 다음 유효한 `should_stop()` 평가
  지점에 도달하면 학습을 cooperative하게 중단한다.
- 이후 기존 workflow의 final artifact/checkpoint 저장 경로를 정상적으로
  수행한다.
- `checkpoint_out`이 있다면 최종 checkpoint에는 `stopped_by_user=True`가
  저장된다(단, §7에서 정의하는 마지막 epoch 예외가 있음).
- 두 번째 Ctrl+C는 즉시 강제 종료할 수 있다.

### 2-2. 비목표

- batch 중간 checkpoint/resume
- `SIGTERM`/`SIGHUP` 처리
- GUI stop button
- background worker/멀티스레드 학습
- Windows console close event(`WM_CLOSE`/콘솔 핸들러 이벤트) 처리
- artifact 저장 도중 첫 번째 Ctrl+C까지 graceful하게 지연시키는 범용
  cancellation framework
- 강제 종료(두 번째 SIGINT) 전에 새로운 checkpoint를 추가로 저장하는 로직
- signal handler 안에서의 다음 작업:
  - checkpoint/artifact 파일 I/O
  - PyTorch/model/optimizer/generator 접근
  - logging/동적 formatting(문자열 조합, `f-string`, `%` 포매팅 등)
  - 사용자 안내용 **고정 bytes**의 저수준 stderr 출력(`os.write(2, ...)`)만
    예외적으로 허용(§5-1)
- core `run_training()`의 epoch 순서 변경
- checkpoint 포맷 버전 변경
- "지금 정확히 어느 단계에서 interrupt가 왔는지"를 구분해 알려주는 phase
  tracking(§7-3에서 이유를 구체적으로 설명)
- 실제 OS SIGINT를 subprocess로 보내는 자동화 E2E 테스트(§13에서 선택
  사항으로 명시)

---

## 3. 정확한 보장 범위

> 첫 번째 SIGINT는 CLI 전용 private controller의 stop flag를 `True`로
> 설정한다. 이 flag는 예외 없이 설정되며, 고정된 안내 메시지의 저수준
> stderr 출력을 정확히 한 번 시도한다(출력 실패나 partial write
> 가능성은 stop flag 설정을 방해하지 않는다, §5-2). 학습 중이었고
> 이번 `run_training()` 호출에 다음
> epoch가 남아 있었다면(`run_epoch < config.epochs`), 현재 epoch가 정상
> 완료된 뒤 `should_stop()` 평가 지점에서 학습이 중단된다. **이번 호출의
> 마지막 요청 epoch 중이었거나, `run_training()`이 이미 반환되어
> artifact/checkpoint 저장·평가·TorchScript export 단계에 들어간 뒤라면,
> 첫 번째 SIGINT는 해당 단계를 중단시키지 않는다** — flag는 설정되지만
> 그 시점 이후로는 아무도 다시 `should_stop()`을 평가하지 않기 때문이다.
> 그 경우 현재 진행 중인 작업이 정상적으로 완료되고 exit code 0으로
> 끝난다. batch 중간 resume이나 진행 중인 파일 저장/export의 즉시 중단은
> 보장하지 않는다. 두 번째 SIGINT는 즉시 `KeyboardInterrupt`를 발생시켜
> 강제 종료하며, 이 경우 final checkpoint/artifact 저장은 보장되지 않지만
> **마지막으로 원자적 저장이 완료된 유효한 checkpoint는 보존되며, 기존
> 파일과 새 파일의 일부가 섞인 checkpoint는 노출되지 않는다**(§1-4).

### 3-1. 시나리오별 동작 표

| 시나리오 | stop flag | 현재 epoch 완료 | `stopped_by_user` | final checkpoint | artifact 저장 | exit code | 데이터 손상 가능성 |
|---|---|---|---|---|---|---|---|
| train batch 처리 중 1차 Ctrl+C | Set | Yes(핸들러가 예외를 던지지 않으므로 방해 없음, §1-7 지연 가능성 있음) | True(다음 epoch 있으면) | Yes(있다면) | Yes | 0 | 없음 |
| validation 중 1차 Ctrl+C | Set | Yes | True(〃) | Yes | Yes | 0 | 없음 |
| checkpoint hook 저장 중 1차 Ctrl+C | Set | Yes(저장도 방해 없이 완료) | True(〃) | Yes | Yes | 0 | 없음(`_atomic_torch_save`가 예외 자체를 받지 않음) |
| progress callback 중 1차 Ctrl+C | Set | Yes | True(〃) | Yes | Yes | 0 | 없음 |
| should_stop 평가 직전 1차 Ctrl+C | Set(평가 시 즉시 읽힘) | Yes | True(〃) | Yes | Yes | 0 | 없음 |
| **마지막 요청 epoch 중 1차 Ctrl+C** | Set(아무도 읽지 않음) | Yes | **False**(§7) | Yes(정상 완료로) | Yes | 0 | 없음 |
| `run_training()` 반환 후 artifact 저장 중 1차 Ctrl+C | Set(효과 없음) | 이미 완료 | 이미 결정됨 | Yes(방해 없음) | Yes(방해 없음) | 0 | 없음 |
| TorchScript export 중 1차 Ctrl+C | Set(효과 없음) | 이미 완료 | 이미 결정됨 | Yes | Yes(export도 완료) | 0 | 없음 |
| 1차 이후 2차 Ctrl+C | `default_int_handler` 호출 → `KeyboardInterrupt` | 실행 위치에 따라 다름(중단될 수 있음) | 실행 위치에 따라 다름(미완료 가능) | **보장 안 됨** | **보장 안 됨** | 130 | `_atomic_torch_save` 실행 중이면 없음(§1-4), 그 외(JSON write, TorchScript export)는 불완전 파일 가능(§9 비목표) |
| checkpoint 저장 실패(디스크 등) 중 Ctrl+C | Set 또는 무관 | — | — | 실패(원래 예외 그대로 전파) | — | 1(일반 오류 경로) 또는 130(KeyboardInterrupt와 겹치면) | 없음(원자적 보호가 signal 유무와 무관하게 항상 적용) |
| `checkpoint_out=None`에서 1차 Ctrl+C | Set | Yes | True(다음 epoch 있으면) | 해당 없음 | Yes(checkpoint 제외 나머지 전부) | 0 | 없음 |

---

## 4. Controller 상태 전이

`_SigintStopController`(§8, §11에서 위치/코드 확정)는 다음 두 개의 상태만
가진다.

```text
초기 상태: _interrupt_requested = False

handle_signal() 1차 호출:
    _interrupt_requested: False → True
    고정 stderr 메시지 출력을 정확히 1회 시도(출력 성공 여부와 무관)
    예외를 던지지 않고 반환

handle_signal() 2차(이후) 호출:
    _interrupt_requested는 이미 True (변화 없음)
    signal.default_int_handler(signum, frame) 호출 → KeyboardInterrupt 발생

should_stop():
    return self._interrupt_requested  (읽기 전용, 부수효과 없음)
```

세 번째 이후 호출은 두 번째와 동일하게 취급한다(별도 상태를 두지 않음 —
`default_int_handler`가 매번 `KeyboardInterrupt`를 일으키므로 3차 호출이
실제로 일어날 상황 자체가 거의 없다: 2차 호출이 이미 예외를 던지므로 그
직후 프로세스가 종료 절차에 들어간다).

---

## 5. 첫 번째 SIGINT

### 5-1. 정책(확정)

- private controller의 `_interrupt_requested`를 `True`로 설정
- 예외를 발생시키지 않음
- 고정된 stderr 안내 메시지의 저수준 출력을 **정확히 한 번 시도**(완전한
  출력을 보장하지는 않는다, §5-2)
- 다음으로 실제 평가되는 `should_stop()` 호출 지점에서 cooperative stop
- 정상 artifact/final checkpoint 저장까지 완료되면 exit code 0
- `checkpoint_out=None`이면 checkpoint 없이 나머지 artifact를 그대로 저장
- signal handler 안에서는 **bool 변경과 고정 안내 출력 외의 어떤 작업도
  하지 않는다** — checkpoint/artifact 파일 I/O, PyTorch/model/optimizer/
  generator 접근, logging/동적 formatting 전부 금지. 사용자 안내용
  **고정 bytes**를 `os.write(2, ...)`로 저수준 stderr에 쓰는 것만 예외로
  허용한다(§5-2에서 정확한 이유와 구현을 설명).

### 5-2. 안내 메시지: 고정 bytes를 저수준 stderr에 직접 쓴다

**`print()`는 signal handler 안에서 쓰지 않는다.** `print()`/`sys.stderr.write()`는
내부적으로 버퍼링된 텍스트 스트림 객체(`io.TextIOWrapper`)에 대한 락(lock)을
잡고 인코딩/버퍼 상태를 갱신하는 연산이다 — 만약 메인 실행 흐름이 이미 같은
스트림에 쓰기 작업을 하던 도중(예: `_print_progress()`가 `print()`로 stdout에
쓰는 중, 또는 다른 코드가 `sys.stderr`에 쓰는 중) signal handler가 재진입해
같은 객체의 락/버퍼를 다시 건드리면, CPython 구현 세부사항에 따라 안전성이
보장되지 않는 재진입 문제가 생길 수 있다. 이 위험을 원천적으로 피하기 위해
**고정 bytes를 파일 디스크립터에 직접 쓰는 `os.write()`**를 사용한다 —
`os.write()`는 텍스트 인코딩이나 Python 레벨 버퍼링 없이 곧바로 OS
syscall(`write(2)`)을 호출하므로, 미리 인코딩해 둔 고정 bytes를 넘기면
동적 문자열 조합/인코딩 같은 추가 연산 없이 안전하게 출력할 수 있다.

```text
Interrupt requested. Training will stop at the next safe epoch boundary.
If training has already finished, remaining output work will complete normally.
Press Ctrl+C again to terminate immediately.
```

```python
_INTERRUPT_MESSAGE_BYTES = (
    b"\nInterrupt requested. Training will stop at the next safe epoch boundary.\n"
    b"If training has already finished, remaining output work will complete normally.\n"
    b"Press Ctrl+C again to terminate immediately.\n"
)
```

`handle_signal()`은 `os.write(2, _INTERRUPT_MESSAGE_BYTES)`를 호출한다(§12-1).
파일 디스크립터를 고정 정수 `2`(표준 stderr)로 쓸지, `sys.stderr.fileno()`를
`main()` 진입 시점(정상적인 실행 흐름, signal handler 밖)에 미리 조회해
캡처해 둔 값을 쓸지 두 방식을 비교했다 — 이 CLI에서 `sys.stderr`가 재배선될
일이 없고, `fileno()` 조회 자체를 signal handler 안에서 수행하지 않는 것이
중요하므로(동적 조회도 "얕은 연산이라 안전"이라고 단정하지 않는다), **고정
정수 `2`를 그대로 쓰는 것을 기본안으로 확정**한다 — 이것이 표준 stderr를
가리킨다는 것은 POSIX/Windows 양쪽에서 안정적인 규약이다. `os.write()`
자체가 실패할 가능성(예: stderr가 닫힌 파이프)에 대비해 `try/except OSError:
pass`로 감싼다(§12-1 코드 참고) — 안내 메시지 출력 실패가 stop flag 설정
자체를 막아서는 안 되기 때문이다.

**출력 보장 범위에 대한 정확한 표현(과도한 보장 금지)**: `os.write()`는
이론적으로 요청한 bytes 전부보다 적게 쓰고 반환할 수 있다(partial write).
이 설계는 그 나머지를 다시 쓰는 재시도 루프를 **signal handler 안에
추가하지 않는다** — 그런 루프는 handler를 더 오래 실행시키고 복잡하게
만들어 signal handler를 "짧고 얕게" 유지한다는 원칙과 충돌한다. 따라서
이 설계가 실제로 보장하는 바는 다음과 같이 정확히 좁혀 표현한다:

> 첫 번째 SIGINT에서 고정 안내 메시지의 저수준 stderr 출력을 정확히 한 번
> 시도한다. 첫 번째 안내를 반복 출력하지 않는다. 출력 실패 또는 부분
> 출력 가능성은 stop flag 설정을 방해하지 않는다.

일반적인 터미널 실행이나 자동 테스트의 `capfd` 환경(§13-1)에서는 메시지
크기가 작아 실제로 항상 한 번에 전부 출력되는 것을 관찰할 수 있지만,
**이를 "모든 OS/파이프 상황에서 전체 bytes가 반드시 완전히 출력된다"는
production 계약으로 정의하지 않는다** — `os.write()`가 반환한 길이를
검사해 반복 출력하거나 나머지를 다시 쓰는 로직은 구현하지 않는다(§9
비목표 목록의 "signal handler 안에서의 파일 I/O 금지"와 같은 이유로,
handler는 항상 1회의 얕은 syscall만 시도한다).

메시지 내용 자체에 "현재 epoch 이후" 대신 "next safe epoch boundary"라는
표현을 쓰는 이유는 §7/§8에서 설명하는, controller가 정확히 어느 단계인지
구분하지 못하는 상황(마지막 epoch, 저장 단계, export 단계)까지 전부
포괄하기 위함이다(이 부분은 기존 설계 의도 그대로 유지).

### 5-3. RNG/state-purity 계약과의 관계

`controller.should_stop`은 기존 `ShouldStopCallback` 계약(Phase 4I §3-5,
Phase 4J §3-5)을 그대로 따른다 — 외부 stop flag를 읽어 bool을 반환하는
용도로만 쓰이고, PyTorch RNG를 소비하지 않으며, model/optimizer/scheduler/
DataLoader generator를 변경하지 않는다. `handle_signal()` 자체도 동일한
계약을 만족해야 한다(bool 대입 + 메시지 출력뿐이므로 자연히 만족).

---

## 6. 두 번째 SIGINT

### 6-1. 정책(확정)

- 캡처해 둔 이전 handler(`previous_handler`)를 **호출하지 않는다.**
- 대신 `signal.default_int_handler(signum, frame)`를 호출해
  `KeyboardInterrupt`를 발생시킨다.
- `previous_handler`는 오직 workflow 종료(정상/예외 무관) 후 원래 SIGINT
  handler를 **복원**하는 용도로만 쓰인다 — 두 번째 SIGINT의 escalation
  경로에서는 전혀 참조되지 않는다.
- `os._exit()`는 사용하지 않는다(cleanup/finally를 건너뛰므로).

### 6-2. 왜 `previous_handler`를 호출하지 않고 `default_int_handler`를 쓰는가

두 후보(A: 캡처된 이전 handler 직접 호출, B: `signal.default_int_handler`
직접 호출) 중 **B로 확정**한다. 이전 handler를 호출하는 A는 이 CLI가 어떤
임베딩 환경(다른 handler가 이미 설치된 상태)에서 실행되는 극히 드문 경우를
위한 일반성을 제공하지만, 실사용 100%에 해당하는 "터미널에서 직접
`python scripts/train_imagefolder.py` 실행" 시나리오에서 `previous_handler`는
사실상 항상 `signal.default_int_handler` 그 자체다. B를 직접 호출하면
동작이 항상 예측 가능하고(`KeyboardInterrupt` 발생이 보장됨), `previous_handler`가
`signal.SIG_IGN`/`signal.SIG_DFL`처럼 호출 불가능한 정수 상수인 경우를
별도로 가드할 필요도 없다 — B는 handler 참조 자체와 무관하게 항상 안전하게
동작하는 표준 라이브러리 함수다. `previous_handler`는 §9(handler 설치와
복원)에서 정의하는 대로 workflow 종료 후 원상 복구에만 쓰인다.

### 6-3. 강제 종료 시 보장 범위

두 번째 SIGINT(또는 그 밖의 어떤 이유로든 발생하는 `KeyboardInterrupt`)로
강제 종료하면:

- final checkpoint 저장은 보장하지 않는다(진행 중이었다면 중단될 수 있음).
- artifact 저장 완료는 보장하지 않는다.
- **마지막으로 원자적 저장이 완료된 유효한 checkpoint는 보존되며, 기존
  파일과 새 파일의 일부가 섞인 checkpoint는 노출되지 않는다**(§1-4,
  `_atomic_torch_save`의 `except BaseException` 계약 + `os.replace()`의
  원자적 이름 교체 — signal handler 설계와 무관하게 이미 성립하는 사실).
- JSON(`training_history.json`/`class_mapping.json`/`test_result.json`)이나
  TorchScript export(`model.ts`/`model_metadata.json`) 등 **원자적 저장을
  쓰지 않는 출력물은 두 번째 SIGINT 도중이면 불완전한 상태로 남을 수 있다.**
  이를 원자적으로 보호하는 것은 명시적 비목표다(§2-2).

---

## 7. 마지막 epoch 정책

### 7-1. 확정 정책

`loop.py`의 다음 조건은 **변경하지 않는다**(§1-1 그대로):

```python
has_next_epoch = run_epoch < config.epochs
```

마지막 요청 epoch에서는 `should_stop()`이 호출되지 않으므로, 마지막 epoch
중 첫 번째 SIGINT가 들어와도:

- 마지막 epoch는 정상 완료된다.
- `history.stopped_by_user`는 `False`일 수 있다(그 SIGINT가 유일한 요청이고
  달리 멈출 이유가 없었다면 항상 `False`).
- final artifact/checkpoint는 정상 저장된다.
- exit code는 0이다.

### 7-2. 왜 core를 바꾸지 않는가

`has_next_epoch` 게이트는 Phase 4I가 "더 이상 건너뛸 epoch이 없으면 평가
자체가 의미 없다"는 명시적 설계 결정으로 만든 불변조건이다. 이를 바꾸면
(마지막 epoch에서도 stop flag를 관찰하도록) 실행되는 epoch 수는 동일하게
유지되면서 `stopped_by_user`의 "의미"만 바뀐다 — core의 exact-resume
불변조건(Phase 4I/4J가 정밀하게 검증해 둔 `loop.py`의 제어 흐름)을 순수
문구적 정확성 하나를 위해 다시 여는 것은 이번 Phase의 "core 무수정" 원칙에
반한다.

### 7-3. CLI가 "마지막 epoch에 interrupt가 들어왔다"는 메시지를 만들지 않는 이유

controller가 가진 정보는 `_interrupt_requested`라는 단일 bool뿐이다. 이
값만으로는 SIGINT가 정확히 다음 중 어느 시점에 들어왔는지 구분할 수 없다:

- 마지막 학습 epoch 진행 중
- `run_training()` 반환 직후
- checkpoint/artifact 저장 중
- TorchScript export 중

이 네 가지를 구분하려면 controller나 workflow에 별도의 "현재 단계"
추적(phase tracking) 상태를 추가해야 하는데, 이는 §2-2에서 명시한 비목표다
— 정확하지 않은 추측 메시지를 내보내는 것보다, §5-2의 범용 안내 메시지
("next safe epoch boundary", "if training has already finished...")로 이미
두 경우를 모두 포괄해 사용자에게 오해를 주지 않는 쪽을 선택한다.

---

## 8. `run_training()` 반환 이후 (artifact/export 중 SIGINT)

첫 번째 SIGINT가 다음 단계 중 하나에서 발생하면 controller의 flag는
설정되지만 **현재 진행 중인 workflow 단계를 중단시키지 않는다**(§1-3에서
확인한 대로 이 구간에서는 아무도 `should_stop()`을 다시 평가하지 않으므로):

- final checkpoint/artifact 저장(`training_history.json`, `class_mapping.json`,
  checkpoint, `best_model_state_dict.pt`)
- test evaluation(`test_result.json`)
- TorchScript export(`model.ts`, `model_metadata.json`)

현재 작업이 방해받지 않고 계속되며, 정상 완료되면 exit code 0이다. 이
구간에서도 두 번째 SIGINT는 즉시 `KeyboardInterrupt`를 발생시켜 강제
종료할 수 있다(§6의 보장 범위 그대로 적용 — 진행 중이던 저장/export가
중단될 수 있고, 원자적 저장이 아닌 산출물은 불완전할 수 있음).

---

## 9. Handler 설치와 복원

### 9-1. 설치 실패 시 정책(확정): 조용한 fallback 금지

`signal.signal(signal.SIGINT, ...)`가 메인 스레드 제약 등으로 `ValueError`를
발생시키면, **조용히 넘어가지 않는다.** 명확한 메시지의 `ValueError`로
다시 던져(`raise ... from exc`) 기존 CLI 오류 처리 경로(`except (...,
ValueError, OSError)`)를 그대로 타서 exit code 1로 처리되도록 한다:

```python
try:
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, controller.handle_signal)
except ValueError as exc:
    raise ValueError(
        "graceful SIGINT handling requires the CLI to run in the main thread"
    ) from exc
```

이는 이전 조사(Phase 4K 후보 비교 라운드)에서 제안했던 "조용한 폴백"
방향을 **뒤집는 확정 결정**이다 — 근거: graceful interrupt는 이번 Phase의
핵심 가치이므로, 설치가 실패했다면 사용자가 그 사실을 명확히 알아야
한다(자신도 모르게 Ctrl+C가 안 먹히는 것보다 즉시 오류로 실패하는 편이
운영상 더 안전하다).

### 9-2. 설치와 복원의 정확한 위치

handler 설치/복원 자체는 여전히 `run_imagefolder_training_workflow()`
호출만을 감싸는 **좁은 `try/finally`**로 수행한다(이 부분은 변경 없음):

```python
controller = _SigintStopController()

try:
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, controller.handle_signal)
except ValueError as exc:
    raise ValueError(
        "graceful SIGINT handling requires the CLI to run in the main thread"
    ) from exc

try:
    result = run_imagefolder_training_workflow(
        request,
        progress_callback=_print_progress,
        should_stop=controller.should_stop,
    )
finally:
    signal.signal(signal.SIGINT, previous_handler)
```

이 블록은 기존 `main()`의 `try: ... except (ModelValidationError,
TrainingConfigError, ValueError, OSError) as exc: return 1` 블록 **안에**
위치한다(`TrainingConfig`/`ImageFolderWorkflowRequest` 조립 코드 바로
다음, 기존 `result = run_imagefolder_training_workflow(...)` 호출을
대체하는 형태). 이렇게 하면:

- handler 설치 실패로 인한 `ValueError`는 기존 `except` 절이 그대로
  잡아 exit code 1로 처리된다(새 예외 처리 분기를 추가할 필요 없음).
- `run_imagefolder_training_workflow()`가 일반 예외(`ValueError`/`OSError`
  등)를 던져도 `finally`가 먼저 handler를 복원한 뒤 그 예외가 기존
  `except`로 전파되어 exit code 1로 처리된다.
- `KeyboardInterrupt`(두 번째 SIGINT의 escalation 또는 그 밖의 원인)가
  발생해도 `finally`가 먼저 handler를 복원한 뒤 예외가 계속 전파되며,
  기존 4개 예외 타입 어디에도 속하지 않으므로 그 `except` 절은 무시하고
  §9-3에서 정의하는 더 바깥의 `except KeyboardInterrupt`로 전달된다.

### 9-3. `KeyboardInterrupt`를 `main()` 실행 전체에서 130으로 통제한다

**Phase 4K 정책은 `main()` 실행 중 발생한 `KeyboardInterrupt`를(어느
코드 위치에서 발생했든) 프로젝트가 명시적으로 exit code 130으로 통제하는
것으로 확정한다.** §9-2의 좁은 `try/finally`는 handler를 정확히
설치/복원하기 위한 것일 뿐이고, `KeyboardInterrupt`를 잡아 exit code로
변환하는 책임은 이와 별개로 `main()` 함수 **전체**를 감싸야 한다 — 그래야
다음 네 가지 경우가 전부 130으로 수렴한다:

- `parse_args(argv)` 실행 중 `KeyboardInterrupt`
- `run_imagefolder_training_workflow()` 실행 중 두 번째 SIGINT(escalation)
- workflow가 정상 반환된 뒤 결과를 출력하는 코드(`history = result.history`
  이후, handler가 이미 복원된 상태) 실행 중 `KeyboardInterrupt`
- 그 밖에 `main()` 안 어디서든 발생하는 `KeyboardInterrupt`

**구조 선택: 기존 `main()` 본문 전체를 가장 바깥 `try/except
KeyboardInterrupt`로 감싼다**(비교한 두 후보 — A. `main()`이 별도 private
`_main()`을 감싸는 구조, B. 기존 `main()` 본문을 그대로 한 단계 더 감싸는
구조 — 중 **B로 확정**). 근거: `_main()`을 새로 도입하는 A는 함수를
분리해야 하는 리팩터링이지만, B는 기존 본문을 그대로 두고 가장 바깥에
`try:`/`except KeyboardInterrupt:` 한 쌍만 추가하는 최소한의 변경이다.
이 저장소의 다른 스크립트들(`run_training_e2e.py` 등)도 전부 `main()`
하나로 끝나는 단일 함수 구조를 유지해 왔으므로, 새 private 함수를
도입하는 것은 이번 기능에 필요한 범위를 넘는 리팩터링이다.

```python
def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        print("ImageFolder Training")
        ...

        try:
            training_config = TrainingConfig(...)
            request = ImageFolderWorkflowRequest(...)

            controller = _SigintStopController()
            try:
                previous_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, controller.handle_signal)
            except ValueError as exc:
                raise ValueError(
                    "graceful SIGINT handling requires the CLI to run in the main thread"
                ) from exc

            try:
                result = run_imagefolder_training_workflow(
                    request,
                    progress_callback=_print_progress,
                    should_stop=controller.should_stop,
                )
            finally:
                signal.signal(signal.SIGINT, previous_handler)
        except (ModelValidationError, TrainingConfigError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        # 기존 결과 출력 로직(수정 없음)
        history = result.history
        ...
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Exiting without completing remaining work.", file=sys.stderr)
        return 130
```

가장 바깥의 `except KeyboardInterrupt:`는 traceback을 출력하지 않고 짧은
안내만 남긴 뒤 정해진 exit code를 반환한다 — 사용자가 의도적으로 두 번
누른 강제 종료에 Python의 기본 traceback을 보여주는 것은 불필요한
소음이기 때문이다. **이 `print()`는 signal handler(`handle_signal()`)
내부가 아니라, 예외가 정상적으로 전파되어 일반 Python 실행 흐름의
`except` 블록이 실행되는 지점이므로 §5-1/§5-2에서 설명한 "signal handler
안에서는 `os.write()`만 허용" 제약과는 무관하다** — 이 시점에는 이미
`finally`가 signal handler를 원래 상태로 복원했고, 우리는 더 이상 signal
delivery 컨텍스트 안에 있지 않으며 통상적인 예외 처리 코드를 실행하고
있을 뿐이다.

`argparse`가 잘못된 인자에 대해 정상적으로 발생시키는 `SystemExit(2)`는
`KeyboardInterrupt`의 서브클래스가 아니고 `BaseException`의 다른 형제
분기이므로, 위 `except KeyboardInterrupt:`에 전혀 영향받지 않는다 —
`parse_args()`의 기존 오류 처리 동작(잘못된 인자 → `SystemExit(2)`)은
그대로 유지된다.

---

## 10. Exit Code 정책

| 상황 | exit code |
|---|---|
| 정상 완료 | 0 |
| 첫 번째 SIGINT cooperative stop 성공(artifact/final checkpoint 저장까지 완료) | 0 |
| 검증/저장/일반 workflow 오류(`ModelValidationError`/`TrainingConfigError`/`ValueError`/`OSError`, handler 설치 실패 포함) | 1 |
| 두 번째 SIGINT 또는 그 밖의 `KeyboardInterrupt` | 130 |

**첫 번째 SIGINT cooperative stop이 왜 0인가**: cooperative stop은 기존
`should_stop`이 이미 표현하는 정상 종료 경로이고(Phase 4I 설계 자체가 이를
"정상 완료"로 취급), `history.stopped_by_user=True`로 종료 사유가
artifact/checkpoint에 기록되며, final artifact/checkpoint 저장도 예외 없이
완료된 상태이기 때문이다 — 사용자가 요청한 대로 정확히 동작했으므로 오류가
아니다.

**130이 왜 별도인가**: 두 번째 SIGINT는 사용자가 "더 기다리지 않고 지금
당장 끝내라"고 명시적으로 요청한 것이며, 그 결과 artifact/checkpoint 저장이
불완전할 수 있다 — 이를 exit code 0/1과 구분해 스크립트/CI가 "완결된
정상 종료"와 "사용자 강제 중단"을 명확히 구별할 수 있게 한다. 130은
POSIX 관례(128+SIGINT)를 그대로 채택한 것으로, 이 저장소가 이미
Phase 4G/4H/4J 전체에서 exit code 0/1을 명시적으로 관리해 온 것과 같은
원칙(플랫폼 기본 동작에 맡기지 않고 프로젝트가 직접 통제)의 연장이다.

---

## 11. Checkpoint 및 atomic save와의 관계

### 11-1. `checkpoint_every`와의 관계(정확한 문서화)

- 첫 번째 SIGINT cooperative stop이 정상 완료되고 `checkpoint_out`이
  지정되어 있었다면, **`checkpoint_every` 설정과 무관하게** final
  checkpoint가 저장된다(Phase 4J §6-4의 "무조건 최종 저장" 계약 그대로).
- `checkpoint_every`는 **두 번째 Ctrl+C, 프로세스 crash, 그 밖의 비정상
  종료처럼 final save 자체에 도달하지 못하는 상황**에 대비해, 학습 도중
  가장 최근에 완료된 epoch까지의 상태를 미리 보존해 두는 기능이다.
- **`checkpoint_every`는 graceful stop의 필수 옵션이 아니다** — 켜져
  있지 않아도 첫 번째 SIGINT cooperative stop은 정상적으로 최종
  checkpoint를 남긴다. `checkpoint_every`가 값을 더하는 지점은 오직 "final
  save까지 도달하지 못한 강제 종료" 시나리오뿐이다.

### 11-2. atomic save가 이미 제공하는 보호(재확인)

§1-4/§6-3에서 확인한 대로, `_atomic_torch_save()`/`_atomic_write_text()`의
`except BaseException` 계약은 이번 Phase 이전부터(Phase 4J) 이미 존재하며,
signal handler의 존재 여부와 무관하게 항상 성립한다. 이번 Phase는 이 보호
위에 "언제 두 번째 SIGINT가 그 보호를 시험하게 되는가"라는 시나리오만
추가할 뿐, 보호 메커니즘 자체를 새로 만들거나 바꾸지 않는다.

---

## 12. CLI 배선

### 12-1. `_SigintStopController`(private, `scripts/train_imagefolder.py` 내부)

```python
class _SigintStopController:
    """SIGINT(Ctrl+C)를 run_imagefolder_training_workflow()의
    should_stop= 콜백으로 변환하는 CLI 전용 private controller.
    signal.signal()의 handler로도, should_stop=으로도 동시에 바인딩된다.
    handle_signal()은 bool 대입과 고정 bytes의 저수준 stderr 출력만
    수행하고, checkpoint/artifact 파일 I/O, PyTorch/model/optimizer/
    generator 접근, logging/동적 formatting을 하지 않는다(RNG/state-purity
    계약, Phase 4I §3-5/Phase 4J §3-5와 동일; §5-1/§5-2의 stream 재진입
    회피 근거 포함)."""

    def __init__(self) -> None:
        self._interrupt_requested = False

    def should_stop(self) -> bool:
        return self._interrupt_requested

    def handle_signal(self, signum, frame) -> None:
        if not self._interrupt_requested:
            self._interrupt_requested = True
            try:
                os.write(2, _INTERRUPT_MESSAGE_BYTES)
            except OSError:
                pass
            return
        signal.default_int_handler(signum, frame)
```

공개 API를 추가하지 않는다 — `image_ai_studio` 패키지의 어떤 `__init__.py`/
`__all__`에도 노출하지 않고, `scripts/train_imagefolder.py` 모듈 안의 밑줄
접두 클래스로만 존재한다. `threading.Event`, workflow 계층의 콜백 결합
helper, `run_training()`/`run_imagefolder_training_workflow()`의 시그니처
변경은 전부 하지 않는다.

### 12-2. `main()` 배선(개념 구조, §9-3의 최종 구조와 동일)

§9-3에서 확정한 대로, `main()` 본문 전체를 가장 바깥 `try/except
KeyboardInterrupt`로 감싸고(구조 B), 그 안에 기존 `try/except
(ModelValidationError, ...)` 블록과 handler 설치/복원용 좁은
`try/finally`를 중첩한다. 전체 코드는 §9-3의 코드 블록을 그대로 참고한다
— 여기서는 중복 기재하지 않는다.

### 12-3. `should_stop` 외부 결합(확정: CLI 전용, workflow API 무변경)

`run_imagefolder_training_workflow()`는 이미 `should_stop:
ShouldStopCallback | None`을 받는다. `train_imagefolder.py`는 현재 외부에서
넘어오는 `should_stop`이 전혀 없으므로(§1-5의 기존 `main()` 코드에
`should_stop=` 호출 자체가 없었음) **결합할 대상 자체가 없다** —
`controller.should_stop`을 그대로 `should_stop=`에 넘기면 끝난다. workflow
계층에 별도 결합(helper) 코드를 추가하지 않는다(불필요한 core API 확장
금지 원칙).

---

## 13. 테스트 전략

### 13-1. Controller 단위 테스트(`tests/scripts/test_train_imagefolder_cli.py`, 실제 OS signal 없음)

- 초기 `controller.should_stop()`은 `False`
- `controller.handle_signal(signal.SIGINT, None)` 1차 호출 후
  `should_stop()`은 `True`
- 1차 호출 시 안내 메시지가 **정상적인 `capfd` 테스트 환경**에서 stderr에
  정확히 한 번 관찰됨(§5-2가 명시하는 "정확히 한 번 시도"라는 production
  계약과, 정상 환경에서 그 시도가 실제로 전체 문구를 남긴다는 관찰을
  구분한다 — production 계약이 partial write까지 보장하는 것은 아니다).
  `os.write(2, ...)`는 Python의 `sys.stderr` 텍스트 스트림을 거치지 않고
  파일 디스크립터에 직접 쓰므로, pytest의 `capsys`(Python 레벨
  `sys.stdout`/`sys.stderr` 치환 기반)는 이 출력을 잡지 못한다. **fd
  레벨 출력까지 캡처하는 `capfd`를 사용한다**(§13-2에서도 동일)
- 2차 호출은 `KeyboardInterrupt`를 발생시킴(`pytest.raises(KeyboardInterrupt)`)
- 1차 호출 후 다시 여러 번 `should_stop()`을 읽어도 값이 안정적으로 `True`
  유지(멱등성)
- `handle_signal()` 호출 전후로 `torch.get_rng_state()`가 동일함을 확인
  (RNG 미소비 회귀 테스트, Phase 4J의 RNG-purity 검증 패턴 재사용)
- `_SigintStopController`가 `model`/`optimizer`/`generator` 등 학습 객체에
  대한 참조를 전혀 갖지 않는 구조임을 코드 구조 자체로 보장(생성자가 인자를
  받지 않음) — 별도 실행 테스트보다 타입/시그니처로 이미 증명됨

### 13-2. CLI 배선 테스트(`cli.main()` 직접 호출, `run_imagefolder_training_workflow`를 monkeypatch)

**handler 캡처 방법**: fake workflow는 `should_stop`(controller의
`should_stop` 메서드)만 인자로 받으므로, fake workflow 내부에서는
`controller.handle_signal`(signal.signal()의 실제 handler 인자)에 직접
접근할 수 없다. 대신 `cli.signal.getsignal`과 `cli.signal.signal`을 **둘
다 완전한 fake로 대체**해 설치되는 handler를 캡처한다 — 실제
`signal.signal()`을 호출하는 방식(예: fake 안에서 진짜 `signal.signal()`을
그대로 다시 호출해 위임하는 형태)은 **절대 쓰지 않는다.** 그런 방식은
pytest 프로세스 자체의 실제 SIGINT handler를 바꿔버려서, 테스트 도중
실제 Ctrl+C 동작에 영향을 주거나 이후 다른 테스트에 상태가 새어나갈 수
있다. 아래 fake는 `signal` 모듈을 전혀 건드리지 않고, "현재 설치된
handler가 무엇인지"를 테스트 프로세스 내부의 순수 Python 상태로만
모사한다:

```python
def test_first_sigint_makes_should_stop_return_true(monkeypatch):
    previous_handler = object()  # "설치 이전 handler"를 흉내내는 sentinel
    current_handler = {"value": previous_handler}
    signal_calls: list[tuple[object, object]] = []

    def fake_getsignal(sig):
        assert sig == cli.signal.SIGINT
        return current_handler["value"]

    def fake_signal(sig, handler):
        assert sig == cli.signal.SIGINT
        signal_calls.append((sig, handler))
        previous = current_handler["value"]
        current_handler["value"] = handler
        return previous

    monkeypatch.setattr(cli.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(cli.signal, "signal", fake_signal)

    captured = {}

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        captured["should_stop"] = should_stop
        # 설치 호출(첫 번째 signal_calls 항목)의 handler를 직접 호출해
        # 실제 SIGINT 전달을 흉내낸다 -- 이 시점의 handler는
        # controller.handle_signal이어야 한다. 이 호출은 실제 OS/pytest
        # SIGINT handler와 무관한, fake_workflow 안에서의 순수한 함수
        # 호출일 뿐이다.
        installed_handler = signal_calls[0][1]
        assert should_stop() is False
        installed_handler(cli.signal.SIGINT, None)
        assert should_stop() is True
        return <최소 유효 ImageFolderWorkflowResult>

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)
    exit_code = cli.main([...])
    assert exit_code == 0
    assert current_handler["value"] is previous_handler  # 정상 종료 후 복원 확인
```

검증 목록:

- 설치 호출(`signal_calls[0]`)의 handler가 callable이고, fake workflow
  안에서 이 handler를 직접 호출하면 이후 workflow에 전달된 `should_stop()`
  이 `True`를 반환하는지 확인
- 정상 완료 시 exit code 0
- 정상 완료 후 fake `current_handler["value"]`가 `main()` 호출 **이전**의
  `previous_handler` sentinel과 동일한지 확인 — 정상/일반 예외/
  `KeyboardInterrupt` 세 경로 **전부**에서 이 마지막 복원 호출이
  `previous_handler`를 쓰는지 각각 확인. `cli.signal.getsignal`/
  `cli.signal.signal`은 항상 함께 fake로 대체해 복원 대상이 결정론적이고,
  테스트가 실제 pytest 프로세스의 SIGINT handler를 전혀 건드리지 않도록
  한다(위 코드 스케치가 유일한 기준 패턴).
- fake workflow가 일반 예외(`ValueError` 등)를 던지도록 만든 뒤에도
  마지막 복원 호출이 `previous_handler`인지 확인
- fake workflow가 `KeyboardInterrupt`를 던지도록 만든 뒤 exit code가 130이고
  마지막 복원 호출이 `previous_handler`인지 확인
- `signal.signal()` 자체를 monkeypatch로 `ValueError`를 던지게 만들어
  handler 설치 실패 시 exit code 1 + stderr에 "main thread" 관련 명확한
  메시지가 포함되는지 확인(이 경우는 별도 fake로 구성 — 설치 자체가
  실패하므로 위의 "설치 호출 캡처" 패턴과는 다른 시나리오)
- `checkpoint_out=None` 조합에서도 `should_stop` 배선이 정상 동작
- 기존 CLI 테스트(§1-6에서 확인한 반복 호출 구조) 전부 무수정 통과

**§9-3의 "main() 실행 전체에서 130" 계약을 커버하는 추가 시나리오**(구조
B 변경에 따른 신규 검증 대상):

- `parse_args`를 monkeypatch로 `KeyboardInterrupt`를 던지도록 만들어,
  workflow 호출 이전 단계에서 발생한 인터럽트도 exit code 130으로
  처리되는지 확인(이 경로에서는 handler가 아직 설치되지 않았으므로
  handler 복원 검증은 해당 없음)
- fake workflow가 정상적으로 결과를 반환한 뒤, `main()`의 결과 출력
  코드(`history = result.history` 이후)가 실행되는 지점에서
  `KeyboardInterrupt`가 발생해도 exit code 130으로 처리되는지 확인(예:
  fake `ImageFolderWorkflowResult`의 `history` 속성을 접근할 때
  `KeyboardInterrupt`를 던지는 객체로 구성하거나, `print`를 monkeypatch해
  특정 호출에서 `KeyboardInterrupt`를 발생시킴) — 이 경로는 workflow가
  이미 정상 반환된 뒤이므로 signal handler는 이미 `previous_handler`로
  복원되어 있어야 함을 함께 확인
- `argparse`가 잘못된 인자에 대해 던지는 기존 `SystemExit(2)` 동작이
  이번 변경으로 영향받지 않았는지(기존 CLI 테스트가 이미 다루고 있다면
  회귀 재확인만)

### 13-3. workflow/E2E 레벨 — 중복 테스트를 만들지 않는다

`tests/training/test_imagefolder_workflow.py`가 이미 다음을 결정론적
callback(실제 signal 아님)으로 검증하고 있음을 확인했다(§1 조사, 재확인
불필요 — 이번 Phase가 workflow의 `should_stop` 처리 로직 자체를 바꾸지
않으므로):

- `test_workflow_forwards_should_stop_and_stops_training_early`: 중간
  epoch에서 `should_stop`이 `True`를 반환하면 `stopped_by_user=True`로
  종료
- `test_workflow_user_stopped_run_produces_full_artifact_set`: 사용자
  중단 후에도 전체 아티팩트 파이프라인이 완주되고, `checkpoint_out`이
  있으면 최종 checkpoint에 `stopped_by_user=True`가 반영됨
- `test_workflow_user_stopped_checkpoint_is_resumable`: 사용자 중단으로
  저장된 checkpoint가 정상적으로 resume 가능함

Phase 4J의 기존 `test_loop.py`가 이미 "scheduled mid-loop checkpoint는
`stopped_by_user=False`, final checkpoint만 정확한 값을 반영한다"는 계약도
커버하고 있다. **이번 Phase는 CLI가 이 기존 `should_stop` 계약에 다른
구현체(controller)를 공급할 뿐이므로, workflow 레벨에 새 테스트를 추가하지
않는다.**

### 13-4. 실제 subprocess SIGINT 테스트: 선택 사항(이번 Phase 필수 범위 아님)

§13-1/§13-2의 조합으로 실제 로직(정확히 무엇이 언제 일어나는가)은 전부
커버되므로, 실제 OS signal을 자식 프로세스에 보내는 자동화 테스트는 이번
Phase의 **필수 범위에서 제외**한다. 근거:

- Windows에서 자식 프로세스에 Ctrl+C를 보내려면
  `CREATE_NEW_PROCESS_GROUP` + `GenerateConsoleCtrlEvent`가 필요해 POSIX의
  `os.kill(pid, signal.SIGINT)`보다 훨씬 까다롭고 타이밍 의존적이라 flaky
  위험이 큼.
- 이 저장소의 기존 4개 E2E 스크립트는 전부 in-process 실행이라 subprocess
  기반 테스트를 새로 도입하는 비용(timeout 설계, 프로세스 정리 보장,
  플랫폼 분기)이 이번 기능의 로직 검증 가치 대비 크다.

도입 필요성이 나중에 확인되면 별도로: (a) 명시적 timeout, (b) Windows/POSIX
분기(`CTRL_C_EVENT` vs `SIGINT`), (c) 테스트 실패/타임아웃 시 자식 프로세스
강제 종료 보장을 갖춰 추가한다.

---

## 14. 수동 Acceptance Test

자동 테스트로 대체할 수 없는 실제 터미널 조작 확인 절차. 최소 다음을
수행한다:

1. 충분히 긴 ImageFolder 학습을 `--checkpoint-out`과 함께 실행한다(예:
   `--epochs 20` 이상, epoch 하나가 눈으로 인지 가능한 시간이 걸리는
   데이터셋/배치 크기).
2. 중간 epoch 진행 중 Ctrl+C를 **1회** 누른다.
3. §5-2의 안내 메시지가 stderr에 출력되는지 확인한다.
4. 현재 진행 중이던 epoch가 끝까지 완료된 뒤 학습이 멈추는지 확인한다.
5. 프로세스 종료 코드가 `0`인지 확인한다(쉘에서 `echo $?` 또는
   `echo %ERRORLEVEL%`).
6. `training_history.json`(또는 CLI 출력의 `stopped_by_user=` 줄)에서
   `stopped_by_user=True`인지 확인한다.
7. `--checkpoint-out`을 지정했다면, 최종 checkpoint를 로드해
   `history["stopped_by_user"] is True`인지 확인한다.
8. 다시 같은 명령을 실행하고, 이번에는 Ctrl+C를 **연속 2회** 눌러 프로세스가
   즉시 종료되는지, 종료 코드가 `130`인지 확인한다.
9. 2회 Ctrl+C 시점 이전에 마지막으로 원자적 저장이 완료된 checkpoint
   파일이 손상되지 않았는지(`torch.load()`로 다시 로드 가능한지) 확인한다.
10. `--checkpoint-out` **없이** 학습을 실행하고, Ctrl+C를 1회 눌러 checkpoint
    없이 나머지 artifact(`training_history.json`, `class_mapping.json`,
    `best_model_state_dict.pt`, `test_result.json`)가 정상 생성되는지
    확인한다.

### 14-1. 실행 결과

**환경**: Windows Anaconda Prompt(cmd), CIFAR-10 ImageFolder fixture +
`examples/models/phase4c_cifar10_model.json`으로 실제
`scripts/train_imagefolder.py`를 직접 실행.

**단일 Ctrl+C graceful stop(위 절차 1~7): PASS.**

- 학습 도중 Ctrl+C를 1회 입력하자, §5-2의 안내 메시지(현재 epoch의 안전한
  경계에서 학습을 중단한다는 안내 + 한 번 더 누르면 즉시 종료된다는 안내)가
  정상적으로 출력됨을 확인했다(절차 3).
- Ctrl+C 입력 시점에 진행 중이던 epoch는 끝까지 정상 완료됐고, 다음
  epoch는 시작되지 않았다(절차 4).
- 프로세스 종료 코드는 `echo %ERRORLEVEL%` 기준 `0`이었다(절차 5).
- 최종 출력과 저장된 history에서 `stopped_by_user=True`를 확인했다(절차 6).
- best model, training history, class mapping, test result, checkpoint,
  checkpoint metadata가 전부 정상 저장됐다(절차 7).
- traceback 없이 깔끔하게 종료됐다.

**두 번째 Ctrl+C 강제 종료(위 절차 8~9): 실제 터미널에서 안정적인 수동
재현 불가 — 자동화 테스트로 검증됨.**

`--checkpoint-every 1`, `--epochs 100`, `--batch-size 1` 등 여러 조합으로
반복 시도했으나, 현재 CIFAR-10 fixture는 규모가 작아 "첫 번째 Ctrl+C 이후
현재 epoch가 끝나고 graceful 종료(artifact 저장까지)가 완료되기까지"의
시간 창(§9-3의 handler가 활성 상태로 남아 있는 구간)이 매우 짧다. 이
때문에 사람이 두 번째 Ctrl+C를 그 창 안에 안정적으로 밀어 넣기가 어려웠다.
실제로 Ctrl+C를 연타했을 때는 graceful 종료와 artifact 저장이 전부 끝난
**뒤**, Python 인터프리터의 종료 정리(atexit 등) 과정에서 추가 Ctrl+C가
들어가 `KeyboardInterrupt`가 발생하는 사례가 나타났다 — 이 경우 Windows가
보고한 종료 코드는 `-1073741510`(`0xC000013A`,
`STATUS_CONTROL_C_EXIT` — Ctrl+C에 의한 Windows 프로세스 강제 종료)였다.
이는 애플리케이션이 이미 정상 종료된 이후 인터프리터 레벨에서 발생한
별개의 인터럽트이지, `main()`이 `_SigintStopController`를 통해 의도적으로
가로채는 §9-3의 `130` 경로를 실제로 통과한 결과가 아니다 — 따라서 이
관측을 Phase 4K의 130 경로 검증으로 간주하지 않는다.

이 경로 자체(두 번째 SIGINT → `signal.default_int_handler` 위임 →
`KeyboardInterrupt`가 최상위 CLI 경계에서 처리되어 `130` 반환, handler
복원 포함)는 §13-2의 자동화 테스트
(`test_cli_handler_restored_after_workflow_keyboard_interrupt_and_exit_code_130`
등)에서 이미 결정론적으로(실제 OS signal 없이 controller/handler를 직접
호출) 검증되어 있다. 실제 터미널에서 재현하기 어려웠던 것은 이 작은
fixture의 timing window가 짧기 때문이며, **구현 결함이 확인된 것이
아니다** — Phase 4K acceptance를 막는 blocker로 보지 않는다.

**종합 판단**: 단일 Ctrl+C graceful stop의 실제 Windows 터미널 수동
검증(PASS) + `stopped_by_user=True`(PASS) + artifact/checkpoint 저장
(PASS) + exit code 0(PASS) + 두 번째 Ctrl+C 강제 종료 로직의 자동화
테스트 검증(PASS, 단 실제 터미널에서의 안정적 수동 재현은 timing window
제약으로 이번 라운드에서 이뤄지지 못함)을 종합해 **Phase 4K 수동
acceptance는 PASS로 판정한다.**

절차 8~9(두 번째 Ctrl+C 실측)와 절차 10(`--checkpoint-out` 없이 1회
Ctrl+C)은 이번 수동 라운드에서 명시적으로 재현되지 않았다 — 절차 10은
`test_cli_checkpoint_out_none_should_stop_wiring_still_works`(§13-2)의
자동화 테스트가 동일 계약을 이미 커버한다.

---

## 15. 하위 호환

완전 하위 호환이다: `_SigintStopController`는 `train_imagefolder.py` 내부
private 구현이라 공개 API 변화가 없고, `run_imagefolder_training_workflow()`/
`run_training()` 시그니처도 무변경이다. Ctrl+C를 누르지 않는 기존 실행
경로는 signal handler가 설치되어 있다는 사실 자체가 어떤 관찰 가능한
차이도 만들지 않으므로 출력/exit code가 완전히 동일하다. `checkpoint_every`,
`--resume-from`/`--checkpoint-out` 등 기존 옵션과의 상호작용도 전부
기존 계약 그대로다(§11).

---

## 16. 파일별 변경 계획

### 16-1. 수정한 파일(구현 완료)

- **`scripts/train_imagefolder.py`**: `import os`/`import signal` 추가,
  `_INTERRUPT_MESSAGE_BYTES` 상수 추가, `_SigintStopController` 클래스 추가,
  `main()`에 handler 설치/복원 + `should_stop=` 배선 + 본문 전체를 감싸는
  `except KeyboardInterrupt: ... return 130` 추가(§9-3/§12-2). 모듈
  docstring에도 Ctrl+C 동작 요약 한 문단 추가.
- **`tests/scripts/test_train_imagefolder_cli.py`**: `_make_fake_result()`/
  `_install_fake_signal_module()` private helper 2개 + controller 단위
  테스트 9개(`test_sigint_controller_*`) + CLI 배선/handler 복원/설치
  실패 테스트 6개(`test_cli_first_sigint_*`, `test_cli_handler_restored_*`
  x3, `test_cli_handler_install_failure_*`,
  `test_cli_checkpoint_out_none_should_stop_wiring_*`) + `main()` 전체
  KeyboardInterrupt 커버리지 테스트 2개(`test_cli_keyboard_interrupt_*`
  x2) 추가 — 총 17개 신규 테스트, 기존 15개는 무수정.
- **`README.md`**: "Phase 4K: Graceful SIGINT and Cooperative Training
  Stop" 절 신설(Ctrl+C 동작, exit code 표, 안내 메시지 예시,
  `checkpoint_every`와의 관계 명시), "현재 지원 범위"/"아직 미지원" 목록과
  Phase 4I 절의 낡은 문구("CLI에서 실제로 중단을 트리거하는 방법은 이번
  Phase 범위 밖") 갱신.
- **`docs/phase4k_graceful_interruption_design.md`**(본 문서): 구현 결과에
  맞게 정합성 갱신(§14 실행 여부, 본 §16, 상태 줄).

### 16-2. 무수정 파일(확인 완료)

- `src/image_ai_studio/training/loop.py`
- `src/image_ai_studio/training/history.py`
- `src/image_ai_studio/training/checkpoint.py`
- `src/image_ai_studio/training/imagefolder_workflow.py`
- `src/image_ai_studio/training/imagefolder_resume.py`

§1에서 확인한 대로 이 다섯 파일이 이미 이번 기능에 필요한 모든 API
(`should_stop`, checkpoint atomic save, "should_stop() 다음 epoch 없으면
평가 안 함" 규칙)를 제공했으므로, **실제 구현에서도 이 다섯 파일을 전부
무수정으로 완료했다** — blocker 없음.

---

## 17. 구현 순서(작은 단계)

1. `scripts/train_imagefolder.py`에 `import signal` + `_INTERRUPT_MESSAGE_BYTES` +
   `_SigintStopController` 추가(아직 `main()`에서 사용하지 않음, 순수 추가).
2. `_SigintStopController` 단위 테스트 작성/통과(§13-1).
3. `main()`에 handler 설치/복원 + `should_stop=` 배선 추가(§12-2).
4. `main()`에 `except KeyboardInterrupt: ... return 130` 추가.
5. CLI 배선 테스트 작성/통과(§13-2), 기존 CLI 테스트 전부 무수정 통과 재확인.
6. 전체 pytest(420개 이상) + 4개 E2E + ImageFolder E2E 연속 2회 + C++
   CPU/CUDA parity 재확인(모두 SIGINT를 쓰지 않는 경로이므로 수치 anchor
   불변 확인이 목적).
7. §14의 수동 acceptance test 실제 수행.
8. `README.md` 갱신.

---

## 18. 위험 요소

- **§1-7의 handler 실행 지연**: 긴 C/CUDA 호출 도중에는 안내 메시지가 즉시
  뜨지 않을 수 있다 — 사용자가 "Ctrl+C가 안 먹혔다"고 오인해 반복해서 누를
  가능성이 있고, 그 반복 입력이 결국 2차 SIGINT로 해석되어 강제 종료로
  이어질 수 있다. 이는 문서(README)에서 명시적으로 안내해야 하는 한계다.
- **§9-1의 "조용한 폴백 금지" 결정**으로 인해, 메인 스레드가 아닌 곳에서
  이 CLI가 임베딩되어 실행되는 경우 학습 자체가 `ValueError`로 실패한다
  (이전 라운드의 "조용히 기능만 끄기" 방향에서 의도적으로 전환한 trade-off).
  이 CLI가 현재 항상 스크립트 진입점으로만 쓰이므로 위험은 낮지만, 향후
  누군가 `main()`을 라이브러리 함수처럼 다른 스레드에서 호출하면 이
  변경이 회귀로 느껴질 수 있다.
- **2차 SIGINT 시 비원자적 산출물(JSON/TorchScript) 손상 가능성**은
  명시적으로 받아들인 위험이다(§6-3, §2-2 비목표).
- **Windows 콘솔 특성**: `signal.CTRL_C_EVENT` 관련 세부 동작, 터미널 종류
  (cmd.exe/PowerShell/Windows Terminal)별 Ctrl+C 전달 방식의 미세한 차이는
  이 저장소 코드로 검증할 수 없는 플랫폼 영역이다. §14-1에서 Anaconda
  Prompt(cmd) 기준 단일 Ctrl+C 경로는 실제로 확인했다. 다만 작은 fixture
  에서는 graceful 종료가 매우 빠르게 끝나(§1-7의 handler 실행 지연과는
  반대 방향의 문제 — 오히려 "너무 빨리 끝나서" 두 번째 Ctrl+C를 그 활성
  구간 안에 사람이 안정적으로 넣기 어려움), 두 번째 Ctrl+C의 실제 터미널
  재현이 불안정했다(§14-1) — 다른 터미널(PowerShell/Windows Terminal)이나
  더 긴 학습에서 재검증이 필요할 수 있다.
- **stderr 메시지가 표준 출력 버퍼링과 섞이는 상황**: `_print_progress()`가
  `print()`로 stdout에 epoch 진행을 계속 찍는 도중 stderr 메시지가 끼어들면
  터미널에서 줄 순서가 뒤섞여 보일 수 있다(파이프/리다이렉션 환경에서
  stdout/stderr 버퍼링 정책에 따라 달라짐) — 기능상 문제는 아니지만 사용자
  경험에 영향을 줄 수 있는 사소한 위험으로 기록.

---

## 19. 회귀 불변조건

이번 Phase 이후에도 다음이 그대로 유지되어야 한다(구현 완료 후 검증 대상):

- 전체 pytest 420개 이상 통과
- 4개 E2E(`run_training_e2e.py`/`run_real_training_e2e.py`/
  `run_resume_training_e2e.py`/`run_imagefolder_training_e2e.py`) anchor
  수치 불변
- ImageFolder E2E 연속 2회 성공
- C++ CPU/CUDA parity PASS
- `checkpoint_hook`/`progress_callback`/`should_stop` 호출 순서 불변
  (`loop.py` 무수정이므로 자동 보장)
- checkpoint 포맷 버전 1 유지
- 자동 checkpoint(`checkpoint_every`) exact-resume 유지
- SIGINT를 전혀 사용하지 않는 기존 CLI 실행의 출력/exit code 완전 불변

---

## 20. 향후 확장

- **SIGTERM/SIGHUP**: 이번 Phase는 SIGINT만 다룬다. 컨테이너 오케스트레이션
  환경(`docker stop` 등)에서의 graceful shutdown이 필요해지면 별도 Phase에서
  검토 — `_SigintStopController`와 유사한 구조를 재사용할 수 있을 것으로
  보이나, SIGTERM은 기본적으로 예외를 발생시키지 않으므로 handler 설계가
  달라질 수 있다.
- **GUI stop button**: Phase 4I §19가 이미 예고한 `threading.Event` 기반
  연동이 이번 controller와는 별개의 소비자로 남는다 — GUI가 생기면
  `should_stop=stop_event.is_set`을 직접 넘기는 기존 경로를 그대로 쓰면
  되고, 이번 CLI 전용 controller와 결합할 필요가 없다.
- **batch 중간 checkpoint/resume**: 여전히 명시적 비목표(Phase 4F 이래
  일관된 스코프 결정).
- **Windows 콘솔 종료 이벤트(`WM_CLOSE`) 대응**: `signal` 모듈 범위 밖이라
  `win32api`류 별도 의존성이 필요 — 필요성이 확인되기 전까지 보류.
