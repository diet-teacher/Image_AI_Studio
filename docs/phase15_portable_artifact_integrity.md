# Phase 15: Portable Artifact Integrity Manifest

## 목적과 비목표

Phase 15는 ImageFolder 학습이 만든 휴대 가능한 세 산출물의 우연한 손상,
누락, stale 파일 및 서로 다른 실행 사이의 혼합을 추론 전에 탐지한다. 기존
산출물 형식과 public training/inference API는 바꾸지 않는다. 이 기능은
archive, 서명, 원격 provenance, 자동 migration, packaging 또는 CUDA 동작을
추가하지 않는다.

## Canonical bundle과 스키마

Canonical bundle은 한 디렉터리 바로 아래의 다음 파일로 구성된다.

1. `model_definition.json`
2. `best_model_state_dict.pt`
3. `class_mapping.json`
4. `artifact_manifest.json`

매니페스트의 `schema_version`은 `1`이다. `artifacts` 배열에는 앞의 세
산출물만 위 순서로 들어가며 각 entry는 `filename`, `size_bytes`, 소문자
64자리 `sha256`을 갖는다. 알 수 없는 필드, 중복 key/entry, 경로 구분자나
상위 디렉터리 참조, 잘못된 버전·크기·digest는 거부된다. JSON은 고정 key
정렬과 구분자, UTF-8, 마지막 newline을 사용해 canonical하게 직렬화된다.
해시는 파일 전체를 메모리에 올리지 않고 고정 크기 chunk로 계산한다.

## 학습 게시 순서

학습 workflow는 새 publication 시작 전에 기존 매니페스트를 무효화한다.
그 뒤 세 산출물을 기존 atomic writer로 각각 게시하고, 디스크에 존재하는
최종 세 파일을 다시 읽어 만든 매니페스트를 마지막에 원자적으로 게시한다.
따라서 이전 매니페스트가 새 partial bundle을 현재 번들로 보이게 하지 않는다.

이 보장은 파일별 atomicity다. 세 산출물 전체가 하나의 multi-file
transaction으로 동시에 교체되거나 실패 시 모두 rollback되는 것은 아니다.
중간 실패에서는 매니페스트가 없으므로 새 형식의 완전한 bundle로 주장되지
않지만, 호출자는 실패를 처리하고 필요하면 출력 디렉터리를 다시 생성해야
한다.

## 추론 검증 순서

세 요청 경로의 basename이 canonical 이름과 일치하고 그 부모들이 같은
물리 디렉터리를 가리키면 optional manifest 경계를 적용한다. 부모 identity는
strict resolution과 플랫폼 case normalization을 사용하므로 상대경로,
`.`/`..`, directory symlink 및 Windows junction alias가 manifest 검증을
우회하지 못한다. 부모 identity를 확정할 수 없으면 bounded integrity
오류로 fail closed한다.

매니페스트가 존재하면 JSON 파싱, schema 검증, 각 canonical 파일의 일반
파일·비-symlink 검사, 크기 및 SHA-256 검증이 모델 JSON, class mapping,
state dict 및 입력 이미지 역직렬화보다 먼저 수행된다. 누락, 손상 또는
mixed-run 파일은 `ManifestError`로 거부되며 정상 prediction은 생성되지
않는다. Directory alias 지원은 개별 artifact symlink 허용을 의미하지
않는다. 원래 artifact 경로는 보존되며 개별 symlink는 계속 거부된다.

## 호환성 경계

`artifact_manifest.json`이 아예 없는 기존 canonical 3-file bundle은
legacy 호환 경로로 계속 로드된다. 명시적인 noncanonical filename 조합도
이전 public 동작을 유지한다. 반대로 매니페스트가 존재하지만 깨졌거나
지원되지 않는 경우는 legacy로 간주하지 않고 실패한다.

## GUI 오류와 recovery

Core의 `ManifestError`는 기존 `InferenceController`와
`QtInferenceWorker.failed` 경로를 그대로 통과한다. `InferencePage`는 raw
traceback을 표시하지 않고 첫 줄만 고정 길이로 제한해 `Failed:` 상태로
보여 준다. 실패 시 stale prediction을 지우고 기존 worker/thread cleanup과
control 복원을 수행한다. 같은 페이지에서 유효한 bundle로 다시 실행하면
오류가 지워지고 정상 결과가 표시된다. Folder progress, cooperative
cancellation, partial export, preview 및 close coordination 계약은 바뀌지
않는다.

## 보안 및 자원 한계

SHA-256은 우연한 corruption과 mixed/stale member 탐지를 위한 checksum일
뿐 signature, authenticity, provenance 또는 confidentiality를 제공하지
않는다. 공격자가 산출물과 매니페스트를 함께 일관되게 교체하면 탐지할 수
없다. 검증이 성공한 뒤 역직렬화 전에 파일이 바뀌는 TOCTOU 가능성도 남는다.
해싱은 streaming이지만 현재 매니페스트 JSON 자체는 전체를 메모리에 읽어
파싱한다. 오류 문자열은 제한되지만 입력 파일 크기 자체의 별도 상한이나
원격 신뢰 체계는 제공하지 않는다.

Windows에서 directory symlink를 실제로 만드는 테스트는 개발자 모드나
권한에 따라 skip될 수 있다. Junction/case/alias identity의 핵심 분기는
권한에 의존하지 않는 회귀 테스트로 별도 검증한다.

## CPU graduation 범위

로컬의 작은 ImageFolder dataset과 1 epoch CPU 학습으로 실제 canonical
bundle 및 매니페스트를 만들고, 매니페스트의 schema·entry·크기·digest를
테스트 측 독립 SHA-256 계산으로 확인한다. 같은 실제 GUI 요청 경로에서
정상 추론, artifact corruption, 다른 학습 실행 파일 혼합, missing artifact,
매니페스트 없는 legacy bundle과 실패 후 재실행을 검증한다. 네트워크,
다운로드, CUDA, 원격 provenance, packaging 및 배포 검증은 이 graduation의
범위 밖이다.
