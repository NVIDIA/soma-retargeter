# SOMA → H2 retarget 기술 인수인계

문서 개정: 2026-11-04. 최신 경로는 [BVH→CSV→SQP/QP→PKL](01_retarget_pipeline.md)이며 실제 H100 코드를 `h2_pipeline/production/`에 보존했다.

SOMA BVH를 H2의 관절·root reference로 변환하는 절차와 NVIDIA 원본 대비 변경 근거를 정리한다. 기하학적 retarget 검증 후 WBC 저장소에서 동역학적 tracking과 정책 학습을 수행한다.

| 문서 기준 | 값 |
|---|---|
| 문서 개정일 | 2026-11-04 |
| 적용 branch | `h2-retarget-support` |
| 검토 코드 | `h2_pipeline/production/source_manifest.json`의 출처/해시 |
| 실행 환경 | 각 실행 가이드의 사전 조건 참조 |

## 1. 문서 구성

최신 재현에는 **01 Retarget pipeline**을 순서대로 수행한다. 00은 기존 기본 변환기의 사용 방법이고, 설계 배경과 과거 실험은03에 보존한다.

| 순서 | 문서 | 내용 |
|---|---|---|
| 00 | [실행 가이드](00_execution_runbook.md) | 설치·asset 경로·설정 생성·변환·검사 명령 |
| 01 | [Retarget pipeline](01_retarget_pipeline.md) | 입력·출력·로봇 설정 및 처리 방식 |
| 02 | [검증 및 학습 인계](02_validation_handoff.md) | CSV 계약·품질 기준·산출물 |
| 03 | [원본 대비 변경 및 튜닝 이력](03_upstream_changes_and_tuning.md) | 초기 문제·설정 전후 수치·확인된 결과 |

## 2. 현재 적용 경로

`python -m h2_pipeline`은 봉인된 중립 보정·G1 비교·SQP/QP·발 검증·PKL 인계를 제공한다. 최상위 기본 변환기의 H2 MJCF는 `H2_MJCF_PATH` 환경변수로 지정할 수 있다. 두 경로의 설정을 섞지 않는다.

## 3. 관련 저장소 및 기존 자료

[WBC H2 실행 가이드](https://github.com/MFIWO/GR00T-WholeBodyControl/blob/h2-transfer-dev/docs/h2/RUNBOOK.md)와 [현재 학습 recipe](https://github.com/MFIWO/GR00T-WholeBodyControl/tree/h2-transfer-dev/recipes/h2_low_work)에서 정책 학습·평가를 진행한다. WBC 저장소 접근 권한이 필요할 수 있다.

## 4. 근거 및 검증 범위

| 구분 | 해석 |
|---|---|
| 구현·설정 확인 | 명시된 commit의 코드와 설정 파일을 대조한 내용 |
| 개발 시험 결과 | 저장소 보고서 또는 개발 담당자 제공 평가 요약. 원본 로그 확인 여부는 해당 절에 명시 |
| 실행 명령 | 실제 진입점·옵션과 shell/Python 구문을 대조한 예시. 경로·데이터·환경은 현지 설정 필요 |
| 미수행 검증 | 본 문서 개정 과정의 GPU 학습, simulator rollout 및 실기 검증 |

재현 자료에는 코드·asset·data·checkpoint 식별자와 설정, 실행 로그, 평가 결과를 함께 보관한다. 설정에 실행 경로가 존재하는 것과 성능 또는 실기 검증이 완료된 것은 구분한다.

본 문서의 적용 경로와 “현재 설정”은 명시된 코드 snapshot 기준이다. 원격에 포함되지 않은 후속 실험을 운영 완료 상태로 간주하지 않는다.

최근 recovery 계산 통합과 PC 정리 후 복구 방법은 [이관 안내](04_recovery_handoff.md)를 따른다.
