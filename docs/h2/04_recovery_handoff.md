# H2 recovery 계산과 이관 — 2026-11-09

`h2_pipeline/recovery`는 날짜별 recovery 제어 폴더에서 반복 사용하던 계산을 모은
소스다. 전체 retarget·검증·CSV→PKL 흐름은 기존 `h2_pipeline` 명령을 계속 사용한다.
이번 변경은 production의 source manifest, 목적 함수 또는 최종 승인 기준을 바꾸지 않는다.

| 모듈 | 역할 |
|---|---|
| `solution_bank.py` | hash로 고정한 donor bank, 자기 자신·mirror donor 제외 |
| `retrieved_start.py` | native 관절 범위·속도 투영, 팔 복구 시 반대쪽 팔 보존 |
| `fast_primitives.py`, `root_z_fast.py` | 모델 축을 확인한 torso/wrist 제안과 root 높이 제안 |
| `adaptive_stopping.py` | 유효 접촉 개선 유지, plateau 검출, 원래 최종 검사 보존 |
| `qualified_upper_fastpath.py`, `fixed_goal_fastpath.py` | 물리·원래 G1 envelope를 만족한 경우에만 계산 생략 |
| `geometry_transport.py`, `shared_gpu_geometry_safe.py` | 공유 geometry 요청·응답, 계산 중 연결 종료 처리 |

이 함수들은 production의 batch/problem 객체와 certified 원본 데이터가 필요하다.
donor 재사용 또는 fast path 통과만으로 training admission certificate를 만들지 않는다.
소규모 검사는 데이터·GPU·H100 접속 없이 실행할 수 있다.

```bash
python -m h2_pipeline.recovery.check_adaptive
python -m h2_pipeline.recovery.check_repair
python -m h2_pipeline.recovery.check_reuse
python -m h2_pipeline.recovery.check_small_error
python -m h2_pipeline.recovery.check_transport
```

각 모듈은 `python h2_pipeline/recovery/check_*.py` 방식으로도 실행할 수 있다.
NumPy/SciPy 외에 실제 QP 경로에는 OSQP와 선택한 solver가 필요하다.

`app/prepare_h2_training_review.py`와 `app/view_h2_training_review.py`는 선택한 H100
학습 데이터의 inventory와 비교용 viewer다. prepare는 파일 목록을 만들며 데이터를
자동으로 옮기지 않는다. 각각 `--help`에서 host·port·입력 경로를 지정한다.

과거 실험별 고유 제어 코드·설정은 비공개
`MFIWO/GR00T-WholeBodyControl`의 `h2-transfer-dev` 브랜치
`archive/h2_20261109/`에 경로와 내용 hash로 보존한다. 이 공개 저장소에는 개인
Pico 기록이나 모델 가중치를 넣지 않는다. 원본 로그·모델은 Git 전송 대상이 아니다.
이관 범위와 로컬 보관 위치는 WBC의 `docs/h2/HANDOFF.md`에 있다.
