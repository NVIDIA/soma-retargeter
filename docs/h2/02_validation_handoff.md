# 검사와 학습 인계 계약

최신 명령은 [BVH→CSV→SQP/QP→PKL](01_retarget_pipeline.md), 과거 경위는 [상류 대비 변경](03_upstream_changes_and_tuning.md)에 있다.

| 경계 | 필요한 근거 |
|---|---|
| BVH→후보 CSV | 원본 BVH/G1 해시,120Hz/frame ID, source/target trace, 봉인 모델·중립 보정 |
| 원래 G1 기준 통과 | 팔 방향·팔꿈치·손목·몸통 좌우/방향·발 통계, 관절 범위·URDF 속도·어깨/손목2도 여유 |
| 새 H2 기준 통과 | 동결 목표의 독립 추종, BVH 주요 지표 비악화, 보정 후 root/다리 보존, 원래 root 회전 보존, 발120/50/600Hz 검사 |
| CSV→PKL | 명시적 m/radian, 관절 이름 재배열, XYZW, float32 재검사, 보고서·입력 해시 일치 |
| PKL→학습 | paired SMPL/SOMA, 시간·좌표계·관절 순서·관측/action 계약, 별도 학습 manifest 검증 |
| 정책→실기 | 동일 입력의 중립·팔·낮은 자세·보행 추종, 동역학과 제어기 검증 |

`new_H2_reference_validated`는 기존 G1 지표를 모두 동일하게 통과했다는 뜻이 아니다. 원래 실패 지표를 보고서에 남기며, 새로 동결한 H2 목표의 추종과 원본 대비 비악화 등을 확인한 별도 범주다. 통과와 보류를 합쳐 같은 품질로 표시하지 않는다.

`h2_pipeline/production/source_manifest.json`은 원본 코드와 SHA256을, `workspace.json`은 실행 경로와 경로 치환을, `bundle_manifest.json`은 코드·모델·중립 수치·메시를 고정한다. 재실행 중 바뀌면 중단한다.

`publish`는 파일 존재만으로 승인하지 않는다. 물리 검사와 원래 G1 통과 또는 새 H2 통과 상태, CSV 해시와 원본 trace 일치를 확인한다. 실제 float32 PKL의 발도 다시 검사하므로 반올림으로 경계를 넘으면 보류할 수 있다.

검증은 `tests/test_h2_handoff.py`에 있다. 고정8,192모션·16,384환경·2GPU 학습 recipe는 [GR00T-WholeBodyControl](https://github.com/MFIWO/GR00T-WholeBodyControl/tree/h2-transfer-dev/recipes/h2_low_work)에서 관리한다. 체크포인트·원본 모션·학습 출력·리타게팅 데이터는 Git에 올리지 않는다.
