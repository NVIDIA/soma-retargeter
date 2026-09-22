# H2 retarget: NVIDIA 원본 대비 변경과 손·발목 튜닝 기록

문서 개정: 2026-09-22 · 원격 대조: `305daa5a5aaa029b10348463d66d40d3ea44e670`.

검토 기준일: 2026-09-17. 본 문서는 개발 당시 관찰과 Git의 설정·구현 차이를 근거로 H2 retarget 변경 이력을 정리한다. 실행 명령과 확인 기준은 [H2 retarget 실행 가이드](00_execution_runbook.md)를 참조한다.

## 1. 비교한 원본과 변경 이력

| 대상 | 고정 기준 |
|---|---|
| 당시 NVIDIA SOMA | [`b3ef2708`](https://github.com/NVIDIA/soma-retargeter/tree/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57), 2026-03-25, v0.1 계열 |
| 첫 H2 수정 | [`99f166e9`](https://github.com/MFIWO/soma-retargeter/commit/99f166e9045cb2bf52a2bfc1d5cfa89d9cee3f30), 2026-04-25, 원본 commit의 직접 자식 |
| 이후 H2 support 묶음 | [`cf4a5910`](https://github.com/MFIWO/soma-retargeter/commit/cf4a59105275aef2d3693f7c7e6803f114590911), 2026-05-27 |
| 현재 H2 branch의 코드 snapshot | [`b2d7ce7a`](https://github.com/MFIWO/soma-retargeter/tree/b2d7ce7a5584c2bd290042baed83cf0c69256873), 2026-07-27 |
| 조사 당시 최신 NVIDIA | [`1733b820`](https://github.com/NVIDIA/soma-retargeter/commit/1733b820f3cdf6f74bbc81a10bda3201b38c7bcf), 2026-09-15, v0.2.0 multi-embodiment |

당시 원본에 있던 BVH 처리, human-to-robot scaler, Newton IK, joint-limit objective, feet stabilizer, CSV export를 재사용했다. fork는 G1 전용 robot/config/asset 선택을 H2에도 적용하고 31-DoF 출력 계약과 로컬 사용 경로를 추가했다. H2 브랜치에 뒤의 T1 추가 commit도 들어 있지만 여기서는 H2 활성 설정을 설명한다.

최신 NVIDIA v0.2에는 [H2 config](https://github.com/NVIDIA/soma-retargeter/blob/1733b820f3cdf6f74bbc81a10bda3201b38c7bcf/soma_retargeter/assets/robotics/unitree/h2/configs/soma_to_h2_retargeter_config.json)와 T1 config가 이미 있다. 본 비교의 자체 변경 범위는 당시 사용한 v0.1 기준으로 해석한다. v0.2는 `ik_match_table`, mask 배열, post/contact processing 등 schema가 달라 기존 fork JSON을 무검증으로 복사하면 안 된다. v0.2 migration 및 버전 간 품질 비교는 본 검토 범위에 포함되지 않는다.

## 2. 초기 관찰 및 기록 범위

**초기 관찰(개발 담당자 보고):** G1 기반 설정 적용 시 H2의 손·발목 retarget이 부자연스럽게 나타났다. 담당자는 관련 objective weight와 후처리 설정을 조정하는 과정에서 개선을 관찰했다. 변경별 기여도는 별도 평가 자료가 필요하다.

**확인한 저장 위치:** 이 브랜치의 retarget 관련 수치는 YAML이 아니라 `soma_retargeter/configs/h2/`의 JSON에 있다. H2 본 IK config의 Git path history는 `99f166e9` 한 commit만 반환한다. 최종값이 묶여 저장돼 있어, 5.5→중간값A→중간값B→30처럼 실제 시도한 sweep 순서나 각 시도의 영상을 복원할 수 없다.

아래는 **원본 G1 config와 commit에 남은 H2 config의 차이**다. 개발 관찰을 근거 수준에 맞게 구분하되, 이 표를 모든 중간 실험의 before/after 성능 증거로 취급하지 않는다.

## 3. 본 IK: 실제로 바뀐 값과 그대로인 값

출처: [원본 G1 retarget config](https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/configs/unitree_g1/soma_to_g1_retargeter_config.json), [H2 retarget config](../../soma_retargeter/configs/h2/soma_to_h2_retargeter_config.json).

| 항목 | 원본 G1 v0.1 | H2 최종 기록 | 의미 |
|---|---:|---:|---|
| `model_height` | 1.70 | 1.8 | target 설정값; MJCF 교체를 대신하지 않음 |
| IK iterations | 24 | 50 | 최적화 반복 증가 |
| joint-limit weight | 10 | 50 | 관절 범위 제약 강화 |
| smooth joint weight | 5.5 | 30 | joint filter objective의 영향 증가 |
| Hips position / rotation | 30 / 2 | 40 / 10 | pelvis 목표 강화 |
| Chest position / rotation | 0.7 / 0.7 | 0 / 1.5 | 위치 제약 해제, 회전 제약 조정 |
| Shin position / rotation | 1 / 1 | 10 / 1 | knee 위치 목표 강화 |
| Arm position / rotation | 1.5 / 0.15 | 동일 | 어깨 목표 weight 유지 |
| ForeArm position / rotation | 1 / 1 | 동일 | 팔꿈치 목표 weight 유지 |
| Hand position / rotation | 2 / 1.2 | 동일 | 손 weight 자체는 이 비교에서 바뀌지 않음 |
| Foot position / rotation | 30 / 2 | 동일 | 본 IK 발목 weight 자체는 유지 |
| 초기화 / 안정화 frames | 10 / 5 | 동일 | 초기 구간 길이 유지 |
| collision weight / postprocessing | 0 / true | 동일 | 이 설정은 collision penalty를 활성화하지 않음 |

양측에 동일한 조정이다. hand→wrist yaw, foot→ankle roll body 대응은 이름상 G1과 같아도 로봇 geometry와 FK는 H2 MJCF를 따라야 한다. 손·발 결과는 pelvis/knee/scaler/feet postprocessing에 의해서도 바뀌므로 “손과 발목의 직접 weight를 모두 바꿨다”고 단순화하지 않는다.

## 4. smooth joint weight의 정확한 의미

[원본 `IKSmoothJointFilter`](https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/pipelines/ik_objectives.py)는 관절 좌표·limit 중심·nonlinear filter로 penalty를 만든다. residual에 `q(t)-q(t-1)`이 없으므로 시간축 저역통과 필터와 다르다. 로봇의 모터 Kp/Kd를 변경하는 것도 아니다.

| body mask | 원본 G1 | H2 |
|---|---:|---:|
| shoulder pitch / roll / yaw | 0.1 / 1 / 0.3 | 동일 |
| pelvis / torso | 명시 없음 | 1 / 1 |
| hip pitch / hip yaw | 명시 없음 | 1 / 0.8 |
| wrist / ankle | 명시 없음 | 명시 없음 |

확인한 pipeline은 body mask가 제공되면 다른 좌표를 기본0으로 만든다. 따라서 5.5→30을 **손목·발목 전체에 직접 적용한 smoothing gain 증가**로 설명하면 부정확하다. 몸통·근위 관절 해가 달라져 끝점 결과가 간접적으로 바뀔 수 있다. 어떤 경로가 당시 개선을 주도했는지는 clip별 ablation이 없다.

## 5. scaler와 feet postprocessing도 함께 달라졌다

[H2 scaler](../../soma_retargeter/configs/h2/soma_to_h2_scaler_config.json)와 [원본 G1 scaler](https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/configs/unitree_g1/soma_to_g1_scaler_config.json)의 비교:

| `joint_scales` 그룹 | G1 | H2 |
|---|---:|---:|
| Hips | 0.82 | 1.0 |
| Chest / Neck | 0.80 | 1.0 |
| Leg / Shin | 0.86 | 1.0 |
| Foot | 0.82 | 1.0 |
| Toe / ToeBase | 0.78 | 0.78 |
| Arm / ForeArm / Hand | 0.85 | 0.85 |

`human_height_assumption=1.8`, joint parent 구조, 기록된 joint offsets는 이 두 JSON에서 동일하다. 모든 축·offset을 H2용으로 새로 튜닝한 것으로 기록하지 않는다. body link의 기하와 scaling·offset 적용 결과를 함께 확인해야 한다.

[H2 feet stabilizer](../../soma_retargeter/configs/h2/h2_feet_stabilizer_config.json)와 [원본 G1 feet stabilizer](https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/configs/unitree_g1/g1_feet_stabilizer_config.json):

| 후처리 항목 | G1 | H2 |
|---|---:|---:|
| IK iterations | 20 | 50 |
| joint-limit weight | 10 | 40 |
| pelvis position / rotation | 30 / 8 | 동일 |
| knee position / rotation | 1 / 1 | 10 / 5 |
| ankle position / rotation | 10 / 2 | 8 / 2 |
| 양다리 hint offset | `[0.25, 0, 0.25]` | `[0.1, 0, 0.05]` |

**본 IK의 ankle 30/2와 후처리 ankle 8/2는 서로 다른 단계의 값이다.** 발목 관련 조정은 후처리에도 남아 있다. knee의 굽힘 방향 hint와 pelvis/knee/ankle의 상대 weight가 달라지므로 본 IK JSON 하나만 보아서는 전체 변경을 설명할 수 없다.

## 6. 코드 구조와 출력 계약의 변경

첫 H2 commit은 다음을 함께 수정했다.

- `pipelines/utils.py`: `TargetType.H2`, H2 config 선택, robot별 MJCF 해석 추가.
- `newton_pipeline.py` 및 `feet_stabilizer.py`: G1 asset만 만드는 분기를 robot별 asset 선택으로 일반화.
- `assets/csv.py`: H2용 joint schema·읽기/쓰기 확장.
- converter와 default JSON: H2 입력·출력 흐름 연결.

단순히 `robot_type` 문자열만 바꾼 작업이 아니다. 현재 branch의 asset 해석에는 로컬 WBC checkout 경로 의존성이 있으므로 다른 머신에서는 [pipeline 안내](01_retarget_pipeline.md)에 따라 MJCF 위치를 준비한다. CSV header와 실제 joint order를 대조하고, base/root quaternion·FPS·단위까지 맞춘 뒤 WBC motion library로 넘긴다. [검증·인계](02_validation_handoff.md).

T1 balanced의 94% reach conditioning·8 Hz 시간축 필터·T1 audit/sharding을 H2 branch에서 이미 사용 중인 기능으로 적지 않는다. 두 브랜치의 날짜와 code snapshot이 다르다.

## 7. 튜닝 결과 관리 및 학습 인계 기준

추가 확보 자료는 **초기 실패 BVH/CSV, 중간 config, 동일 clip의 전후 영상, 변경별 정량 효과**다. 해당 자료가 확보되기 전까지 최종 설정의 적용 범위와 개별 조치의 기여도는 제한적으로 해석한다.

다음 튜닝에서는 같은 입력·asset·FPS로 본 IK와 feet postprocessing을 각각 비교하고, 손 endpoint/orientation, knee/ankle ROM·limit saturation, stance foot slip/tilt, joint 속도 spike를 함께 남긴다. retarget에서 이미 틀어진 reference와 policy가 reference를 따라가지 못하는 경우를 분리한다. 고정 motion batch나 teacher 보존은 학습 중 forgetting을 다루지만, 잘못된 retarget 자체를 수정하는 대체 수단은 아니다.
