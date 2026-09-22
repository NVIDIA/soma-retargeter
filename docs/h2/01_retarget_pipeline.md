# SOMA BVH → H2 CSV

문서 개정: 2026-09-22 · 원격 대조: `305daa5a5aaa029b10348463d66d40d3ea44e670`.

작성 기준: 2026-09-17 · `MFIWO/soma-retargeter` / `h2-retarget-support` / `b2d7ce7a5584c2bd290042baed83cf0c69256873`.

본 문서는 명시된 코드 버전의 구현·설정·실험 기록을 기준으로 작성한 기술 인수인계 자료다. 단계별 실행 명령과 확인 기준은 [실행 가이드](00_execution_runbook.md)를 참조한다.

원본 대비 수치·코드 변경과 개발 당시 튜닝 기록은 [변경 및 검증 이력](03_upstream_changes_and_tuning.md)에 정리했다. IK joint filter weight, 시간축 filtering, actuator PD gain을 구분한다.

## 1. 환경과 asset

```bash
git clone --branch h2-retarget-support --single-branch \
  https://github.com/MFIWO/soma-retargeter.git soma-retargeter-h2
cd soma-retargeter-h2
git lfs install
git lfs pull
conda create -n soma-retargeter-h2 python=3.12 pip tk -y
conda activate soma-retargeter-h2
python -m pip install -e .
```

이미 준비된 환경은 재생성할 필요가 없다. 이 브랜치 `pyproject.toml`은 Python ≥3.12, Newton 1.0.0, warp-lang 1.12.0 등을 고정한다. 원래 [설치 안내](../../README.md)와 [의존성](../../pyproject.toml)을 함께 확인한다. LFS pointer만 받아 실제 로봇 asset이 비어 있지 않은지 확인한다.

## 2. 입력과 출력의 로컬 설정

BONES-SEED의 SOMA BVH와 데이터 사용 조건을 준비한다. checked-in JSON에는 작성자의 절대 경로가 들어 있으므로 로컬 복사본으로 바꾼다. 실행 위치는 repository root다.

```bash
mkdir -p artifacts
cp assets/default_bvh_to_csv_converter_config.json artifacts/h2_local.json
```

복사한 JSON을 다음 구조로 편집한다. `/absolute/path/...`는 실제 존재하는 경로로 바꾼다. converter는 import/export를 JSON에서 읽는다.

```json
{
  "import_folder": "/absolute/path/to/soma_uniform/bvh",
  "export_folder": "/absolute/path/to/h2_csv",
  "batch_size": 4,
  "retargeter": "Newton",
  "retarget_source": "soma",
  "retarget_target": "h2",
  "retarget_source_facing_direction": "Mujoco"
}
```

## 3. 소수 clip부터 변환

```bash
python app/bvh_to_csv_converter.py \
  --config artifacts/h2_local.json --viewer null --device cpu
```

`--viewer gl`로 source/robot 포즈를 확인할 수 있다. CPU 예시는 기능 확인용이며 대규모 throughput 보장은 아니다. `batch_size`는 retarget batch 크기이고 PPO의 env 수가 아니다. 출력은 입력 상대 디렉터리 구조에 대응하는 CSV다.


## H2 target 구성

`soma_to_h2_retargeter_config.json`은 `Hips→pelvis`, `Chest→torso_link`, 손 target→wrist yaw link, 발 target→ankle roll link를 사용한다. scaler/feet-stabilizer도 H2 전용 JSON을 연결한다. 설정의 `model_height=1.8`은 retarget 모델 스케일 값이다.

코드 기준 IK 50회, joint-limit weight 50, smooth weight 30, collision weight 0, initialization/stabilization 10/5 frame이다. 발 위치·몸통·손 가중치를 바꿀 때 walking뿐 아니라 squat, turn, hand reach를 함께 비교한다. post-processing이 켜져 있어도 동역학적 균형을 보증하지 않는다.

이 브랜치의 converter에는 T1 최신 브랜치의 `--num-shards`, `--shard-index`, `--batch-size` 추가 옵션이 없다. batch 크기는 JSON에서 수정한다. T1 README의 CLI를 그대로 복사하지 않는다.


## 4. 다음 단계

[CSV 검증과 학습 인계](02_validation_handoff.md)를 통과한 자료만 학습 manifest에 포함한다. 이 단계에서는 RL 학습이나 실기 명령을 실행하지 않는다.

## 구현 근거

- [app/bvh_to_csv_converter.py](https://github.com/MFIWO/soma-retargeter/blob/b2d7ce7a5584c2bd290042baed83cf0c69256873/app/bvh_to_csv_converter.py)
- [soma_retargeter/configs/h2/soma_to_h2_retargeter_config.json](https://github.com/MFIWO/soma-retargeter/blob/b2d7ce7a5584c2bd290042baed83cf0c69256873/soma_retargeter/configs/h2/soma_to_h2_retargeter_config.json)
- [soma_retargeter/pipelines/newton_pipeline.py](https://github.com/MFIWO/soma-retargeter/blob/b2d7ce7a5584c2bd290042baed83cf0c69256873/soma_retargeter/pipelines/newton_pipeline.py)
