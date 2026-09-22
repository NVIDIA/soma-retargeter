# SOMA → H2 retarget 실행 및 검증 절차

문서 개정: 2026-09-22 · 원격 대조: `305daa5a5aaa029b10348463d66d40d3ea44e670`.

문서 기준: 2026-09-17 · `h2-retarget-support` · 검토 코드 `b2d7ce7a5584c2bd290042baed83cf0c69256873`.

## 1. 기존 fork 재현 범위 및 준비 사항

본 절차는 SOMA BVH를 H2 31-DoF CSV로 변환한다. 정지·걷기·회전·앉기·손 reach의 소수 clip으로 먼저 확인한다. 명령은 Bash와 retarget 저장소 루트 기준이며 `/absolute/path/...`는 실제 경로로 교체한다.

Python 3.12의 retarget 전용 환경과 저장소를 준비한다. 동일 환경이 있으면 `conda create`를 생략한다. Git LFS와 conda 설치가 선행되어야 한다.

```bash
set -euo pipefail
git clone --branch h2-retarget-support --single-branch \
  https://github.com/MFIWO/soma-retargeter.git soma-retargeter-h2
cd soma-retargeter-h2
git lfs install
git lfs pull
conda create -n soma-retargeter-h2 python=3.12 pip tk -y
conda activate soma-retargeter-h2
python --version
python -m pip install -e .
git rev-parse HEAD
```

기존 checkout에서는 clone을 생략한다. 로봇 MJCF·참조 mesh, 원본 BVH의 Frame Time, 출력 디렉터리를 준비한다.

## 2. H2 asset 경로 설정

이 branch는 `soma_retargeter/pipelines/utils.py`의 `_H2_MJCF_PATH` 상수로 H2 모델을 찾는다. T1 branch의 환경변수 override 방식과 다르므로, 다른 머신에서는 해당 상수를 현지 WBC checkout 경로로 수정한다.

```bash
rg -n '_H2_MJCF_PATH' soma_retargeter/pipelines/utils.py
"${EDITOR:-vi}" soma_retargeter/pipelines/utils.py
```

수정 대상은 다음 형태의 **상수 한 줄**이다.

```python
_H2_MJCF_PATH = pathlib.Path("/absolute/path/to/GR00T-WholeBodyControl-h2/gear_sonic/data/assets/robot_description/mjcf/h2.xml")
```

경로와 변경 내역을 확인한다.

```bash
mkdir -p artifacts
python - <<'PY'
from soma_retargeter.pipelines.utils import TargetType, get_robot_mjcf_path
path = get_robot_mjcf_path(TargetType.H2)
assert path.is_file(), path
print(path)
PY
git diff -- soma_retargeter/pipelines/utils.py > artifacts/h2_asset_path.patch
```

**확인 결과:** 실제 H2 MJCF 경로 출력. mesh 상대 경로까지 유효해야 한다. 본 절차의 명령 예시는 로컬 설정 변경이며, 저장소 코드에 자동으로 반영되는 기능은 아니다.

## 3. 입력·출력 설정과 CSV 생성

```bash
export SOMA_BVH_DIR=/absolute/path/to/small_bvh_subset
export SOMA_CSV_DIR=/absolute/path/to/h2_csv_v1
test -d "$SOMA_BVH_DIR"

python - <<'PY'
import json, os
from pathlib import Path
config = {
    "import_folder": os.environ["SOMA_BVH_DIR"],
    "export_folder": os.environ["SOMA_CSV_DIR"],
    "batch_size": 4,
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "h2",
    "retarget_source_facing_direction": "Mujoco"
}
Path("artifacts/h2_local.json").write_text(json.dumps(config, indent=2) + "\n")
PY
python -m json.tool artifacts/h2_local.json

python app/bvh_to_csv_converter.py \
  --config artifacts/h2_local.json --viewer null --device cpu \
  2>&1 | tee artifacts/h2_retarget.log
```

**산출물:** 입력 상대 경로에 대응하는 H2 CSV와 실행 로그. H2 retarget/scaler/feet 설정은 `soma_retargeter/configs/h2/`에서 읽는다. 이 branch에는 T1 최신 converter의 `--batch-size`, `--num-shards` 옵션이나 custom `retargeter_config` override가 없다. batch 크기는 로컬 JSON에서, IK/scaler/feet 값은 해당 H2 설정 파일에서 조정한다. 설정 변경 실험은 별도 CSV 출력 경로를 사용한다.

## 4. CSV 및 손·발목 품질 확인

### 4.1 schema·joint order·유한값 검사

```bash
python - <<'PY'
import csv, os
from pathlib import Path
import numpy as np
from soma_retargeter.assets.csv import get_csv_config
files = sorted(Path(os.environ["SOMA_CSV_DIR"]).rglob("*.csv"))
assert files, "No CSV outputs"
expected = get_csv_config("h2").csv_header
for path in files:
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == expected, f"Schema/order mismatch: {path}"
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    assert data.shape[0] > 1 and data.shape[1] == len(expected), path
    assert np.isfinite(data).all(), f"NaN/Inf: {path}"
    print(path.name, "frames=", len(data), "dof=", 31)
print("PASS: CSV schema, joint order and finite values")
PY
```

**확인 기준:** head·wrist를 포함한 31축, ankle roll/pitch 순서, 유효 frame 수와 NaN/Inf 부재. FPS는 source BVH와 별도로 대조한다.

### 4.2 source/robot preview

GUI 환경에서 소수 clip이 들어 있는 동일 config로 preview한다.

```bash
python app/bvh_to_csv_converter.py \
  --config artifacts/h2_local.json --viewer gl --device cpu
```

| 관찰 항목 | 점검 내용 |
|---|---|
| 손 | source reach와 robot 손 위치·orientation, wrist limit 포화 |
| 무릎·발목 | knee 굽힘 방향, stance foot slip/tilt, 지면 침투 |
| 몸통·root | 높이·자세 연속성, 목표 끝점과의 동시 만족 여부 |
| 시간축 | 순간적인 joint jump, 원본 대비 시간 길이·속도 |

이 branch에는 H2 전용 자동 품질 audit CLI가 없다. 위 schema 검사와 preview 결과를 기록하고, 필요 시 H2 FK 기반 오차를 별도로 산출한다. T1 audit 도구는 T1 모델 전용이다.

손·발목 문제가 발생하면 본 IK weight, scaler, feet 후처리를 나누어 비교한다. 본 IK의 foot weight와 후처리 ankle weight는 서로 다른 단계이며, `smooth_joint_filter_weight`는 모터 PD gain이나 시간축 cutoff가 아니다. 수치와 변경 근거는 [원본 대비 변경 및 튜닝 이력](03_upstream_changes_and_tuning.md)에 정리되어 있다.

## 5. 설정 기록 및 WBC 인계

```bash
sha256sum artifacts/h2_local.json \
  soma_retargeter/configs/h2/soma_to_h2_retargeter_config.json \
  soma_retargeter/configs/h2/soma_to_h2_scaler_config.json \
  soma_retargeter/configs/h2/h2_feet_stabilizer_config.json \
  > artifacts/h2_retarget_configs.sha256
git rev-parse HEAD > artifacts/h2_retarget_commit.txt
git diff -- soma_retargeter/pipelines/utils.py soma_retargeter/configs/h2 \
  > artifacts/h2_local_changes.patch
```

별도로 실제 H2 MJCF와 참조 asset hash, BVH/CSV 대응표, FPS·단위, 실패·제외 목록, 전후 preview 결과를 보관한다. CSV의 기하학적 적합성 확인 후 [WBC H2 실행 절차](https://github.com/MFIWO/GR00T-WholeBodyControl/blob/h2-transfer-dev/docs/h2/00_execution_runbook.md)에서 PKL 변환·G1 전이·학습을 진행한다. 자세한 인계 항목은 [검증·인계 문서](02_validation_handoff.md)를 따른다.
