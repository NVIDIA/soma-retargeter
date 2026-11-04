from pathlib import Path
import csv
import json
import numpy as np
import pytest
from h2_pipeline.publish import convert_csv, joint_contract, certificate, ROOT_COLUMNS
from h2_pipeline.workspace import PRODUCTION, digest


def fixture(tmp_path, *, official=False, reordered=False):
    xml = PRODUCTION / 'models/mjcf/h2.xml'
    names = [r[0] for r in joint_contract(xml)]
    if reordered:
        names = names[::-1]
    header = ROOT_COLUMNS + [name + '_dof' for name in names]
    values = np.zeros((3, 38)); values[:, 0] = np.arange(3)
    values[:, 3] = 100 if official else 1
    values[:, 7 + names.index('left_elbow_joint')] = 10 if official else np.deg2rad(10)
    source = tmp_path / 'clip.csv'
    with source.open('w') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(values)
    bvh = tmp_path / 'clip.bvh'
    bvh.write_text('HIERARCHY\nMOTION\nFrames: 3\nFrame Time: 0.008333333333\n')
    return source, bvh, xml


def test_cm_degree_and_m_radian_convert_identically_by_joint_name(tmp_path):
    a = tmp_path / 'a'; b = tmp_path / 'b'; a.mkdir(); b.mkdir()
    csv1, bvh1, xml = fixture(a, official=True, reordered=True)
    csv2, bvh2, _ = fixture(b)
    x, _, _, names = convert_csv(csv1, bvh1, xml, units='official-cm-deg')
    y, _, _, _ = convert_csv(csv2, bvh2, xml, units='native-m-rad')
    for key in x:
        np.testing.assert_array_equal(x[key], y[key])
    assert x['fps'] == 120 and x['root_rot'].shape == (3, 4)
    assert x['root_rot'][0].tolist() == [0, 0, 0, 1]
    assert x['dof'][0, names.index('left_elbow_joint')] == np.float32(np.deg2rad(10))


def test_frame_mismatch_and_implicit_units_refused(tmp_path):
    source, bvh, xml = fixture(tmp_path)
    with pytest.raises(ValueError, match='explicit'):
        convert_csv(source, bvh, xml, units=None)
    bvh.write_text('Frames: 4\nFrame Time: 0.008333333333\n')
    with pytest.raises(ValueError, match='timing'):
        convert_csv(source, bvh, xml, units='native-m-rad')


def test_admission_cannot_be_inferred_from_file_existence(tmp_path):
    source, _, _ = fixture(tmp_path)
    report = dict(refined_sha256=digest(source), physical_and_margin_pass=True,
                  status='needs_review', checks=[{'passed': True}])
    with pytest.raises(ValueError, match='comparison'):
        certificate(report, source)
    report.update(status='accepted')
    certificate(report, source)
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='hash'):
        certificate(report, source)
