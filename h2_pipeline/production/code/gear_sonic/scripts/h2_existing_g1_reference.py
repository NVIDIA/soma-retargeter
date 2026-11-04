"""Read supplied Bones G1 results; convert units without fitting or retargeting."""
import csv
from pathlib import Path
import numpy as np

def normalize_existing_g1(source, destination, expected_frames):
    with Path(source).open() as f:
        rows=csv.reader(f);header=next(rows);data=np.array(list(rows),dtype=np.float64)
    if data.shape!=(expected_frames,36):
        raise ValueError(f'G1 source/frame shape mismatch: {source}: {data.shape}, expected {(expected_frames,36)}')
    if header[:7]!=['Frame','root_translateX','root_translateY','root_translateZ','root_rotateX','root_rotateY','root_rotateZ']:
        raise ValueError('Unsupported G1 CSV root layout: '+str(source))
    # Downloaded Bones CSV: cm, degrees. Internal audit CSV: metres, radians.
    data[:,1:4]/=100.0
    data[:,4:]=np.deg2rad(data[:,4:])
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.partial')
    with temp.open('w') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(data)
    temp.replace(destination)
