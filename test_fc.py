import numpy as np
import sys
import json
sys.path.insert(0, '.')
from datasets.pds3_reader import read
cube, _, _ = read('data/ato00027155_01_if126s_trr3.img', None)

bands = (26, 53, 80)
rgb = np.stack([cube[bands[0]], cube[bands[1]], cube[bands[2]]], axis=-1)

valid = cube[cube != 0.0]

output = {
    'cube_shape': list(cube.shape),
    'valid_mean': float(valid.mean()),
    'valid_std': float(valid.std()),
    'channels': []
}

for i in range(3):
    ch = rgb[:, :, i].astype(np.float64) # Force 64-bit for precision test
    v = ch[ch != 0]
    lo = float(np.percentile(v, 2))
    hi = float(np.percentile(v, 98))
    
    out_ch = np.clip((ch - lo) / (hi - lo), 0, 1)
    out_ch[ch == 0] = 0.0
    u = np.unique(out_ch)
    
    output['channels'].append({
        'lo': lo,
        'hi': hi,
        'unique_v_count': len(np.unique(v)),
        'unique_out_ch_count': len(u),
        'min_out': float(out_ch.min()),
        'max_out': float(out_ch.max()),
        'samples': [float(x) for x in u[:5]] + [float(x) for x in u[-5:]]
    })

with open('fc_out.json', 'w') as f:
    json.dump(output, f, indent=2)
