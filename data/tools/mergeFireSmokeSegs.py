import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


fire_path = './fireseg'
smoke_path = './smokeseg'
merged_path = './merged_seg'
fires = os.listdir(fire_path)
smokes = os.listdir(smoke_path)
fires.sort(key=lambda x: int(x.split('_')[1]))
os.makedirs(merged_path ,exist_ok=True)
for fire in fires:
    if fire not in smokes:
        continue
    fire_seg = np.asarray(Image.open(f'{fire_path}/{fire}'))
    smoke_seg = np.asarray(Image.open(f'{smoke_path}/{fire}'))
    filter_vec = fire_seg.sum(axis=2) > 0
    smoke_seg[filter_vec] = fire_seg[filter_vec]
    Image.fromarray(smoke_seg).save(f'{merged_path}/{fire}')

