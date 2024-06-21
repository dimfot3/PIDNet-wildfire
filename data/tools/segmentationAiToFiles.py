
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def download_dataset(apikey, username, datasetname, version):
    client = SegmentsClient(apikey)
    release = client.get_release(f'{username}/{datasetname}', version) 
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])
    export_dataset(dataset, export_format='semantic')

def save_segmentations(username, datasetname, version, out_folder):
    files = os.listdir(f'./segments/{username}_{datasetname}/{version}')
    files = [file for file in files if ('semantic' in file)]
    files_newnames = [file.replace('_ground-truth_semantic', '').replace('ir', 'gt') for file in files]
    os.makedirs(out_folder, exist_ok=True)
    for i,file in enumerate(files):
        shutil.copy(f'./segments/{username}_{datasetname}/{version}/{file}', f'{out_folder}/{files_newnames[i]}')

def change_colormap(out_folder, colors):
    files = os.listdir(out_folder)
    colors = np.array(colors).astype('uint8')
    for file in files:
        img = np.array(Image.open(f'{out_folder}/{file}'))
        new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        new_img = Image.fromarray(colors[img])
        new_img.save(f'{out_folder}/{file}')
        

if __name__ == '__main__':
    apikey = 'd0c976406a67abb5df2507d3f42fc95c213732d6'
    username = 'dimfot3'
    datasetname = 'finlandsmoke'
    version = 'v1'
    out_folder = './smokeseg'
    colors = [[0, 0, 0], [125, 125, 125]]
    download_dataset(apikey, username, datasetname, version)
    save_segmentations(username, datasetname, version, out_folder)
    change_colormap(out_folder, colors)
