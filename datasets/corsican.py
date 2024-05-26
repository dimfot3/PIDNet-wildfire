# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .base_dataset import BaseDataset

class Corsican(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=1024, 
                 crop_size=(720, 960),
                 scale_factor=16,
                 mean=[0.21562974739056145, 0.27942731011997907, 0.402625553988713, 0.1878679056174089],
                 std=[0.1608569349670365, 0.1885546372797602, 0.28214404172689506, 0.17329260037921992],
                 bd_dilate_size=4, nir='rgb'):

        indices = {'rgb': [0, 1, 2], 'nir': [3], 'fusion': [0, 1, 2, 3]}       # based on nir mode
        mean = [mean[i] for i in indices[nir]]
        std = [std[i] for i in indices[nir]]
        super(Corsican, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line[:-1] for line in open(root+list_path)]

        self.files = self.read_files()

        self.ignore_label = ignore_label
        
        self.color_list = [[0, 0, 0], [255, 255, 255]]
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size

        self.nir = nir

    
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, nir_path, label_path = item.replace('XXX', 'rgb'), item.replace('XXX', 'nir'), item.replace('XXX', 'gt')
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "nir": nir_path,
                "label": label_path,
                "name": name.replace('_gt_', '_')
            })
            
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]
            
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        if self.nir=='rgb':
            image = np.array(Image.open(os.path.join(self.root,'images',item["img"])).convert('RGB'))
        elif self.nir=='nir':
            image = np.array(Image.open(os.path.join(self.root,'images',item["nir"])).convert('L'))
            image = image.reshape(image.shape[0], image.shape[1], 1)
        elif self.nir=='fusion':
            image = np.array(Image.open(os.path.join(self.root,'images',item["nir"])).convert('L'))
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image1 = np.array(Image.open(os.path.join(self.root,'images',item["img"])).convert('RGB'))
            image = np.append(image, image1, 2)
        size = image.shape
        color_map = Image.open(os.path.join(self.root,'images',item["label"])).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        if(len(preds.shape)>3):
            preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        else:
            preds = np.asarray(preds.cpu(), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
if __name__ == '__main__':
    dataset = Corsican('../data/CorsicanDB/', 'list/train.txt')
    mean = np.array(dataset.mean)
    std = np.array(dataset.std)
    for i in np.random.choice(len(dataset), 3):
        img, label, edge, size, _ = dataset[i]
        f, ax = plt.subplots(1, 3)
        img = img.transpose((1, 2, 0))
        img = (((img * std) + mean) * 255).astype('int')
        gt_img = dataset.label2color(label).astype('int')
        edge = edge.astype('int')
        edge[edge==1] = 255
        ax[0].imshow(img)
        ax[1].imshow(edge)
        ax[2].imshow(gt_img)
        plt.show()

        
