import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class ChromosomeDataset(Dataset):
    f'''
    回傳img(torch.uint8)(4軸)及label(torch.float32)(4軸)\n
    '''
    def __init__(self,img_dir,mask_dir,imgsize, transform = None):
        """imgsize 可以為某個特定的正整數或是 None 如果為None,img 跟 label 不被resize"""
        # assert len(os.listdir(img_dir)) == len(os.listdir(mask_dir)), "numbers of img and label dismatch."
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = [i for i in os.listdir(img_dir)]
        self.imgsize = imgsize
        
        self.transforms = transform

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_a = _load_img(file=img_path,size=self.imgsize, interpolation=cv2.INTER_CUBIC,rgb=True)
        if img_a.ndim == 2:
            img_a = np.expand_dims(img_a,2) #
            
        img = torch.permute(torch.from_numpy(img_a),(2,0,1))

        if self.mask_dir:
            label_name = img_name
            label_path = os.path.join(self.mask_dir, label_name)
            label = _load_img(file=label_path,size=self.imgsize, interpolation=cv2.INTER_NEAREST,rgb=False)
    
            if label.ndim == 2:
                label = np.expand_dims(label,2) #如果沒通道軸，加入通軸
            
            #label內的值不只兩個，這導致除以255後值介於0~1的值在後續計算iou將label轉回int64的時候某些值被無條件捨去成0
            ret,label_binary = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
            if label_binary.ndim == 2:
                label_binary = np.expand_dims(label_binary,0) #如果沒通道軸，加入通軸
            # print(np.unique(label_binary))
            label_t = torch.from_numpy(label_binary).to(torch.float32) # (n, h, w)
            # 處理標籤，将像素值255改為1
            if label_t.max() > 1:
                label_t = label_t / 255
        else:
            label_t = torch.zeros_like(img, dtype = torch.int64)

        if self.transforms:
            img = self.transform(img)
            if self.mask_dir:
                label_t = self.transform(label_t)

        return img,label_t


    def __len__(self):
        return len(self.img_list)
    
def _load_img(file, size, interpolation, rgb:bool):
    """size可以為正整數或None或0"""
    if rgb:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if size: #如果size為正整數
        img = cv2.resize(img, (size,size), interpolation)
    return np.asarray(img, np.float32)

if __name__ == "__main__":
    ds = ChromosomeDataset(img_dir = r'F:\2023\chromosomes\ADVENT\advent\data\Chang_Gung\images',
                           mask_dir= r'F:\2023\chromosomes\ADVENT\advent\data\Chang_Gung\masks',
                           imgsize=256)
    img, mask = ds[10]
    print(f"img shape: {img.shape}")
    print(f"mask shape: {mask.shape}")

    cv2.imwrite("test.jpg", img.permute(1,2,0).numpy())
    cv2.imwrite("test_mask.jpg", (mask*255).permute(1,2,0).numpy(),)