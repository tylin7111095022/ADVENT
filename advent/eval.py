from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import argparse
import os 
from PIL import Image
import cv2
from tqdm import tqdm

from libs.models.deeplabv2 import get_deeplab_v2
from libs.dataset.datasets import ChromosomeDataset
from config import get_cfg_defaults

# 要預測的圖片路徑
IMG_PREDICT = r"F:\2023\chromosomes\ADVENT\advent\test_img\zong5img.jpg"

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str,default="./configs/advent_nopretrained.yml",
                        help='optional config file', )
    return parser.parse_args()

def main():
    args = get_arguments()
    cfg = get_cfg_defaults()

    if args.cfg:
        cfg.merge_from_file(args.cfg)

    model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
    print(f'Loading model {cfg.TEST.RESTORE_FROM[0]}')
    model.load_state_dict(torch.load(cfg.TEST.RESTORE_FROM[0], map_location="cpu",))
    print('Model loaded!')
    
    # evaluate images
    testset = ChromosomeDataset(img_dir=cfg.DATASET.IMGTEST,
                                mask_dir=cfg.DATASET.MASKTEST,
                                imgsize=cfg.TEST.INPUT_SIZE_TARGET,)
    miou = evaluate_imgs(net=model,testdataset=testset, cfg = cfg)
    print(f'miou = {miou:6.4f}')

    # predict one images
    predict_mask(net=model,imgpath=IMG_PREDICT)

def evaluate_imgs(net,
                testdataset,cfg):
    """需要修改，因為評估方式會將原始圖片跟ground truth 全都resize成固定大小"""
    net.eval()
    total_iou = 0
    count = 0
    miou_list = []
    for (img, truth) in tqdm(testdataset):
        img = img.unsqueeze(0)#加入批次軸
        img = img.to(dtype=torch.float32)
        truth = truth.unsqueeze(0).to(dtype=torch.int64)#加入批次軸
        # print("shape of img: ", img.shape)
        # print('shape of truth: ',truth.shape)
        with torch.no_grad():
            aux_prob, mask_pred_prob = net(img)
            interp = torch.nn.Upsample(size=(truth.shape[2], truth.shape[3]), mode='bilinear', align_corners=True)  # interpolate output segmaps
            mask_pred_prob = interp(mask_pred_prob)
            # if aux_prob:
            #     aux_prob = interp(aux_prob)
            mask_pred = torch.argmax(torch.softmax(mask_pred_prob, dim=1),dim=1,keepdim=True).to(torch.int32)
            # print('shape of mask_pred: ',mask_pred.shape)
            mask = mask_pred.squeeze(0).detach()#(1,h ,w)
            mask *= 255 #把圖片像素轉回255
            #compute the mIOU
            miou = compute_mIoU(mask_pred.numpy(), truth.numpy())
            # print(miou)
            miou_list.append(miou)
            # print('Mean Intersection Over Union: {:6.4f}'.format(miou))

    # return total_iou / count #回傳miou
    return sum(miou_list) / len(miou_list)

def compute_mIoU(pred, label):
    """計算整體類別的平均iou"""
    assert len(pred.shape) == len(label.shape), f"dim dismatch, pred dim:{len(pred.shape)} label dim {len(label.shape)}"
    assert len(pred.shape) == 4, "dim must be 4 , (BCHW)"
    if label.shape[1] == 1: #預測的map已經變為一張圖了
        pass
    else:
        label = pred[:,0,:,:]

    # print("shape of pred: ", pred.shape)
    # print("shape of label: ", label.shape)
    cf_m = confusion_matrix(label.flatten(), pred.flatten())
    # print(cf_m)
    intersection = np.diag(cf_m)  # TP + FN
    union = np.sum(cf_m, axis=1) + np.sum(cf_m, axis=0) - intersection # 模型預測全是某類的值 + 實際真的是該類的值 - 正確預測的值
    IoU = intersection / union
    mIoU = np.nanmean(IoU) # Compute the arithmetic mean along the specified axis, ignoring NaNs.

    return mIoU

def predict_mask(net,imgpath:str,threshold:float= 0.5):
    net = net.to(device="cpu")
    net.eval()
    img = torch.from_numpy(cv2.imread(imgpath)).permute(2,0,1)
    img = img.unsqueeze(0)#加入批次軸
    img = img.to(dtype=torch.float32, device='cpu')
    interp = torch.nn.Upsample(size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=True)  # interpolate output segmaps
    aux_pred, mask_pred_prob = net(img)
    mask_pred_prob = interp(mask_pred_prob)
    # if aux_pred:
    #     aux_pred = interp(aux_pred)
    mask_pred = torch.argmax(torch.softmax(mask_pred_prob, dim=1),dim=1,keepdim=True).to(torch.int32)
    print(f"mask_pred value: {np.unique(mask_pred.numpy())}")
    mask_pred = mask_pred.squeeze().numpy()
    mask_pred = mask_pred.astype(np.uint8)*255
    im = Image.fromarray(mask_pred)
    im.save(f"./predict_{os.path.basename(imgpath)}")

    return mask_pred

if __name__ == "__main__":
    main()