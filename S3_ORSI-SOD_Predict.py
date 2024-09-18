import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, argparse
import imageio
from network.SimSOD import SimSOD 
from setting.dataLoader import test_dataset
from tqdm import tqdm
import time
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, 
        default='./EORSSD/', 
        help='Name of dataset:[EORSSD, ORSSD, ORSI4199]')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument("--model", type=str, default='SimSOD',
        help='model name:[SimSOD]')

parser.add_argument("--smap_save", type=str, default='../SalPreds/',
        help='model name')
parser.add_argument("--load", type=str,
            default='',
              help="restore from checkpoint")

opt = parser.parse_args()


def create_folder(save_path):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Create Folder [“{save_path}”].")
    return save_path


model = eval(opt.model)()

if opt.load is not None and os.path.isfile(opt.load):
    checkpoint = torch.load(opt.load, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    print("Model restored from %s" % opt.load)
    
model.cuda()
model.eval()


test_datasets = ['ORSSD', 'EORSSD', 'ORSI4199']
for dataset in test_datasets:
    # load data
    image_root = opt.test_path + dataset + '/Test/Images/'
    depth_root = opt.test_path + dataset + '/Test/Depth/'
    gt_root = opt.test_path + dataset + '/Test/Masks/'
    test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)
    method=opt.load.split('/')[-1].split('.')[0]
    save_path = create_folder(opt.smap_save + dataset + '/'+method+'/')
    print('{} preds for {}'.format(method, dataset))
   
    cost_time = list()
    for i in tqdm(range(test_loader.size), desc=dataset):
        with torch.no_grad():
            image, depth, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            name = name.split('/')[-1]
            image = image.cuda()
            depth = depth.cuda()
            start_time = time.perf_counter()
            sal, sal_sig = model(image)
            res=sal
            cost_time.append(time.perf_counter() - start_time)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res = res.convert('RGBA')
            # imageio.imsave(save_path+name, res)
            # cv2.imwrite(save_path+name, res)
            im = Image.fromarray(res*255).convert('RGB')

            p8 = im.convert("P")  # 将24位深的RGB图像转化为8位深的模式“P”图像
            p8.save(save_path+name)
            
    cost_time.pop(0)
    print('Mean running time is: ', np.mean(cost_time))
    print("FPS is: ", test_loader.size / np.sum(cost_time))
print("Test Done!")


