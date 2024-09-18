import torch
import numpy as np
import pdb, os, argparse
from datetime import datetime
import sys
from tqdm import tqdm
from setting.dataLoader import get_loader
from setting.utils import clip_gradient, adjust_lr
from network.SimSOD import SimSOD
from network.MAE import mae_vit_tiny, mae_vit_small, mae_vit_base
import pytorch_iou
from metrics.SOD_metrics import SODMetrics
from torch.utils.tensorboard import SummaryWriter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument("--trainset_path", type=str, 
        default='/opt/data/private/FYX/Dataset/EORSSD/Train/')
parser.add_argument("--testset_path", type=str, 
        default='/opt/data/private/FYX/Dataset/EORSSD/Test/')
parser.add_argument("--dataset", type=str, default='EORSSD', 
                    help='Name of dataset:[EORSSD, ORSSD, ORSI4199]')

parser.add_argument("--mae", type=str, default='mae_vit_base',
        help='model name:[mae_vit_tiny, mae_vit_small, mae_vit_base]')
parser.add_argument('--model', type=str, default='SimSOD', 
                    help='model name:[SimSOD]')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument("--pretrained", type=str,
            default='', 
            help="restore from checkpoint")

parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument("--n_cpu", type=int, default=8, help="num of workers")
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=10000, help='every n epochs decay learning rate')
parser.add_argument('--save_path', type=str, default='./CHKP_MMPT/', help='')
parser.add_argument('--log_dir', type=str, default='./logs/', help='')
parser.add_argument('--save_ep', type=int, default=5, help='')
opt = parser.parse_args()



CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)


if __name__=='__main__':
    # build models
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    opt.log_dir=opt.log_dir+'{}_{}_ep{}'.format(opt.model, opt.dataset, opt.epoch)
    tb_writer = SummaryWriter(opt.log_dir)

    print(opt)
    mae = eval(opt.mae)(img_size=opt.trainsize)
    model = eval(opt.model)()

    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    metrics=SODMetrics(cuda=True)
    # load data
    train_loader, train_num = get_loader(opt.trainset_path+'Images/',
                                        opt.trainset_path+'Depth/', 
                                        opt.trainset_path+'Masks/', 
                                        opt.batchsize, opt.trainsize, num_workers=opt.n_cpu)
  
    print(f'Loading data, including {train_num} training images .')


    if opt.pretrained is not None and os.path.isfile(opt.pretrained):
        checkpoint = torch.load(opt.pretrained, map_location=torch.device('cpu'))
        pretrained_epochs=checkpoint['epoch']
        print('pretrained_epochs:',pretrained_epochs)
        try:
            mae.load_state_dict(checkpoint['model'])
            print('try: mae load pth from:', opt.pretrained)
        except:
            model_dict      = mae.state_dict()
            pretrained_dict = checkpoint['model']
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                # if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v) and k.split('.')[0]=='layers':
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    print(k)
                    temp_dict[k] = v
                    load_key.append(k)
                # else:
                #     no_load_key.append(k)
            model_dict.update(temp_dict)
            mae.load_state_dict(model_dict)

            print('except: mae load pth from:', opt.pretrained)
        mae=mae.to(device)

    print("Start Pretraining!")

    mae.eval()
    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        model.train()
        running_loss = 0.0
        data_loader = tqdm(train_loader, file=sys.stdout)
        steps=0
        for i, (images, depths, gts) in enumerate(data_loader, start=1):
            steps+=1
            optimizer.zero_grad()
            images = images.cuda()
            depths = depths.cuda()
            gts = gts.cuda()

            loss, y, mask= mae(imgs=images, mask_ratio=opt.mask_ratio)
            x=images
            y = mae.unpatchify(y)
            # visualize the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = mae.unpatchify(mask)  # 1 is removing, 0 is keeping
            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask

            sal, sal_sig = model(im_paste, depths)
            loss = CE(sal, gts) + IOU(sal_sig, gts)

            running_loss += loss.data.item()
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, Learning Rate: {}, loss={:.4f}".format(epoch, opt.epoch,
                                opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), running_loss / i)


        tags = ["train_loss", "learning_rate"]

        tb_writer.add_scalar(tags[0], running_loss/steps, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)

        if (epoch+1) % opt.save_ep == 0:
            torch.save(model.state_dict(), opt.save_path+'lastest_{}_{}_mmpt.pth'.format(opt.model, opt.dataset))
        
       