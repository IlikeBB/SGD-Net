import numpy as np, torchvision, torch, os, glob
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import nibabel as nib
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.metrics import *
from utils.loss import DiceBCELoss
from utils.utils import clip_gradient
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import ndimage
from tqdm import tqdm   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# data augmentation
transform = A.Compose(
    [   
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=15, p=0.2),
        A.GridDistortion(p=0.3),
        ToTensorV2(),
    ]
)
transformv = A.Compose(
    [   
        ToTensorV2(),
    ]
)

class DataLoaderSegmentation(object):
    def __init__(self, base_path, transform=None):
        super(DataLoaderSegmentation, self).__init__()
        # path
        # get single or mini batch nii data
        self.data_image_path = sorted(glob.glob(os.path.join(base_path,'image','*.*')))
        self.data_masks_path = sorted(glob.glob(os.path.join(base_path,'masks','*.*')))
        self.transform = transform
    def __getitem__(self, index):
        img_path = self.data_image_path[index]
        mask_path = self.data_masks_path[index]
        image = self.__nii_load__(img_path)
        masks = self.__nii_load__(mask_path)

        if self.transform is not None: #image and masks must be transformed to numpy array
            transformed = self.transform(image=image.copy(), mask=masks.copy())
            image = transformed["image"]
            masks = transformed["mask"]
            return image, masks
        return torch.from_numpy(image.copy()), torch.from_numpy(masks.copy())

    def __len__(self):
        return len(self.data_image_path)

    def __nii_load__(self, nii_path):
        image = nib.load(nii_path)
        # print(nii_path)
        affine = image.header.get_best_affine()
        image = image.get_fdata()
        volume = np.float32(image.copy())
        if affine[1, 1] > 0:
            volume = ndimage.rotate(volume, 90, reshape=False, mode="nearest")
        if affine[1, 1] < 0:
            volume = ndimage.rotate(volume, -90, reshape=False, mode="nearest")
        if affine[1, 1] < 0:                 
            volume = np.fliplr(volume)
        return volume

class logs_realtime_reply:
    def __init__(self):
        self.avg_dice = 0.0
        self.avg_loss=np.inf
        self.avg_tn = 0
        self.avg_fp = 0
        self.avg_fn = 0
        self.running_metic = {"Loss":0, "dice_coef": 0,  "IoU":0, "TP":0, "FP":0, "FN": 0}
        self.end_epoch_metric = None
    def metric_stack(self, inputs, targets, loss):
        smooth = 1
        self.running_metic['Loss'] +=loss
        # dice
        inputs = torch.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = ((2.*intersection)/(inputs.sum() + targets.sum())).item()
        self.running_metic['dice_coef'] += round(dice, 5)
        # IoU
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = ((intersection + smooth)/(union + smooth)).item()
        self.running_metic['IoU']+=round(IoU, 5)
        # TP
        self.running_metic['TP'] += int((inputs * targets).sum().item())
        # FP
        self.running_metic['FP'] += int(((1-targets) * inputs).sum().item())
        # FN
        self.running_metic['FN'] += int((targets * (1-inputs)).sum().item())

    def mini_batch_reply(self, current_step, epoch, iter_len):
        avg_reply_metric = {"Loss":None, "dice_coef": None,  "IoU": None,"TP":None, "FP":None, "FN": None}
        for j in avg_reply_metric:
            if j in 'TP FP FN':
                avg_reply_metric[j] = int(self.running_metic[j]/int(current_step))
            else:
                avg_reply_metric[j] = round(self.running_metic[j]/int(current_step),5)
        
        if current_step ==iter_len:
            self.end_epoch_metric = avg_reply_metric
        return avg_reply_metric

    def epoch_reply(self):
        return self.end_epoch_metric

# model create
def model_create():
    model = smp.Unet(encoder_name=params['model'], encoder_weights=None, in_channels=1, classes=1)
    model.to(device)
    return model

# model train
def train(train_loader, model, criterion, optimizer, epoch):
    global mini_loss
    get_logs_reply = logs_realtime_reply()
    model.train()
    stream = tqdm(train_loader)
    size_rate = [0.75,1.00,1.25]
    for i, (images, target) in enumerate(stream, start=1):
        for rate in size_rate:
            images = images.to(device)
            target = target.to(device)
            trainsize = int(round(params['train_size']*rate/32)*32)
            if rate!=1.00:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                target = F.upsample(target, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # mdoel1
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, params['clip'])
            optimizer.step()
            # model2
            # model score pred 
            # y net arch: SGDNet + CGRD value data
            # multiclass 
            # loss =normalized(dwi classification(loss1 x w1) + t1 socre lstm(loss2 x w2))

            if rate==1.00:
                get_logs_reply.metric_stack(output, target, loss = round(loss.item(), 5))
                avg_reply_metric = get_logs_reply.mini_batch_reply(i, epoch, len(stream))
                avg_reply_metric['lr'] = optimizer.param_groups[0]['lr']
                stream.set_description(f"Epoch: {epoch}. Train. {str(avg_reply_metric)}")
    scheduler.step()
    for x in avg_reply_metric:
        # print(avg_reply_metric)
        writer.add_scalar(f'{x}/Train {x}', avg_reply_metric[x], epoch)
# model validate
def validate(valid_loader, model, criterion, epoch):
    global best_dice
    get_logs_reply2 = logs_realtime_reply()
    model.eval()
    stream_v = tqdm(valid_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream_v, start=1):
            images = images.to(device)
            target = target.to(device)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            get_logs_reply2.metric_stack(output, target, loss = round(loss.item(), 5))
            avg_reply_metric = get_logs_reply2.mini_batch_reply(i, epoch, len(stream_v))
            stream_v.set_description(f"Epoch: {epoch}. Valid. {str(avg_reply_metric)}")
        avg_reply_metric = get_logs_reply2.epoch_reply()
    for x in avg_reply_metric:
        if x =='dice_coef' and avg_reply_metric[x] > best_dice:
            best_dice = avg_reply_metric[x]
            current_loss = avg_reply_metric['Loss']
            save_ck_name = f'{ck_pth}/{project_name} --  epoch:{epoch} | vDice:{round(best_dice, 5)} | vLoss:{round(current_loss,5)}.pt'
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'loss':  current_loss,}, save_ck_name)
            print('save...', save_ck_name)
            best_ck_name = f'{ck_pth}/best-{project_name}.pt'
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'loss':  current_loss,}, best_ck_name)
            print('save...', best_ck_name)
        # print(avg_reply_metric)
        writer.add_scalar(f'{x}/Valida {x}', avg_reply_metric[x], epoch)

# model iteration process
def  train_valid_process_main(model, train_dataset, valid_dataset, batch_size):
    global mini_loss, best_dice
    mini_loss = np.inf
    best_dice = 0.0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, params["epochs"] + 1):
        # adjust_lr(optimizer, params['lr'], epoch, 0.1, 100)
        train(train_loader, model, loss, optimizer, epoch)
        validate(valid_loader, model, loss, epoch)
    return model

# model config
# dataset
train_dataset = DataLoaderSegmentation('./dataset/normalized_zscore/train/', transform=transform)
valid_dataset = DataLoaderSegmentation('./dataset/normalized_zscore/valid/', transform=transformv)
# model hyper parameter
params = {
    'architecture':'unet',
    "model": 'resnet18', #baseline = 'resnet18'
    "device": "cuda",
    "lr": 0.003, #baseline = 0.003
    "batch_size": 32, #baseline resnet18 : 32
    "epochs": 250,
    "train_size": 384,
    "clip":0.5,
    "adjust01": "optimizer weight decay - cosine decay",
    "adjust02": "gradient clipping",
    "adjust03" : "*multi scale training*"
}

# checkpoint setting
project_name = f"2DRes18Unet - lr_{params['lr']} - DCEL"
project_folder = f'2021.11.03.t1 - 2DRes18Unet'
ck_pth = f'./checkpoint/{project_folder}'
if os.path.exists(ck_pth)==False:
    os.mkdir(ck_pth)
ck_name = project_name

# write training setting txt
#txt record model config and adjust 
path = f'./checkpoint/{project_folder}/{project_name}.txt'
f = open(path, 'w')
lines = params
f.writelines([f'{i} : {params[i]} \n' for i in params])
f.close()
# tensorboard setting
tensorboard_logdir = f'./logsdir/ {project_folder} - {project_name}'
writer=SummaryWriter(tensorboard_logdir)



# model create
model = model_create()
# loss
loss = DiceBCELoss()
# optimizer
optimizer = Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max = 20)
logs  = train_valid_process_main(model, train_dataset, valid_dataset, params['batch_size'])

writer.close()