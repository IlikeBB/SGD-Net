import torch, random, os, multiprocessing
import numpy as np, pandas as pd, nibabel as nib 
import torch.backends.cudnn as cudnn
import torchio as tio
# multiprocess cpu 
from sklearn.metrics import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
from scipy import ndimage
from utils.S1_utils import clip_gradient
from utils.loss import FocalLoss, FocalTverskyLoss
from utils.model_res import generate_model
num_workers = multiprocessing.cpu_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
random.seed(1234)
torch.manual_seed(1234)
# load csv label
if True:
    csv_path = './20211104_label_1-350.csv'
    table =  pd.read_csv(csv_path)
    table_3t = table[table['1/0: 3T/1.5T MRI']==1.0]
    table_3t_train = np.array(table_3t[table_3t['Valid data']!='V'])
    table_3t_valid = table_3t[table_3t['Valid data']=='V']
    table_3t_valid = np.array(table_3t_valid[table_3t_valid['排除']!='Test data'])
    nii_3t_train = sorted([i for i in os.listdir(os.path.join('./dataset/S2_data/','train'))])
    nii_3t_valid = sorted([i for i in os.listdir(os.path.join('./dataset/S2_data/','valid'))])

# Subject Function Building
def tio_process(nii_3t_, table_3t_, basepath_='./dataset/S2_data/train/'):
    subjects_ = []
    for  (nii_path, nii_table) in zip(nii_3t_ , table_3t_):
        if (params['type']=='ap') and (nii_table[3]=='A' or nii_table[3]=='P'):
            subject = tio.Subject(
                dwi = tio.ScalarImage(os.path.join(basepath_, nii_path)), 
                ap = nii_table[3], 
                score=[])
            subjects_.append(subject)
        elif (params['type']=='nl'):
            subject = tio.Subject(
                dwi = tio.ScalarImage(os.path.join(basepath_, nii_path)), 
                nl  = nii_table[4], 
                score=[])
            subjects_.append(subject)
    return subjects_

class logs_realtime_reply:
    def __init__(self):
        self.avg_dice = 0.0
        self.avg_loss=np.inf
        self.avg_tn = 0
        self.avg_fp = 0
        self.avg_fn = 0
        # self.running_metic = {"Loss":0, "TP":0, "FP":0, "FN": 0, "Spec": 0, "Sens": 0}
        self.running_metic = {"Loss":0, "Accuracy":0, "Spec": 0, "Sens": 0}
        self.end_epoch_metric = None
    def metric_stack(self, inputs, targets, loss):
        self.running_metic['Loss'] +=loss
        # metric setting
        _, SR = torch.max(inputs, 1)
        GT = targets
        TP = int((SR * GT).sum()) #TP
        FN = int((GT * (1-SR)).sum()) #FN
        TN = int(((1-GT) * (1-SR)).sum()) #TN
        FP = int(((1-GT) * SR).sum()) #FP
        self.running_metic['Accuracy'] += round((TP + TN)/(TP + TN + FP + FN), 5)*100
        self.running_metic['Sens'] += round(float(TP)/(float(TP+FN) + 1e-6), 5)
        self.running_metic['Spec'] += round(float(TN)/(float(TN+FP) + 1e-6), 5)

    def mini_batch_reply(self, current_step, epoch, iter_len):
        # avg_reply_metric = {"Loss":None, "TP":None, "FP":None, "FN": None, "Spec": None, "Sens": None}
        avg_reply_metric = {"Loss":None, "Accuracy": None,"Spec": None, "Sens": None}
        for j in avg_reply_metric:
            avg_reply_metric[j] = round(self.running_metic[j]/int(current_step),5)
        
        if current_step ==iter_len:
            self.end_epoch_metric = avg_reply_metric
        return avg_reply_metric

    def epoch_reply(self):
        return self.end_epoch_metric

def model_create(depth=18):
    model = generate_model(model_depth=depth, n_input_channels=1, n_classes=2)
    model.to(device)
    return model

def label2value(label):
    if params['type']=='nl':
        target = [0 if i=='N' else 1 for i in label]
    else:
        target = [0 if i=='A' else 1 for i in label]
    return torch.LongTensor(target).to(device)

# model train
def train(train_loader, model, criterion, optimizer, epoch):
    get_logs_reply = logs_realtime_reply()
    model.train()
    stream = tqdm(train_loader)
   
    for i, data in enumerate(stream, start=1):
        images = data['dwi'][tio.DATA].to(device)
        target = label2value(data[params['type']])
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, params['clip'])
        optimizer.step()
        
        get_logs_reply.metric_stack(output, target, loss = round(loss.item(), 5))
        avg_reply_metric = get_logs_reply.mini_batch_reply(i, epoch, len(stream))
        avg_reply_metric['lr'] = optimizer.param_groups[0]['lr']
        stream.set_description(f"Epoch: {epoch}. Train. {str(avg_reply_metric)}")
    if epoch>params['scheduler_epoch']:
        scheduler.step()
    for x in avg_reply_metric:
        # print(avg_reply_metric)
        writer.add_scalar(f'{x}/Train {x}', avg_reply_metric[x], epoch)
# model validate
def validate(valid_loader, model, criterion, epoch):
    global best_vloss, best_vacc
    get_logs_reply2 = logs_realtime_reply()
    model.eval()
    stream_v = tqdm(valid_loader)
    with torch.no_grad():
        for i, data in enumerate(stream_v, start=1):
            images = data['dwi'][tio.DATA].to(device)
            target = label2value(data[params['type']])
            images = images.to(device)
            target = target.to(device)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            get_logs_reply2.metric_stack(output, target, loss = round(loss.item(), 5))
            avg_reply_metric = get_logs_reply2.mini_batch_reply(i, epoch, len(stream_v))
            stream_v.set_description(f"Epoch: {epoch}. Valid. {str(avg_reply_metric)}")
        avg_reply_metric = get_logs_reply2.epoch_reply()

    for x in avg_reply_metric:
        if x =='Accuracy' and avg_reply_metric[x] > best_vacc:
            best_vacc = avg_reply_metric[x]
            current_loss = avg_reply_metric['Loss']
            save_ck_name = f"{ck_pth}/{project_name} --  epoch:{epoch} | vLoss:{round(current_loss,5)} | vAcc:{round(avg_reply_metric['Accuracy'], 5)}.pt"
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'loss':  current_loss,}, save_ck_name)
            print('save...', save_ck_name)
            best_ck_name = f'{ck_pth}/best - {project_name}.pt'
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'loss':  current_loss,}, best_ck_name)
            print('save...', best_ck_name)
        # print(avg_reply_metric)
        writer.add_scalar(f'{x}/Valida {x}', avg_reply_metric[x], epoch)
# X
def  train_valid_process_main(model, training_set, validation_set, batch_size):
    global best_vloss, best_vacc
    best_vloss = np.inf
    best_vacc = 0.00
    # Subject Dataloader Building
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers)

    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, loss, optimizer, epoch)
        validate(valid_loader, model, loss, epoch)
    return model

if True: #model record
    params = {
        "type": "ap",
        "model": '3dresnet', #baseline = 'resnet18'
        "model_depth": 18,
        "device": "cuda",
        "opt": "Adam",
        "lr": 0.001, #baseline = 0.003
        "scheduler_epoch": 0 , #nl: 5, ap: None
        "batch_size": 8, #baseline resnet18 : 8
        "epochs": 200,
        "clip":0.5,
        # "adjust01": "CosineAnnealingLR, loss funciton -> focal tversky loss:[0.5, 0.5, 1.00, 1e-6]",
        "adjust01": "CosineAnnealingWarmRestarts, loss funciton -> focal loss:[0.8,2]",
        "adjust02": "learning rate 0.003 -> 0.001",
        "augment": "RandomFlip['AP'], RandomElasticDeformation",
        }

if True: #data augmentation, dataloader, 
    training_subjects = tio_process(nii_3t_train, table_3t_train, basepath_ = './dataset/S2_data/train/')
    validation_subjects = tio_process(nii_3t_valid, table_3t_valid, basepath_ = './dataset/S2_data/valid/')
    print('Training set:', len(training_subjects), 'subjects   ', '||   Validation set:', len(validation_subjects), 'subjects')
    # Transform edit
    training_transform = tio.Compose([
        # tio.HistogramStandardization({'mri': landmarks}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.OneOf({
            tio.RandomElasticDeformation(): 0.2,
            tio.RandomFlip(axes=('AP',), flip_probability=0.5): 1.0,
            # tio.RandomAffine(degrees=15, scales=(1.0, 1.0)): 0.3,
        }),
    ])
    validation_transform = tio.Compose([
        # tio.HistogramStandardization({'mri': landmarks}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])
    training_set = tio.SubjectsDataset(training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(validation_subjects, transform=validation_transform)

    # checkpoint setting
    project_name = f"{params['type']} - {params['model']}{params['model_depth']} - lr_{params['lr']} - CE"
    project_folder = f"2021.11.11.t6 - 3DResNet18 - {params['type']}"
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
    tensorboard_logdir = f'./logsdir/S2/ {project_folder} - {project_name}'
    writer=SummaryWriter(tensorboard_logdir)

if True: #model edit area
    # model create
    model = model_create(depth=params['model_depth'])
    # loss
    # loss = torch.nn.CrossEntropyLoss()
    loss = FocalLoss()
    # optimizer
    if params['opt']=='Adam':
        optimizer = Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = CosineAnnealingLR(optimizer, T_max = 20)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, last_epoch=-1)
    logs  = train_valid_process_main(model, training_set, validation_set, params['batch_size'])

writer.close()