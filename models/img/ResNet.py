import torch, os, copy, math, json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from processor_utils import set_rng_seed
class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.datas = {
            'data': None,
            'loader': None,
        }

    def collate_fn(self, batch):
        return {
            'image': torch.stack([item[0] for item in batch]),
            'label': torch.tensor([item[1] for item in batch]),
        }


def config_for_model(args):
    # args.model['optim_sched'] = ['AdamW_', 'cosine']
    #args.model['optim_sched'] = ['AdamW_', 'linear']
    args.model['optim_sched'] = ['SGD', 'linear']

    return args
             
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    set_rng_seed(args.train['seed'])
    
    ## 2. 导入数据
    dataset = MyDataset()
    if args.train['tasks'][-1] == 'cifar10':
        from datasets.img.data_loader import get_specific_dataset
        datas = get_specific_dataset(args, data_name='cifar10')
        dataset.datas['data'] = {'train': datas['train'], 'test': datas['test']}
        dataset.datas['loader'] = {
            'train': DataLoader(datas['train'], batch_size=args.train['batch_size'], shuffle=True, collate_fn=dataset.collate_fn),
            'valid': DataLoader(datas['test'], batch_size=args.train['batch_size'], shuffle=False, collate_fn=dataset.collate_fn),
            'test': DataLoader(datas['test'], batch_size=args.train['batch_size'], shuffle=False, collate_fn=dataset.collate_fn),
        }
        dataset.name = 'cifar10'
        dataset.task = 'cls'
        dataset.metrics = ['accuracy']
        dataset.n_class = datas['n_class']
        args.train['do_test'] = False

    ## 3. 导入模型
    model = _ResNet(args, dataset)
    return model, dataset

class _ResNet(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.plm_model = models.resnet50(pretrained=True)
        self.hidden_dim = 1000
        self.classifier = nn.Linear(self.hidden_dim, dataset.n_class)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)
        self.loss_bce = nn.BCELoss()

        # self.conv2d = nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        # self.activate = nn.Sigmoid()
        # self.loss_bce = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    def encode(self, inputs, method=['cls', 'asp', 'all']):
        output = self.plm_model(inputs['image'])
        return output

    def forward(self, inputs, stage='train'):
        output = self.plm_model(inputs['image'])
        logits = self.classifier(output)
        loss = self.loss_ce(logits, inputs['label'])

        return {
            'logits': logits,
            'loss': loss,
            'preds': torch.argmax(logits, dim=-1).cpu(),
            'labels': inputs['label'],
        }