import torch, os, copy, fitlog, math, json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from utils.processor_utils import set_rng_seed
from models.img.resnet_basic import resnet50

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.datas = {
            'data': None,
            'loader': None,
        }

    def get_dataloader(self, batch_size, shuffle=None, collate_fn=None, only=None, split=None):
        if shuffle is None: shuffle = {'train': True, 'valid': False, 'test': False}
        if collate_fn is None: collate_fn = self.collate_fn

        dataloader = {}
        for desc, data_embed in self.datas['data'].items():
            if only is not None and desc!=only: continue
            dataloader[desc] = DataLoader(dataset=data_embed, batch_size=batch_size, shuffle=shuffle[desc], collate_fn=collate_fn, num_workers=8)
            
        if only is None and 'valid' not in dataloader: dataloader['valid'] = dataloader['test']
        return dataloader

    def collate_fn(self, batch):
        return {
            'image': torch.stack([item[0] for item in batch]),
            'label': torch.tensor([item[1] for item in batch]),
        }

    def __len__(self):
        return len(self.datas['data']['train'])


def config_for_model(args):
    # args.model['optim_sched'] = ['AdamW_', 'cosine']
    # args.model['optim_sched'] = ['AdamW_', 'linear']
    args.model['optim_sched'] = ['SGD', None]

    return args


def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args)  # 添加模型参数, 获取任务数据集
    set_rng_seed(args.train['seed'])

    ## 2. 导入数据
    split = 0.2
    data_dir = args.file['data_dir'] + f"{args.train['tasks'][1]}/"
    data_path = data_dir + f"dataset_{split}.pt"
    if os.path.exists(data_path):
        dataset = torch.load(data_path)
    else:
        dataset = MyDataset()
        from data_loader import get_specific_dataset
        datas = get_specific_dataset(args, data_name=args.train['tasks'][-1])
        dataset.datas['data'] = {'train': datas['train'], 'test': datas['test']}

        dataset.name  =args.train['tasks'][-1]
        dataset.task, dataset.n_class = 'cls', datas['n_class']
        if split:
            half_size = int(len(datas['train'])*split) # 0.2 0.4 0.6 0.8 1.0
            datas['train'], _ = random_split(datas['train'], [half_size, len(datas['train']) - half_size])

        torch.save(dataset, data_path)

    ## 3. 导入模型
    model = _ResNet(args, dataset)
    return model, dataset


class _ResNet(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super().__init__()
        self.args = args
        self.dataset = dataset

        resnet = resnet50()
        hidden_dim = 2048

        self.plm_model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.hidden_dim = hidden_dim
        self.classifier = nn.Linear(self.hidden_dim, dataset.n_class)
        # self.dropout = nn.Dropout(args.model['drop_rate'])
        self.dropout = None
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)
        self.loss_bce = nn.BCELoss()

        # self.conv2d = nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        # self.activate = nn.Sigmoid()
        # self.loss_bce = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

    def encode(self, inputs, method=['cls', 'asp', 'all']):
        output = self.plm_model(inputs['image']).reshape(-1, self.hidden_dim)
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
