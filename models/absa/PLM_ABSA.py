
import torch, os
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.processor_utils import set_rng_seed
# config 中已经添加路径了
from data_loader import get_specific_dataset, ABSADataset

"""
| dataset     | rest  | lap   | twi   |

| baseline    | 83.42 | 79.36 | 00.00 |
| performance | 82.08 | 80.31 | 75.34 | -> deberta-base

"""
baselines = {
    'base': {'rest': 0.805, 'lap': 0.79, 'twi': 0.75}, 
    'large': {'rest': 0.84, 'lap': 0.82, 'twi': 0.76},
}

class DataLoader_PLM(Dataset):
    def __init__(self, dataset, d_type='single', desc='train') -> None:
        self.d_type = d_type
        self.dataset = dataset
        self.samples = dataset.datas['data'][desc]
        dataset.batch_cols = {
            'idx': -1,
            'input_ids': dataset.tokenizer.pad_token_id,
            'attention_mask': -1,
            'label': -1,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    

class ABSADataset_PLM(ABSADataset):
    def get_vector(self, only=None, is_dep=False):    
        tokenizer = self.tokenizer
        for desc, samples in self.datas['data'].items():
            if samples is None: continue
            if only is not None and desc!=only: continue
            self.info['class_category'][desc] = {l: 0 for l in self.tokenizer_['labels']['itol'].keys()}
            for sample in samples:
                embedding = tokenizer.encode_plus(sample['sentence'], sample['aspect'], return_tensors='pt')
                sample['input_ids'] = embedding['input_ids'].squeeze(dim=0)
                sample['attention_mask'] = embedding['attention_mask'].squeeze(dim=0)
                sample['token_type_ids'] = embedding['token_type_ids'].squeeze(dim=0)
                sample['label'] = self.tokenizer_['labels']['ltoi'][sample['polarity']]
                
                self.info['class_category'][desc][sample['label']] += 1

    def get_vector_(self, args=None, tokenizer=None, only=None):      
        self.args, self.tokenizer = args, tokenizer
        self.mask_token, self.mask_token_id = tokenizer.mask_token, tokenizer.mask_token_id
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                #embedding = tokenizer(item['text'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                query = f"The sentiment of {item['aspect']} is {self.mask_token}"
                embedding = tokenizer.encode_plus(item['sentence'], query, return_tensors='pt')
                item_embed = {
                    'index': item['index'],
                    'input_ids': embedding.input_ids.squeeze(dim=0),
                    'attention_mask': embedding.attention_mask.squeeze(dim=0),
                    'token_type_ids': embedding.token_type_ids.squeeze(dim=0),
                    'polarity': item['polarity'],
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed
    
    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs

def config_for_model(args, scale='base'):
    scale = args.model['scale']
    # if scale=='large': args.model['plm'] = args.file['plm_dir'] + 'bert-large'
    # else: args.model['plm'] = args.file['plm_dir'] + 'bert-base-uncased'
    args.model['plm'] = args.file['plm_dir'] + f'deberta-{scale}'

    args.model['baseline'] = baselines[scale][args.train['tasks'][1]]
    args.model['tokenizer'] = None
    args.model['optim_sched'] = ['AdamW', 'linear']
    return args

def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    set_rng_seed(args.train['seed'])
    
    ## 2. 导入数据
    data_path = args.file['data_dir'] + f"{args.train['tasks'][1]}/"
    if os.path.exists(f"{data_path}dataset.pt"):
        dataset = torch.load(f"{data_path}dataset.pt")
    else:
        dataset = ABSADataset_PLM(data_path, lower=True)
        dataset.tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
        dataset.get_vector()
        dataset.shuffle = {'train': True, 'valid': False, 'test': False}
        for desc, data in dataset.datas['data'].items():
            dataset.datas['data'][desc] = DataLoader_PLM(
                dataset,
                d_type='single',
                desc=desc
            )
        dataset.task = 'cls'
        torch.save(dataset, f"{data_path}dataset.pt")

    ## 3. 导入模型
    model = PLMForAbsa(
        args=args,
        dataset=dataset,
    )
    return model, dataset

class PoolerAll(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states # [:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

## 非pretrain 方式
class PLMForAbsa(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super(PLMForAbsa, self).__init__()
        self.args = args
        self.dataset = dataset
        self.n_class = dataset.n_class

        self.plm_model = AutoModel.from_pretrained(plm if plm is not None else args.model['plm'])
        self.plm_model.pooler = PoolerAll(self.plm_model.config)

        self.hidden_dim = self.plm_model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_dim, self.n_class)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.loss_ce = CrossEntropyLoss()
    
    def encode(self, inputs, method=['cls', 'asp', 'all']):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        plm_out = self.plm_model(
            input_ids, 
            attention_mask=attention_mask
            )
        hidden_states = self.plm_model.pooler(plm_out.last_hidden_state)
        hidden_states = self.dropout(hidden_states)

        if 'cls' in method: return hidden_states[:,0]
        if 'all' in method: return hidden_states
        if 'asp' in method:
            token_type_ids = inputs['token_type_ids']
            hidden_states_aspect = torch.mul(token_type_ids.unsqueeze(dim=-1), hidden_states)
            output['asp'] = torch.div(torch.sum(hidden_states_aspect, dim=1), torch.sum(token_type_ids, dim=-1).unsqueeze(dim=-1))

        return output

    def forward(self, inputs, stage='train'):
        outputs = self.encode(inputs, ['cls'])
        logits = self.classifier(outputs) 
        loss = self.loss_ce(logits, inputs['label'])
        
        return {
            'loss':   loss,
            'logits': logits,
            'preds':  torch.argmax(logits, dim=-1).cpu(),
            'labels': inputs['label'],
        }