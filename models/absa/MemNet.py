
import torch, os
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from utils.attention import Attention
from utils.processor_utils import set_rng_seed
from data_loader import get_specific_dataset, ABSADataset

"""
| dataset     | rest  | lap   | twi   |

| baseline    | 65.88 | 63.28 | 66.17 |
| performance | 71.29 | 68.90 | 70.59 |

"""
baselines = {'rest': 0.70, 'lap': 0.675, 'twi': 0.69} # √

class ABSADataset_Glove(ABSADataset):
    def get_vector(self, only=None, is_dep=False):
        def get_adjs(seq_len, sample_embed, self_token_id, directed=True, loop=False):
            # head -> adj (label: dep)
            idx_asp = [idx for idx, val in enumerate(sample_embed['attention_mask_asp']) if val]
            heads, deps = sample_embed['dep_heads'], sample_embed['dep_deprels']
            adjs = np.zeros((seq_len, seq_len), dtype=np.float32)
            edges = np.zeros((seq_len, seq_len), dtype=np.int64)
            # head 指向 idx：第i个token的父节点是第i个head对应位置的节点
            for idx, head in enumerate(heads):
                if idx in idx_asp: # 是 aspect
                    for k in idx_asp: 
                        adjs[idx, k], edges[idx, k] = 1, self_token_id
                        adjs[k, idx], edges[k, idx] = 1, self_token_id
                if head != -1: # non root
                    adjs[head, idx], edges[head, idx] = 1, deps[idx]
                if not directed: # 无向图
                    adjs[idx, head], edges[idx, head] = 1, deps[idx]
                if loop: # 自身与自身相连idx
                    adjs[idx, idx], edges[idx, idx] = 1, self_token_id
            
            sample_embed['dep_graph_adjs'] = torch.tensor(adjs)
            sample_embed['dep_graph_edges'] = torch.tensor(edges)
            return sample_embed

        def get_dependency(sample, sample_embed, tokenizers):
            sample_embed['dep_heads'] = torch.tensor(sample['heads'])
            for desc in ['deprels', 'tags']:
                temp, vocab, unk_token = sample[desc], tokenizers[desc].vocab, tokenizers[desc].unk_token
                temp_id = [vocab[unk_token] if item not in vocab else vocab[item] for item in temp]
                sample_embed['dep_'+desc] = torch.tensor(temp_id)
            
            dep_self_token_id = tokenizers['deprels'].vocab[tokenizers['deprels'].self_token]
            sample_embed = get_adjs(len(sample_embed['dep_heads']), sample_embed, dep_self_token_id)
            return sample_embed
        
        def get_embedding(sample):
            tokens, asp_poses = sample['tokens'], sample['aspect_pos']
            tokens_asp = tokens[asp_poses[0]:asp_poses[1]]
            embedding, embedding_asp = self.tokenizer.encode(tokens), self.tokenizer.encode(tokens_asp)
            assert all(embedding['input_ids'][asp_poses[0]:asp_poses[1]]==embedding_asp['input_ids'])
            
            mask_asp, dis_asp = torch.zeros_like(embedding['input_ids']), {'left':[], 'mid': [], 'right':[]}
            for i, idx in enumerate(embedding['input_ids']):
                if i < asp_poses[0]: # aspect 左边
                    dis_asp['left'].extend([i-asp_poses[0]]) # *len(idx))
                elif i >= asp_poses[1]: # aspect 右边
                    dis_asp['right'].extend([i+1-asp_poses[1]]) # *len(idx))
                else: # aspect 中间
                    dis_asp['mid'].extend([0]) # *len(idx))
                    mask_asp[i] = 1
            
            distance_asp = dis_asp['left']+dis_asp['mid']+dis_asp['right']
            return {
                'input_ids': embedding['input_ids'],
                'attention_mask': embedding['attention_mask'],
                'attention_mask_asp': mask_asp,
                'asp_dis_ids': torch.tensor(distance_asp),
            }
                
        for stage, samples in self.datas['data'].items():
            if samples is None: continue
            if only is not None and stage!=only: continue
            for sample in samples:
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], return_tensors='pt')
                embedding = get_embedding(sample)
                sample['input_ids'] = embedding['input_ids']
                sample['attention_mask'] = embedding['attention_mask']
                sample['attention_mask_asp'] = embedding['attention_mask_asp']
                sample['asp_dis_ids'] = embedding['asp_dis_ids']
                sample['label'] = self.tokenizer_['labels']['ltoi'][sample['polarity']]

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs

def config_for_model(args):
    args.model['hops'] = 3
    args.model['plm'] = None
    # args.model['baseline'] = baselines[scale][args.train['tasks'][1]]

    args.model['tokenizer'] = args.file['data_dir'] + args.train['tasks'][-1] + '/glove.tokenizer'
    args.model['optim_sched'] = ['SGD', 'linear']
    # args.model['optim_sched'] = ['SGD', None]

    args.model['initial'] = False
    
    return args

def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    set_rng_seed(args.train['seed'])
    
    ## 2. 导入数据
    data_path = args.file['data_dir'] + f"{args.train['tasks'][1]}/"
    if os.path.exists(f"{data_path}dataset-{args.model['backbone']}.pt"):
        dataset = torch.load(f"{data_path}dataset-{args.model['backbone']}.pt")
    else:
        dataset = ABSADataset_Glove(data_path, lower=True)
        from utils.tokenizer_glove import get_tokenizer
        dataset.tokenizer = get_tokenizer(args.model['tokenizer'], dataset)
        dataset.get_vector()
        dataset.task = 'cls'
        dataset.batch_cols = {
            'idx': -1,
            'input_ids': 0,
            'attention_mask': 0,
            'attention_mask_asp': 0,
            'label': -1,
        }
        torch.save(dataset, f"{data_path}dataset-{args.model['backbone']}.pt")

    ## 3. 导入模型
    model = MemNet(
        args=args,
        dataset=dataset,
    )
    return model, dataset

## pretrain 方式
class MemNet(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super(MemNet, self).__init__()
        self.args = args
        self.dataset = dataset
        self.n_class = dataset.n_class

        ## dataset related
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(
                dataset.tokenizer.word_embedding, 
                dtype=torch.float, 
                requires_grad=True)
            )
        self.hidden_dim = dataset.tokenizer.embed_dim
        self.plm_model = None

        self.attention = Attention(self.hidden_dim, score_function='mlp')
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.classifier = nn.Linear(self.hidden_dim, self.n_class)
        self.loss_ce = CrossEntropyLoss()

    def encode(self, inputs, method=['cls', 'asp', 'all']):
        mask, mask_asp = inputs['attention_mask'], inputs['attention_mask_asp']
        input_ids, mask_cxt = inputs['input_ids'], mask-mask_asp

        sentence_embedding = self.embedding(input_ids) # 这个更好
        # sentence_embedding = self.dropout(self.embedding(input_ids))
        mask_asp_extend = mask_asp.unsqueeze(dim=-1).repeat(1, 1, self.hidden_dim)
        aspect_embedding = torch.div(torch.sum(sentence_embedding*mask_asp_extend, dim=1), torch.sum(mask_asp, dim=-1).unsqueeze(dim=-1))

        x = aspect_embedding.unsqueeze(dim=1)
        for _ in range(self.args.model['hops']):
            x = self.dropout(self.linear(x))
            out_at, _ = self.attention(sentence_embedding, x, mask=mask_cxt)
            x = out_at + x
        x = x.view(x.size(0), -1)

        output = {'idx': inputs['idx'], 'lab': inputs['label']}
        if 'cls' in method: return x
        if 'all' in method: output['all'] = x
        if 'asp' in method: output['asp'] = aspect_embedding

        return output

    def forward(self, inputs, stage='train'):
        features = self.encode(inputs)
        logits = self.classifier(features) 
        loss = self.loss_ce(logits, inputs['label'])
        
        return {
            'loss':   loss,
            'logits': logits,
            'preds':  torch.argmax(logits, dim=-1).cpu(),
            'labels': inputs['label'],
        }