import random, torch, os
from collections import defaultdict

## tokenizer
def iterSupport(func, query):
    # 迭代处理 list 数据
    if isinstance(query, (list, tuple)):
        return [iterSupport(func, q) for q in query]
    try:
        return func(query)
    except TypeError:
        return func[query]

class Embedding(object):
    def __init__(self, inputs) -> None:
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.attention_mask_asp = inputs['attention_mask_asp']

class Tokenizer(object):
    def __init__(self, embed_dim=None, lower=True, is_full=True):
        self.lower = lower
        self.embed_dim = embed_dim
        self.words = {}
        if is_full:
            self.pad_token  = '<pad>' # '[PAD]'
            self.unk_token  = '<unk>' # '[UNK]'
            self.mask_token = '<mask>' # '[MASK]'
            self.vocab = {self.pad_token: 0, self.unk_token: 1, self.mask_token: 2}
            self.ids_to_tokens = {0: self.pad_token, 1: self.unk_token, 2: self.mask_token}
        else:
            self.vocab = {}
            self.ids_to_tokens = {}
    
    def count(self, ele):
        if self.lower: ele = ele.lower()
        if ele not in self.words: self.words[ele] = 0
        self.words[ele] += 1

    def add_tokens(self, tokens):
        if not isinstance(tokens, list): tokens = [tokens]
        for token in tokens:
            if self.lower: token = token.lower()
            if token not in self.vocab: 
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[len(self.ids_to_tokens)] = token
        self.vocab_size = len(self.vocab)

    def get_vocab(self, min_count=1):
        for ele, count in self.words.items():
            if count > min_count:
                self.vocab[ele] = len(self.vocab)
                self.ids_to_tokens[len(self.ids_to_tokens)] = ele
        self.vocab_size = len(self.vocab)

    def get_word_embedding(self):
        glove_file = "/home/jzq/My_Codes/Pretrained_Datasets/glove/glove.840B.300d.txt"
        words_embed = defaultdict(list)

        with open(glove_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fr:
            for line in fr:
                tokens = line.rstrip().split()
                word = ''.join(tokens[:-self.embed_dim])
                if word in self.vocab:
                    words_embed[word] = [float(ele) for ele in tokens[-self.embed_dim:]]

        self.word_embedding = [[float(ele) for ele in words_embed[key]] if key in words_embed else [0]*self.embed_dim if key == self.pad_token else [random.uniform(0, 1) for _ in range(self.embed_dim)] for key in self.vocab]

    def get_sent_embedding(self, sentence, method='mean'):
        # 输入一句话，输出一个语义向量
        idxs = self.tokens_to_ids(sentence.split(' '))
        embedding = torch.tensor([self.word_embedding[idx] for idx in idxs])
        if method == 'mean':
            return embedding.mean(dim=0)

    def tokens_to_ids(self, tokens):
        # 输入tokens，输出句子的 id 表示
        return torch.tensor([self.vocab[token] if token in self.vocab else self.vocab[self.unk_token] for token in tokens])

    def encode(self, words, return_tensors='pt', add_special_tokens=False):
        if not isinstance(words, list): words = [words]
        input_ids = []
        for word in words:
            if word in self.vocab: input_ids.append(self.vocab[word])
            else: input_ids.append(self.vocab[self.unk_token])
        
        input_ids = torch.tensor(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
        }

    def encode_(self, sample, max_length=None, return_tensors='pt', add_special_tokens=False):
        tokens_snt, tokens_asp = sample['sentence'].split(' '), sample['aspect'].split(' ')
        attention_mask = torch.tensor([1] * len(tokens_snt))
        input_ids = self.tokens_to_ids(tokens_snt)

        # 定位aspect位置
        attention_mask_asp = torch.zeros_like(attention_mask)
        char_start, char_end, char_point = sample['asp_pos'][0], sample['asp_pos'][1], 0
        for i, token in enumerate(tokens_snt):
            if char_point >= char_start and char_point <= char_end:
                if token in tokens_asp:
                    attention_mask_asp[i] = 1
            char_point += len(token)+1

        assert all(input_ids[attention_mask_asp.type(torch.bool)] == self.tokens_to_ids(tokens_asp))

        if max_length is not None: # 需要截断
            pass

        return Embedding({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_asp': attention_mask_asp,
        })

def get_tokenizer(path, dataset):
    ## 保存tokenizer
    if os.path.exists(path):
        tokenizer = torch.load(path)
    else:
        tokenizer = Tokenizer(embed_dim=300, lower=False)
        tokens = [item['tokens'] for item in dataset.datas['data']['train']]
        tokens.extend([item['sentence'].split(' ') for item in dataset.datas['data']['test']]) # 性能好点儿
        iterSupport(tokenizer.count, tokens) # 统计词频
        tokenizer.get_vocab(min_count=0) # 获得指向
        tokenizer.get_word_embedding() # 获取embedding
        torch.save(tokenizer, path)
        
    return tokenizer