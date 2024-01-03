import torch
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from config import config
from run_alsc_plm import get_model
from utils.processor import Processor

class Processor_(Processor):
    def get_features(self, stage='test'):
        for bi, batch in enumerate(self.dataloader[stage]):
            self.model.eval()
            with torch.no_grad():
                outs = self.model_calculate(batch, stage) 
            
            for i, idx in enumerate(batch['idx']):
                self.dataset.datas['data'][stage].samples[idx]['fea'] = outs['cls'][i]
                self.dataset.datas['data'][stage].samples[idx]['pred'] = outs['preds'][i]

        return self.dataset.datas['data'][stage].samples


def tsne_plot(features, labels=None, preds=None, tsne_name='./tsne_ce'):
    tsne = TSNE(init='pca', random_state=2023)
    low_features = tsne.fit_transform(features)
    palette = sns.color_palette("bright", 10)
    
    fig = sns.scatterplot(
        x=low_features[:,0], 
        y=low_features[:,1], 
        hue=labels, 
        style=labels,
        #legend='full', 
        palette=palette,
        c=['blue' if label == pred else 'red' for label, pred in zip(labels, preds)]
    )

    diffs = [0 if label == pred else 1 for label, pred in zip(labels, preds)]
    error_features = low_features[np.array(diffs, dtype=bool)]
    error_labels = labels[np.array(diffs, dtype=bool)]
    fig = sns.scatterplot(
        x=error_features[:,0], 
        y=error_features[:,1], 
        style=error_labels, 
        c='blueviolet',
    )

    h, _ = fig.get_legend_handles_labels()
    fig.legend(h, ['Negative', 'Positive', 'Neutral'])
    fig.get_figure().savefig(f"{tsne_name}.png", dpi=800)


args = config(tasks=['absa','rest'], models=['seel_absa', 'plm'])
args.model['scale'] = 'base'

model, dataset = get_model(args)
dataset.metrics = ['macro_f1', 'accuracy']
dataset.lab_range = list(range(dataset.n_class))
processor = Processor_(args, model, dataset)

save_path = f"{args.file['save_dir']}/{args.model['name']}/"
for strategy in ['ce', 'scl', 'seel']:
    checkpoint = torch.load(save_path+f"{args.model['backbone']}_model_{strategy}.state")
    processor.model.load_state_dict(checkpoint['net'])

    samples = processor.get_features('test')
    features = [sample['fea'] for sample in samples]
    labels = [sample['label'] for sample in samples]
    preds = [sample['pred'] for sample in samples]
    plt.figure()
    tsne_plot(np.array(features), np.array(labels), np.array(preds), tsne_name=f"./tsne_test_{strategy}")

    print('k')