from torch.utils.data.dataloader import DataLoader
from utils import sample
from utils import set_seed
set_seed(42)
from main_train import AdditionDataset
import torch
import numpy as np
from plotter import Plotter
from dataset import AdditionDataset

ndigit = 2
train_dataset = AdditionDataset(ndigit=ndigit, split='train')
test_dataset = AdditionDataset(ndigit=ndigit, split='test')
batch_size = 1
loader = DataLoader(test_dataset, batch_size=batch_size)
model = torch.load('model')

def test_terms():
    terms = []
    for i in range(len(test_dataset)):
        x, _ = test_dataset[i]
        terms.append(x.tolist()[0:4])
    return terms


if __name__ == '__main__':
    term1_space = np.arange(0, 50)
    term2_space = np.arange(0, 50)

    def create_mesh(feature_0, feature_1):
        x, y = np.meshgrid(feature_0, feature_1)
        return np.vstack([x.ravel(), y.ravel()]).T

    mesh = create_mesh(term1_space, term2_space)
    num_true = 0
    num_all = 0
    probs = []

    labels = []
    test_data_terms = test_terms()

    for term_1, term_2 in mesh:
        term_1_str = f'{term_1:0{ndigit}d}'
        term_2_str = f'{term_2:0{ndigit}d}'
        terms = [int(term_1_str[0]), int(term_1_str[1]), int(term_2_str[0]), int(term_2_str[1])]
        d1d2 = torch.tensor([terms])
        factors = torch.tensor([[10,  1]])
        d1i = (d1d2[:,:ndigit] * factors).sum(1)
        d2i = (d1d2[:,ndigit:ndigit*2] * factors).sum(1)
        d3i_gt = d1i + d2i

        # if term in test_data then color mask pixel otherwise transparent
        if terms in test_data_terms:
            labels.append(1)
        else:
            labels.append(np.nan)

        d1d2d3, step_probs = sample(model, d1d2, ndigit)

        # Get the probability of the second digit being 9
        label = 9
        prediction_index = 1
        prob = step_probs[prediction_index][0][label]
        probs.append(prob)
        d3 = d1d2d3[:, -(ndigit):]
        d3i_pred = (d3 * factors).sum(1)
    
        if terms in test_data_terms:
            num_true += (d3i_gt == d3i_pred).sum().item()
            num_all += 1
    print(f'num_true {num_true}')
    print(f'num_all {num_all}')
    print(f'accuracy {num_true/num_all:.4f}')

    plotter = Plotter()
    plotter.plot_relative_frequency(labels,
                                    term1_space, term2_space, probs,
                                    filename='neural-network.png',
                                    x_name='term 1',
                                    y_name='term 2',
                                    mask=False
                                    )
    plotter.plot_relative_frequency(labels,
                                    term1_space, term2_space, probs,
                                    filename='neural-network-masked.png',
                                    x_name='term 1',
                                    y_name='term 2',
                                    mask=True
                                    )