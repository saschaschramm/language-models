from utils import set_seed
set_seed(42)
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import AdditionDataset

if __name__ == '__main__':
    ndigit = 2
    train_dataset = AdditionDataset(ndigit=ndigit, split='train')
    test_dataset = AdditionDataset(ndigit=ndigit, split='test')
    num_train = len(train_dataset)
    num_test = len(test_dataset)
    print(f'num_train: {num_train}, num_test: {num_test}')

    # initialize a baby GPT model
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, 
                    n_layer=2, n_head=4, n_embd=128)
    model = GPT(mconf)
    batch_size = 512

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=200, batch_size=batch_size, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=1024, final_tokens=50*len(train_dataset)*(ndigit+1),
                        num_workers=0)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()


