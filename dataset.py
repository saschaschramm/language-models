import numpy as np
import torch
from torch.utils.data import Dataset


def generate_permutations():
    def create_mesh(x, y):
        x, y = np.meshgrid(x, y)
        return np.vstack([x.ravel(), y.ravel()]).T

    terms_1 = np.linspace(0, 49, 50).astype(int)
    terms_2 = np.linspace(0, 49, 50).astype(int)
    mesh = create_mesh(terms_1, terms_2)
    perm = np.asarray([int(f"{element[0]:02d}{element[1]:02d}") for element in mesh])
    np.random.shuffle(perm)
    return perm


class AdditionDataset(Dataset):
    """
    Returns addition problems of up to some number of digits in the inputs. Recall
    that all GPT cares about are sequences of integers, and completing them according to
    patterns in the data. Therefore, we have to somehow encode addition problems
    as a sequence of integers.

    The sum of two n-digit numbers gives a third up to (n+1)-digit number. So our
    encoding will simply be the n-digit first number, n-digit second number,
    and (n+1)-digit result, all simply concatenated together. Because each addition
    problem is so structured, there is no need to bother the model with encoding
    +, =, or other tokens. Each possible sequence has the same length, and simply
    contains the raw digits of the addition problem.

    As a few examples, the 2-digit problems:
    - 85 + 50 = 135 becomes the sequence [8, 5, 5, 0, 1, 3, 5]
    - 6 + 39 = 45 becomes the sequence [0, 6, 3, 9, 0, 4, 5]
    etc.

    We will also only train GPT on the final (n+1)-digits because the first
    two n-digits are always assumed to be given. So when we give GPT an exam later,
    we will e.g. feed it the sequence [0, 6, 3, 9], which encodes that we'd like
    to add 6 + 39, and hope that the model completes the integer sequence with [0, 4, 5]
    in 3 sequential steps.

    fun exercise: does it help if the result is asked to be produced in reverse order?
    """

    def __init__(self, ndigit, split):
        self.split = split  # train/test
        self.ndigit = ndigit
        self.vocab_size = 10  # 10 possible digits 0..9
        self.block_size = ndigit + ndigit + ndigit - 1
        num = 2500
        perm = generate_permutations()
        num_test = int(num * 0.9)
        self.ixes = perm[:num_test] if split == "test" else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx % nd
        c = a + b

        render = f"{a:0{self.ndigit}}{b:0{self.ndigit}}{c:0{self.ndigit}}"  # e.g. 03+25=28 becomes "032528"
        dix = [int(s) for s in render]  # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(
            dix[1:], dtype=torch.long
        )  # predict the next token in the sequence
        return x, y
