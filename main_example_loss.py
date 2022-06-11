import torch
from torch.nn import functional


def predict(logits):
    # logits.shape = (batch_size, len_vocab)
    probs = functional.softmax(logits, dim=-1)
    index = torch.argmax(probs, dim=-1)
    print("30 + 48 = 7")
    print("predicted next number", index.item())
    print(f"P({index.item()}|3,0,4,8,7)={probs[0, index.item()].item():.3f}")


if __name__ == "__main__":

    model = torch.load("model")

    # 30 + 48 = 78
    # inputs: [3, 0, 4, 8, 7]
    # targets:   [0, 4, 8, 7, 8]

    inputs = torch.tensor([[3, 0, 4, 8, 7]])
    targets = torch.tensor([[0, 4, 8, 7, 8]])
    logits, _ = model(inputs)

    # The model returns the following probabilities:
    # logits[:, -1, :] -> P(X|3,0,4,8,7)
    # logits[:, -2, :] -> P(X|3,0,4,8)
    # logits[:, -3, :] -> P(X|3,0,4)
    # logits[:, -4, :] -> P(X|3,0)
    # logits[:, -5, :] -> P(X|3)

    predict(logits[:, -1, :])

    # Before computing the cross entropy loss, logits and targets are being reshaped
    # logits.shape = (batch_size, block_size, len_vocab) -> (batch_size * blocksize, len_vocab)
    # targets.shape = (batch_size, block_size) -> (batch_size * blocksize)
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)

    loss = functional.cross_entropy(logits, targets)
    print("loss", loss.item())

    # target = 8
    # P(8|3,0,4,8,7)
    prob_8_given_3_0_4_8_1 = functional.softmax(
        logits[
            -1,
        ],
        dim=-1,
    )[targets[-1]]

    # target = 7
    # P(7|3,0,4,8)
    prob_7_given_3_0_4_8 = functional.softmax(
        logits[
            -2,
        ],
        dim=-1,
    )[targets[-2]]

    # target = 8
    # P(8|3,0,4)
    prob_8_given_3_0_4 = functional.softmax(
        logits[
            -3,
        ],
        dim=-1,
    )[targets[-3]]

    # target = 4
    # P(4|3,0)
    prob_4_given_3_0 = functional.softmax(
        logits[
            -4,
        ],
        dim=-1,
    )[targets[-4]]

    # target = 0
    # P(0|3)
    prob_0_given_3 = functional.softmax(
        logits[
            -5,
        ],
        dim=-1,
    )[targets[-5]]

    # Chain rule:
    # P(3,0,4,8,7,8) = P(8|3,0,4,8,7) * P(7|3,0,4,8) * P(8|3,0,4) * P(4|3,0) * P(0|3)
    prob = (
        prob_8_given_3_0_4_8_1
        * prob_7_given_3_0_4_8
        * prob_8_given_3_0_4
        * prob_4_given_3_0
        * prob_0_given_3
    )
    log_prob = (
        torch.log(prob_8_given_3_0_4_8_1)
        + torch.log(prob_7_given_3_0_4_8)
        + torch.log(prob_8_given_3_0_4)
        + torch.log(prob_4_given_3_0)
        + torch.log(prob_0_given_3)
    )

    print("log_prob", torch.log(prob).item())
    print("log_prob", log_prob.item())

    cross_entropy_loss = (-log_prob) / 5
    print("loss", cross_entropy_loss.item())

    # Perplexity can be computed from cross entropy
    perplexity = torch.exp(cross_entropy_loss)
    print("perplexity", perplexity.item())

    perplexity = torch.exp(torch.log(torch.pow(prob, -1/5)))
    print("perplexity", perplexity.item())

    perplexity = torch.pow(prob, -1/5)
    print("perplexity", perplexity.item())
