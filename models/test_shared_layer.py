import torch
from torch import nn
import random
from copy import deepcopy


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2, bias=False)
        self.linear2 = nn.Linear(3, 2, bias=False)
        self.shared = nn.Linear(2, 1, bias=False)

    def set_requires_grad(self, val: bool):
        for p in self.parameters():
            p.requires_grad = val

    def check_grad(self):
        print("-------- check grad")

        for n, p in self.named_parameters():
            print(n, p.requires_grad)
        print("--- done checking grad")

    def forward(self, x, is_linear1: bool):
        self.set_requires_grad(True)
        self.check_grad()

        print('is_linear1=', is_linear1)
        if is_linear1:
            self.linear2.weight.requires_grad = False
            self.linear2.weight.grad = None
            self.check_grad()

            x = self.linear1(x)
        else:
            self.linear1.weight.requires_grad = False
            self.linear1.weight.grad = None
            self.check_grad()

            x = self.linear2(x)

        x = self.shared(x)
        return x

if __name__ == '__main__':

    model = ToyModel()
    mse = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    n_epochs = 100
    for e in range(n_epochs):
        print("epoch", e + 1)
        print("ln1", model.linear1.weight)
        print("ln2", model.linear2.weight)

        l1 = deepcopy(model.linear1.weight.data)
        l2 = deepcopy(model.linear2.weight.data)

        x = torch.rand((4, 3))
        y = torch.rand((4, 1))

        rand = random.randint(0, 1)
        is_l1 = (rand == 0)
        yhat = model(x, is_linear1=is_l1)

        loss = mse(yhat, y)

        optimizer.zero_grad()
        loss.backward()

        print("grads:")
        print("model.linear1.weight.requires_grad", model.linear1.weight.requires_grad)
        print("grad ln1", model.linear1.weight.grad)

        print("model.linear2.weight.requires_grad", model.linear2.weight.requires_grad)
        print("grad ln2", model.linear2.weight.grad)

        optimizer.step()

        print("AFTER ln1", model.linear1.weight)
        print("AFTER ln2", model.linear2.weight)
        l1_after = deepcopy(model.linear1.weight.data)
        l2_after = deepcopy(model.linear2.weight.data)

        if is_l1:
            assert not torch.equal(l1, l1_after)
            assert torch.equal(l2, l2_after)
        else:
            assert torch.equal(l1, l1_after)
            assert not torch.equal(l2, l2_after)



