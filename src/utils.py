import torch


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss
