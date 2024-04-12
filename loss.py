import torch as t
from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):

    def __init__(self, K, image_crop_size):
        """

        :param K:
        :param image_crop_size: (w, h)
        """
        super(Loss, self).__init__()
        self.K = K
        self.image_size = image_crop_size

    def forward(self, model_output, target):
        """

        :param model_output: [N, K, h, w]
        :param target: [N, h, w], range of element of target {0, 1, ..., K − 1}
        :return:
        """
        loss = 0
        for i in range(model_output.size()[0]):
            out = model_output[i]  # [K, h, w]
            pos_prob = out.permute(dims=[1, 2, 0])  # [h, w, K]
            neg_prob = 1 - pos_prob  # [h, w, K]
            ti = target[i]  # [h, w]
            accum_one_hot_out = 0
            while t.sum(ti).item() >= 0:
                accum_one_hot_out += F.one_hot(ti, self.K)  # [h, w, K]
                if t.sum(ti).item() == 0:
                    break
                ti[ti > 0] = ti[ti > 0] - 1
            accum_one_hot_out[accum_one_hot_out > 2] = 1
            loss -= t.mean(t.sum(t.log(pos_prob + 0.00000000001) * accum_one_hot_out + t.log(neg_prob + 0.00000000001) * (1 - accum_one_hot_out), dim=2) / self.K)
        loss = loss / model_output.size()[0]
        return loss


if __name__ == "__main__":
    x = t.randint(0, 10, (2, 3))
    print(x)
    y = F.one_hot(x, 11)
    print(y)
