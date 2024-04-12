from blocks import *
from utils import SID
from torch.nn import functional as F


class Dorn(nn.Module):

    def __init__(self, resnet_type, K, us_use_interpolate):
        """

        :param resnet_type: "resnet18", "resnet34", "resnet50", "resnet101"
        :param K: depth range is discretized into K sub-intervals
        :param us_use_interpolate: True will use interpolate,False use ConvTranspose2d
        """
        super(Dorn, self).__init__()
        self.sum = SUM(resnet_type, us_use_interpolate)  # Scene understanding modular
        self.orid_reg = OridReg()  # Ordinal regression
        self.us = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=K, kernel_size=2, stride=2, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        sum_out = self.sum(x)
        reg_out = self.orid_reg(sum_out)
        reg_out = self.us(reg_out)  # [N, K, input_h, input_w]
        if t.onnx.is_in_onnx_export():
            pos_prob = reg_out[0]  # [K, input_h, input_w]
            pos_prob = pos_prob.permute(dims=[1, 2, 0])  # [input_h, input_w, K]
            pos_mask = (pos_prob > 0.5).type(t.IntTensor)  # [input_h, input_w, K]
            # pos_mask = t.sum(pos_mask, dim=2)
            # pos_mask[pos_mask == 0] = 1  # [input_h, input_w, K], value range of element 1 ~ K, (ti[element - 1] + ti[element]) / 2 - epsilon
            return pos_mask
        return reg_out


if __name__ == "__main__":
    model = Dorn(resnet_type="resnet101", K=10, us_use_interpolate=True)
    d = t.randn(5, 3, 640, 640)
    out = model(d)
    print(out.size())