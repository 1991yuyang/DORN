import os
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
from network import Dorn
from torch import nn, optim
from loss import Loss
from datasets import *


def train_epoch(train_loader, current_epoch, optimizer, model, criterion):
    model.train()
    steps = len(train_loader)
    current_step = 1
    for d_train, l_train in train_loader:
        d_train_cuda = d_train.cuda(0)
        l_train_cuda = l_train.cuda(0)
        train_output = model(d_train_cuda)
        train_loss = criterion(train_output, l_train_cuda)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if current_step % 5 == 0:
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f" % (current_epoch, epochs, current_step, steps, train_loss.item()))
        current_step += 1
    t.save(model.module.state_dict(), os.path.join(model_save_dir, "epoch.pth"))
    return model


def valid_epoch(valid_loader, current_epoch, criterion, model):
    global best_valid_loss
    model.eval()
    steps = len(valid_loader)
    accum_loss = 0
    for d_valid, l_valid in valid_loader:
        d_valid_cuda = d_valid.cuda(0)
        l_valid_cuda = l_valid.cuda(0)
        with t.no_grad():
            valid_output = model(d_valid_cuda)
            valid_loss = criterion(valid_output, l_valid_cuda)
            accum_loss += valid_loss.item()
    avg_loss = accum_loss / steps
    print("##########valid epoch:%d##########" % (current_epoch,))
    print("valid_loss:%.5f" % (avg_loss,))
    if avg_loss < best_valid_loss:
        print("saving best model......")
        best_valid_loss = avg_loss
        t.save(model.module.state_dict(), os.path.join(model_save_dir, "best.pth"))
    return model


def main():
    model = Dorn(resnet_type=resnet_type, K=K, us_use_interpolate=us_use_interpolate)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(0)
    # model.module.load_state_dict(t.load("model/epoch.pth"))
    optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    lr_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=final_lr)
    for e in range(epochs):
        current_epoch = e + 1
        train_loader = make_loader(data_root_dir, True, image_crop_size, alpha, beta, K, batch_size, num_workers, scale_max_factor, input_image_size)
        valid_loader = make_loader(data_root_dir, False, image_crop_size, alpha, beta, K, batch_size, num_workers, scale_max_factor, input_image_size)
        model = train_epoch(train_loader, current_epoch, optimizer, model, criterion)
        model = valid_epoch(valid_loader, current_epoch, criterion, model)
        lr_sch.step()


if __name__ == "__main__":
    epochs = 1000
    batch_size = 4
    init_lr = 0.0001
    final_lr = 0.00001
    image_crop_size = [320, 320]  # [w, h]
    input_image_size = [512, 512]  # [w, h]
    K = 120  # count of sub interval of depth
    resnet_type = "resnet34"  # resnet18, resnet34, resnet50, resnet101
    us_use_interpolate = True
    model_save_dir = r"model"
    data_root_dir = r"/home/yuyang/data/make3d"
    scale_max_factor = 1.5  # max ratio of scale augmentation
    alpha = 0  # min depth of dataset
    beta = 255  # max depth of dataset
    num_workers = 4
    weight_decay = 0.00005
    alpha = alpha // scale_max_factor
    device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
    best_valid_loss = float("inf")
    criterion = Loss(image_crop_size=image_crop_size, K=K)
    main()