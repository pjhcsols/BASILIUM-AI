# import default package
import time
import os
import argparse
import glob

# import torch package
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torch.utils.data import DataLoader

# import third-party package
import numpy as np
import cv2
from util import flow_util
from collections import OrderedDict

# import user-defined package
from options.test_options import TestOptions
from data.test_dataset import TestDataset
from models.networks import ResUnetGenerator
from models.afwm import AFWM
from models.network_swinir import SwinIR as sr_model

from torchvision import transforms


def create_VITON_model(opt):
    warp_model = AFWM(opt, 3)       # warping 모델
    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)        # generate 모델
    warp_model.eval()
    warp_model = warp_model.to(opt.device)
    gen_model.eval()
    gen_model = gen_model.to(opt.device)

    load_checkpoint(warp_model, opt.warp_model_checkpoint)
    load_checkpoint(gen_model, opt.gen_model_checkpoint)

    return warp_model, gen_model


def create_SR_model(opt):
    swinIR = sr_model(upscale=opt.scale, in_chans=3, img_size=64, window_size=8,     # sr 모델
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'

    swinIR.eval()
    swinrIR = swinIR.to(opt.device)

    load_checkpoint(swinIR, opt.sr_model_checkpoint, key = param_key_g)

    return swinIR


def create_denoise_model(opt):
    assert opt.noise in [0, 15, 25, 50]

    denoise_model = sr_model(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'

    denoise_model.eval()
    denoise_model = denoise_model.to(opt.device)

    if opt.noise == 50:
        load_checkpoint(denoise_model, opt.denoise50_model_checkpoint, key = param_key_g)
    elif opt.noise == 25:
        load_checkpoint(denoise_model, opt.denoise25_model_checkpoint, key = param_key_g)
    elif opt.noise == 15:
        load_checkpoint(denoise_model, opt.denoise15_model_checkpoint, key = param_key_g)

    return denoise_model


def load_checkpoint(model, checkpoint_path, key = None):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path)

    if key is None:
        model.load_state_dict(checkpoint, strict = False)
    else:
        model.load_state_dict(checkpoint[key] if key in checkpoint.keys() else checkpoint, strict=True)


def create_data_loader(opt):
    dataset = TestDataset(opt)
    data_loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.nThreads)

    return dataset, data_loader


def test():
    ######options######
    opt = TestOptions().parse()

    ######define model######
    warp_model, gen_model = create_VITON_model(opt)
    swinIR = create_SR_model(opt)
    if opt.noise: denoise_model = create_denoise_model(opt)
    f2c = flow_util.flow2color()        # 플로우 측정기

    ######load data######
    dataset, data_loader = create_data_loader(opt)
    dataset_size = len(dataset)

    ######define parameters######
    if opt.noise == 0:
        save_dir, border, window_size = opt.output_path, 0, 8
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir, border, window_size = opt.output_path, 0, 128
        os.makedirs(save_dir, exist_ok=True)

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size / opt.batchSize

    original_shape = dataset.get_image_shape()

    for epoch in range(1,2):
        for i, data in enumerate(data_loader, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            real_image = data['image']
            clothes = data['clothes']
            edge = data['edge']

            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
            clothes = clothes * edge

            flow_out = warp_model(real_image.to(opt.device), clothes.to(opt.device))
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.to(opt.device), last_flow.permute(0, 2, 3, 1),
                            mode='bilinear', padding_mode='zeros')
            
            warped_cloth = transforms.CenterCrop((256 - 32, 192 - 28))(warped_cloth)
            warped_edge = transforms.CenterCrop((256 - 32, 192 - 28))(warped_edge)

            warped_cloth = transforms.Resize((256, 192))(warped_cloth)
            warped_edge = transforms.Resize((256, 192))(warped_edge)
            
            gen_inputs = torch.cat([real_image.to(opt.device), warped_cloth, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)

            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            print('viton complete')
            
            ###SR###
            imgname, img_lq = opt.name, utils.return_image(
                    p_tryon,
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1,1),)  # image to HWC-BGR, float32

            img_lq = np.transpose(img_lq, (2, 0, 1))
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(opt.device)  # CHW-RGB to NCHW-RGB
            
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = SR_test(img_lq, swinIR, opt, window_size)
                output = output[..., :h_old * opt.scale, :w_old * opt.scale]

                if opt.noise: output = SR_test(output, denoise_model, opt, window_size)

            print('sr complete')

            print(data['p_name'])

            # save image
            output = output.data.squeeze().float().clamp_(0, 1)

            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

            if output.shape[1] < original_shape[1]:
                output = cv2.resize(output, (original_shape[1], original_shape[0]), interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(f'{save_dir}/{data["p_name"][0]}', output)

            step += 1
            if epoch_iter >= dataset_size:
                break
            
            if step == 100:
                # iter_end_time = time.time()
                # print(f'one epoch time: {iter_end_time - iter_start_time}')
                return 0
            if step % 10 == 0:
                torch.cuda.empty_cache()


def SR_test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == "__main__":
    test()
    torch.cuda.empty_cache()