"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
from numpy.lib.type_check import imag
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from PIL import Image

from training.data.demo_loader import get_loader

import warnings
warnings.filterwarnings("ignore")
from colorama import init
from colorama import Fore, Style
from icecream import ic
init(autoreset=True)
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
from PIL import Image
import shutil
import cv2
from convert import convert_label
#----------------------------------------------------------------------------

def save_image(img, name):
    x = denormalize(img.detach().cpu())
    x = x.permute(1, 2, 0).numpy()
    x = np.rint(x) / 255.
    plt.imsave(name+'.png', x)
    
def denormalize(tensor):
    pixel_mean = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    pixel_std = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    denormalizer = lambda x: torch.clamp((x * pixel_std) + pixel_mean, 0, 255.)

    return denormalizer(tensor)

def visualize_demo(i, img, inv_mask, erased_img, pred_img, comp_img):
    lo, hi = [-1, 1]

    img = np.asarray(img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    inv_mask = torch.stack([inv_mask[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    inv_mask = np.asarray(inv_mask, dtype=np.float32).transpose(1, 2, 0)
    inv_mask = np.rint(inv_mask).clip(0, 255).astype(np.uint8)

    mask = PIL.Image.fromarray(inv_mask)
    mask.save('visualizations/masks/' + i + '.png')

    erased_img = np.asarray(erased_img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    erased_img = (erased_img - lo) * (255 / (hi - lo))
    erased_img = np.rint(erased_img).clip(0, 255).astype(np.uint8)
    erased_img = erased_img * (1 - inv_mask)

    pred_img = np.asarray(pred_img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    pred_img = (pred_img - lo) * (255 / (hi - lo))
    pred_img = np.rint(pred_img).clip(0, 255).astype(np.uint8)
    
    comp_img = np.asarray(comp_img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    comp_img = (comp_img - lo) * (255 / (hi - lo))
    comp_img = np.rint(comp_img).clip(0, 255).astype(np.uint8)
    
    plt.imsave('visualizations/images/' + i + '.png', img / 255)
    plt.imsave('visualizations/erased_images/' + i + '.png', erased_img / 255)
    plt.imsave('visualizations/comp_images/' + i + '.png', comp_img / 255)
    plt.close()

def create_folders():
    if not os.path.exists('visualizations/comp_images/'):
        os.makedirs('visualizations/comp_images/')
    if not os.path.exists('visualizations/images/'):
        os.makedirs('visualizations/images/')
    if not os.path.exists('visualizations/masks/'):
        os.makedirs('visualizations/masks/')
    if not os.path.exists('visualizations/erased_images/'):
        os.makedirs('visualizations/erased_images/')
#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--img_data', help='Training images (directory)', metavar='PATH', required=True)
@click.option('--resolution', help='Res of Images [default: 256]', type=int, metavar='INT')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    img_data: str,
    resolution: int,
    class_idx: Optional[int],
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




    if resolution is None:
        resolution = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    
    G = G.eval().to(device)
    
    dataloader = get_loader(img_path=img_data, resolution=resolution)
    ic(G.encoder.b256.img_channels)
    
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    
    netG_params = sum(p.numel() for p in G.parameters())
    print(Fore.BLUE +"Generator Params: {} M".format(netG_params/1e6))
    print(Style.BRIGHT + Fore.GREEN + "Starting Visualization...")
    times = []

    create_folders()
    j = 0
    
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc='Visualizing..'):

        with torch.no_grad():
            ## data is a tuple of (rgbs, rgbs_erased, amodal_tensors, visible_mask_tensors, erased_modal_tensors) ####
            images, erased_images, invisible_masks, fnames = data
            print(erased_images.shape)

            erased_img = erased_images.to(device)
            mask  = invisible_masks.to(device)
            fname = fnames[0]
            #print(fname)

            start_time = time.time()
            pred_img = G(img=torch.cat([0.5 - mask, erased_img], dim=1), c=label, truncation_psi=truncation_psi, noise_mode='const')
            print(pred_img.shape)
            comp_img = invisible_masks.to(device) * pred_img + (1 - invisible_masks).to(device) * images.to(device)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if torch.mean(invisible_masks).item() != 0:
                j += 1
                visualize_demo(fname, images, invisible_masks, erased_images, pred_img.detach(), comp_img.detach())

    avg_time = np.mean(times)
    print(Fore.CYAN + "Duration per image: {} s".format(avg_time))


    '''# 输入文件夹和输出文件夹的路径
    input_folder = 'visualizations/comp_images/'
    output_folder = 'data/cityscapes/color/val'

    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 确保文件是图像文件
            # 读取彩色语义分割图
            input_path = os.path.join(input_folder, filename)

            # 转换为灰度图
            label_semantic_segmentation = convert_label(input_path)

            # 构造输出文件路径
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_gray.png')

            # 保存灰度图到输出文件夹
            cv2.imwrite(output_path, label_semantic_segmentation)

    print('转换完成！')'''



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter


#----------------------------------------------------------------------------
'''
    # 源文件夹路径
    source_folder = 'data\cityscapes\irregular_mask'
    # 目标文件夹路径
    destination_folder = img_data
    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件是否为PNG格式的图像
        if filename.lower().endswith('.png'):
            # 构建源文件和目标文件的完整路径
            a_filepath = os.path.join(source_folder, filename)
            b_filename = filename.replace('_mask', '')
            #print(b_filename)
            b_filepath = os.path.join(destination_folder, filename)
            print(b_filepath)
            if os.path.exists(os.path.join(destination_folder, b_filename)):
            # 打开图像
                with Image.open(a_filepath) as img:
                # 将黑色变为白色，白色变为黑色
                    img = img.convert('L').point(lambda x: 255 if x < 128 else 0)

                # 保存处理后的图像到目标文件夹
                    img.save(b_filepath)
    print("处理完成。")'''