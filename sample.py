import argparse
import glob
import numpy as np
import random, os
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
import torch.nn as nn
import tqdm

from sampler import SD3EulerAGSM, SDXLEulerAGSM, SD1EulerAGSM
from dataset.datasets import get_target_dataset
import json

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def _checkpoint_tensor(obj):
    if isinstance(obj, nn.Embedding):
        return obj.weight.detach()
    if isinstance(obj, nn.Parameter):
        return obj.detach()
    if torch.is_tensor(obj):
        return obj.detach()
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")


def resolve_token_checkpoint(checkpoint_dir, stem, load_ep=None):
    candidates = []
    if load_ep is not None:
        candidates.append(os.path.join(checkpoint_dir, f'{stem}_{load_ep}.pth'))
    candidates.append(os.path.join(checkpoint_dir, f'{stem}.pth'))
    candidates.extend(sorted(glob.glob(os.path.join(checkpoint_dir, f'{stem}_*.pth'))))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find {stem}.pth in {checkpoint_dir}. "
        "Expected a released token file without an epoch suffix, or a legacy "
        f"{stem}_<epoch>.pth file."
    )


def load_token_checkpoint(target, path):
    target_tensor = target.weight if isinstance(target, nn.Embedding) else target
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    loaded_tensor = _checkpoint_tensor(loaded).to(
        device=target_tensor.device,
        dtype=target_tensor.dtype,
    )
    if tuple(loaded_tensor.shape) != tuple(target_tensor.shape):
        raise ValueError(
            f"Shape mismatch for {path}: checkpoint {tuple(loaded_tensor.shape)} "
            f"vs target {tuple(target_tensor.shape)}"
        )
    target_tensor.data.copy_(loaded_tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sampling config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img_size', type=int, default=1024, choices=[256,512,768,1024])
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='0 for null prompt, 1 for only using conditional prompt')
    parser.add_argument('--batch_size', type=int, default=1)
    # path
    parser.add_argument('--load_dir', type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument('--load_ep', type=int, default=None, help="optional epoch suffix for legacy checkpoints")
    parser.add_argument('--save_dir', type=str, default=None, help="output root; defaults to load_dir or outputs/samples")
    parser.add_argument('--datadir', type=str, default='', required=True, help='data path')
    # model config
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'sdxl', 'sd1.5'], help='Model to use')
    parser.add_argument('--use_soft_tokens', action='store_true', default=False)
    parser.add_argument('--use_soft_t', action='store_true', default=False, help='note for saving path')
    parser.add_argument('--n_soft_tokens', type=int, default=4)
    parser.add_argument('--n_soft_layers', type=int, default=5, help='sd3')
    parser.add_argument('--apply_soft_tokens', nargs='+', type=str, default=[True, True, False], help='sdxl, sd1.5')
    # sampling policy
    parser.add_argument('--uncon2con', action='store_true', default=False, help='use nagative policy with condition instead of unconditional generaiton')
    parser.add_argument('--uncon2neg', action='store_true', default=False, help='use negative policy with unconditional generation')

    # one sample generation
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--save_name', type=str, default="image_sd3")
    # set generation
    parser.add_argument('--num', type=int, default=-1, help='number of sampling images. -1 for whole dataset')
    parser.add_argument('--dataset', type=str, nargs='+', default=['coco'], choices=['coco'])
    
    args = parser.parse_args()
    set_seed(args.seed)

    args.apply_soft_tokens = [str2bool(x) for x in args.apply_soft_tokens]
    args.use_soft_t = str2bool(args.use_soft_t)

    interpolation = INTERPOLATIONS['bilinear']
    transform = get_transform(interpolation, args.img_size)

    # save dir
    config=f'{"-".join(args.dataset)}-cfg{args.cfg_scale}-soft{args.use_soft_tokens}-softt{args.use_soft_t}'
    if args.load_ep is not None:
        config = f'{config}-ep{args.load_ep}'
    config = f'{config}-num{args.num}'
    if args.uncon2neg:
        config = f'{config}-uncon2neg'
    checkpoint_dir = args.load_dir
    output_root = args.save_dir or checkpoint_dir or os.path.join("outputs", "samples")
    if args.use_soft_tokens and checkpoint_dir is None:
        raise ValueError("--load_dir is required when --use_soft_tokens is set")
    savedir = os.path.join(output_root, config)
    os.makedirs(savedir, exist_ok=True)

    # load model
    if args.model == 'sd3':
        sampler = SD3EulerAGSM(n_soft_tokens=args.n_soft_tokens, use_soft_t=args.use_soft_t, n_soft_layers=args.n_soft_layers)
        sample_cfg = {'NFE': args.NFE, 'img_shape': (args.img_size, args.img_size), 'cfg_scale': args.cfg_scale}
    elif args.model == 'sdxl':
        sampler = SDXLEulerAGSM(n_soft_tokens=args.n_soft_tokens, use_soft_t=args.use_soft_t, apply_soft_tokens=args.apply_soft_tokens)
        sample_cfg = {'NFE': args.NFE, 'img_shape': (args.img_size, args.img_size), 'cfg_scale': args.cfg_scale} 
    elif args.model == 'sd1.5':
        sampler = SD1EulerAGSM(n_soft_tokens=args.n_soft_tokens, use_soft_t=args.use_soft_t, apply_soft_tokens=args.apply_soft_tokens)
        sample_cfg = {'NFE': args.NFE, 'img_shape': (args.img_size, args.img_size), 'cfg_scale': args.cfg_scale} 
    else:
        raise ValueError('args.model should be one of [sd3, sdxl, sd1.5]')
    
    # load tokens
    if checkpoint_dir is not None:
        load_token_checkpoint(sampler.denoiser.soft_tokens, resolve_token_checkpoint(checkpoint_dir, 'soft_tokens', args.load_ep))
        print('**soft tokens loaded')
        if args.use_soft_t:
            load_token_checkpoint(sampler.denoiser.soft_t_tokens, resolve_token_checkpoint(checkpoint_dir, 'soft_t_tokens', args.load_ep))
            print('**soft_t tokens loaded')
    train_datasets = []
    for ds in args.dataset:
        train_datasets.append(get_target_dataset(ds, args.datadir, train=False, transform=transform))

    train_dataset = ConcatDataset(train_datasets)
    num = args.num if args.num != -1 else len(train_dataset)
    train_dataset = Subset(train_dataset, list(range(num)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    pbar = tqdm.tqdm(train_dataloader)
    i=0
    results = []
    for _, label in pbar:
        if os.path.exists(os.path.join(savedir, f'{i+args.batch_size:04d}.png')):
            i+=1
            continue
        img = sampler.sample(label, use_soft_tokens=args.use_soft_tokens, batch_size=len(label), use_neg=args.uncon2neg, uncon2con=args.uncon2con, **sample_cfg)
        for bi in range(img.shape[0]):
            imgname = f'{i:04d}.png'
            save_image(img[bi], os.path.join(savedir, imgname), normalize=True)
            results.append({"prompt": label[bi], "img_path": imgname})
            pbar.set_description(f'SD Sampling [{i}/{num}]')
            i+=1
    
    # save config
    results_path = os.path.join(output_root, f"results-{config}.json")
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as file:
            results_all = json.load(file)
            if isinstance(results_all, list):
                results_all.extend(results)
            else:
                results_all = [results_all] + results
    else:
        results_all = results
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results_all, file, indent=4)  # `indent=4` makes the JSON more readable
