import argparse
import numpy as np
import os
import yaml
import wandb
import os.path as osp
import pandas as pd
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from dataset.datasets import get_target_dataset, EncodedDataset

from sampler import SD3EulerAGSM, SDXLEulerAGSM, SD1EulerAGSM
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

from util import setup, cleanup, set_seed, save_on_master
import ImageReward as RM
from eval_utils import PickScore, HPSv2
import torch.distributed as dist

CUDA_DEVICE = torch.device("cuda")

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
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



def build_train_dataloader(args, transform, rank, world_size):
    if args.use_precomputed_encodings:
        encoding_root = args.encoding_dir or os.path.join(args.datadir, f"encodings_{args.model}")
        train_datasets = []
        for ds in args.dataset:
            encoding_path = os.path.join(encoding_root, ds)
            if not os.path.isdir(encoding_path):
                raise FileNotFoundError(
                    f"Missing precomputed encodings at {encoding_path}. "
                    "Pass --encoding_dir or omit --use_precomputed_encodings to train from images."
                )
            train_datasets.append(EncodedDataset(encoding_path, model=args.model))
    else:
        train_datasets = [
            get_target_dataset(ds, args.datadir, train=True, transform=transform)
            for ds in args.dataset
        ]

    train_dataset = ConcatDataset(train_datasets)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=not args.use_precomputed_encodings,
        pin_memory=not args.use_precomputed_encodings,
    )


@torch.no_grad()
def encode_training_batch(args, sampler, image, label):
    image = image.to(args.device)
    if args.dtype == "float16":
        image = image.half()

    prompts = [label] if isinstance(label, str) else list(label)
    latent = sampler.encode(image)
    prompt_embs = sampler.encode_prompt(prompts)
    return latent, [pe.detach() for pe in prompt_embs], image.shape[-2:]


def eval(args, model, target_dataset, eval_run_folder, rank, use_ema, use_neg, uncon2con, **sample_cfg):
    pbar_eval = tqdm.tqdm(range(rank, args.num_eval, dist.get_world_size()))
    eval_results = [{"prompt": data[-1], "img_path": f'{i:04d}.png'} for i,data in enumerate(target_dataset)]
    with torch.no_grad():
        for vi in pbar_eval:
            _, encoded, label = target_dataset[vi]
            with autocast(enabled=args.dtype == 'float16'):
                img = model.sampler.sample([""], prompt_emb=[x.to("cuda") for x in encoded], null_prompt_emb=[x.to("cuda") for x in model.null_embs], use_ema=use_ema, use_neg=use_neg, uncon2con=uncon2con, **sample_cfg)
            save_image(img, osp.join(eval_run_folder,f'{vi:04d}.png'), normalize=True)
            pbar_eval.set_description(f'SD Evaluation Sampling [{vi}/{args.num_eval}]')

    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]
    benchmark_results = {}
    for benchmark_type in benchmark_types:
        print('Benchmark Type: ', benchmark_type)
        eval_model = None
        reward_list = []
        if benchmark_type == "ImageReward-v1.0":
            eval_model = RM.load(name=benchmark_type, device="cuda")
        elif benchmark_type == "PickScore":
            eval_model = PickScore(device="cuda")
        elif benchmark_type == "HPS":
            eval_model = HPSv2()
        elif benchmark_type == "CLIP":
            eval_model = RM.load_score(
                name=benchmark_type, device="cuda"
            )

        with torch.no_grad():
            for vi in range(args.num_eval):
                prompt = eval_results[vi]["prompt"]
                img_path = os.path.join(eval_run_folder, eval_results[vi]["img_path"])

                if benchmark_type in ["ImageReward-v1.0", "PickScore", "HPS"]:
                    rewards = eval_model.score(prompt, [img_path])
                else:
                    _, rewards = eval_model.inference_rank(prompt, [img_path])
                
                if isinstance(rewards, list):
                    rewards = float(rewards[0])

                reward_list.append(rewards)
                dist.barrier()
        reward_list = np.array(reward_list)
        benchmark_results[benchmark_type] = reward_list.mean()
    return benchmark_results


class AGSMObjective(nn.Module):
    def __init__(self, sampler, temp=0.1, scale=4.0, neg_scale=1.0, device='cuda', dtype='float16'):
        super().__init__()
        self.sampler = sampler
        self.device = device
        self.dtype = dtype
        self.null_embs = self.sampler.encode_prompt([""])
        self.temp = torch.nn.Parameter(torch.tensor(temp)).to(self.device)  # temperature for contrastive loss
        self.guidance = torch.nn.Parameter(torch.tensor(scale)).to(self.device)  # guidance weight for diffusion score matching
        self.neg_guidance = torch.nn.Parameter(torch.tensor(neg_scale)).to(self.device)

    def forward(self, latent, prompt_embs, t, use_soft_tokens=True, labels=None, img_shape=(512, 512), neg=False):
        if self.dtype == 'float16':
            latent = latent.half()

        # append unconditional generation
        n_b,_,h,w = latent.shape
        n_p, n_tkn, n_dim = prompt_embs[0].shape[-3:]
        batch_latent = torch.cat([latent]*n_p, 0)
        batch_pidxs = torch.arange(n_p, device="cuda").unsqueeze(0).repeat(n_b,1).transpose(0,1).contiguous().reshape(-1)

        # set noise and timestep
        self.sampler.set_noise(img_shape=img_shape, batch_size=1)
        batch_nidxs = torch.zeros(n_p*n_b, device="cuda").long().contiguous()
        if neg:
            tid = torch.ones([n_p, n_b], device="cuda").long()
            tid[labels, torch.arange(n_b)] = 0 # 0 for positive
            tid = tid.view(-1)
        else:
            tid=False

        v, pred_v, ema_v = self.sampler.error(batch_latent, batch_nidxs, batch_pidxs, prompt_embs, t, use_soft_tokens=use_soft_tokens, get_ema=True, neg=tid)

        #reshape
        v = v.reshape(n_p, n_b, *v.shape[-3:]).transpose(0,1) # (b, n_p, c, dim, dim)
        pred_v = pred_v.reshape(n_p, n_b, *pred_v.shape[-3:]).transpose(0,1) # (b, n_p, c, dim, dim)
        ema_v = ema_v.reshape(n_p, n_b, *ema_v.shape[-3:]).transpose(0,1) # (b, n_p, c, dim, dim)
        
        # compute guidance target
        reward = - F.mse_loss(v, ema_v, reduction='none').mean(dim=(2, 3, 4), keepdim=True)
        weight = torch.exp(reward/self.temp) # (b, n_p)

        normalized_weight = weight / weight.sum(dim=1, keepdim=True) # (b, n_p)
        target_guidance = (ema_v) - torch.sum(normalized_weight * (ema_v), dim=1, keepdim=True) # (b, n_p, c, dim, dim)

        # compute error
        mask = torch.zeros((n_b,n_p), device=self.device)
        index = torch.arange(n_b, device=self.device)
        if labels is not None:
            mask[index, labels] = 1
        else:
            mask[index, index] = 1

        pos_error = mask*F.mse_loss(pred_v, v+self.guidance*target_guidance, reduction='none').mean(dim=(2, 3, 4)) # (b, n_p)
        neg_error = (1-mask)*F.mse_loss(pred_v, v-self.neg_guidance*target_guidance, reduction='none').mean(dim=(2, 3, 4))
        if self.neg_guidance > 0:
            error = pos_error.sum()/n_b + neg_error.sum()/(n_b*n_p-n_b)
        else:
            error = pos_error.sum()/n_b
        # Diagnostics: how often the positive prompt receives more weight than
        # the strongest negative prompt, and by what margin.
        positive_indices = labels if labels is not None else index
        pos_score = normalized_weight[index, positive_indices]
        neg_score = normalized_weight*(1-mask.view(n_b,n_p,1,1,1))
        neg_score = neg_score.max(dim=1)[0]
        
        winrate = torch.sum(pos_score>neg_score)/n_b
        score_margin = torch.sum(pos_score-neg_score)/n_b
        return error, winrate, score_margin
    

def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', nargs='+', type=str, default=['coco'], choices=['coco'], help='Dataset to use')
    parser.add_argument('--target_dataset', type=str, default='coco', choices=['coco'], help='Dataset to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers to split the dataset across')

    # run args
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'sdxl', 'sd1.5'], help='Model to use')
    parser.add_argument('--n_soft_tokens', type=int, default=4, help='the number of learnable soft text tokens')
    parser.add_argument('--n_soft_layers', type=int, default=5, help='the number of layers to append soft_tokens (sd3)')
    parser.add_argument('--apply_soft_tokens', nargs='+', type=str, default=[True, True, False], help='down/mid/up layers of unet (sd1.5, sdxl)')
    parser.add_argument('--use_soft_t', action='store_true', default=False, help='t dependent tokens')
    parser.add_argument('--use_soft_tokens', action='store_true', default=False, help='use soft tokens')
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--min_t', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0, help='guidance scale for guidance target noise')
    parser.add_argument('--neg_scale', type=float, default=1.0, help='guidance scale for negative guidance target noise')

    # training args
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay parameter')
    parser.add_argument('--lr', type=float, default=1e-3, help='train learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512, 768, 1024), help='training image size')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--nrows', type=int, default=1, help='number of rows per backward')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))

    # save/eval args
    parser.add_argument('--wandb', type=str2bool, default=False, help='Use wandb for logging')
    parser.add_argument('--logdir', type=str, default='./data', help='path for save checkpoint')
    parser.add_argument('--datadir', type=str, default='', required=True, help='data path')
    parser.add_argument('--use_precomputed_encodings', action='store_true', default=False, help='train from precomputed latents/text embeddings')
    parser.add_argument('--encoding_dir', type=str, default=None, help='precomputed latent/text embedding directory')
    parser.add_argument('--num_iter', type=int, default=2500, help='number of iterations before validation')
    parser.add_argument('--num_eval', type=int, default=50, help='number of generating images during validation')
    parser.add_argument('--benchmark', default="ImageReward-v1.0, CLIP, PickScore", type=str,
                        help="ImageReward-v1.0, Aesthetic, BLIP or CLIP, PickScore, HPS splitted with comma(,) if there are multiple benchmarks.")
    parser.add_argument('--note', type=str, default=None, help='note for saving path')
    parser.add_argument('--use_ema_eval', action='store_true', default=False, help='use EMA parameters to evaluate')

    args = parser.parse_args()
    args.apply_soft_tokens = [str2bool(x) for x in args.apply_soft_tokens]

    # setup
    set_seed(42)
    rank, world_size = setup()

    args.device = "cuda"

    # make run output folder
    name = f"dist-{args.model}"
    if args.img_size != 512:
        name += f'_{args.img_size}'
    name += f'_np{args.n_soft_tokens}'
    name += f'_nl{args.n_soft_layers}'
    name += f'_usesoftt{args.use_soft_t}'
    if args.note != None:
        name += f'_{args.note}'
    run_folder = osp.join(args.logdir, "-".join(args.dataset), name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # save arguments to a YAML file
    with open(os.path.join(run_folder, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    print('Arguments saved to config.yaml')
    if args.wandb and rank==0:
        wandb.init(project="AGSM",
                   name=name,
                   config=vars(args))
        
        wandb.define_metric('train/iteration')
        wandb.define_metric('train/*', step_metric='train/iteration')
        wandb.define_metric('eval/iteration')
        wandb.define_metric('eval/*', step_metric='eval/iteration')

    # set up dataset for train
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)

    # set up dataset for eval
    target_dataset = get_target_dataset(args.target_dataset, args.datadir, train=False, transform=transform)

    # load pretrained models
    if args.model == 'sd3':
        sampler = SD3EulerAGSM(n_soft_tokens=args.n_soft_tokens, n_soft_layers=args.n_soft_layers, device="cpu", use_soft_t=args.use_soft_t)
        sample_cfg = {'NFE':28, 'img_shape':(1024,1024), 'cfg_scale':4, 'use_soft_tokens':args.use_soft_tokens}
        sampler.text_enc_1.to("cuda")
        sampler.text_enc_2.to("cuda")
        sampler.text_enc_3.to("cuda")
        sampler.vae.to("cuda")
    elif args.model == 'sdxl':
        sampler = SDXLEulerAGSM(n_soft_tokens=args.n_soft_tokens, device="cpu", use_soft_t=args.use_soft_t, apply_soft_tokens=args.apply_soft_tokens)
        sample_cfg = {'NFE':30, 'img_shape':(1024,1024), 'cfg_scale':7.0, 'use_soft_tokens':args.use_soft_tokens} 
        sampler.text_enc.to("cuda")
        sampler.text_enc_2.to("cuda")
        sampler.vae.to("cuda")
        sampler.denoiser.to("cuda")
    elif args.model == 'sd1.5':
        sampler = SD1EulerAGSM(n_soft_tokens=args.n_soft_tokens, device="cpu", use_soft_t=args.use_soft_t, apply_soft_tokens=args.apply_soft_tokens)
        sample_cfg = {'NFE':30, 'img_shape':(512,512), 'cfg_scale':7.0, 'use_soft_tokens':args.use_soft_tokens} 
        sampler.text_enc.to("cuda")
        sampler.vae.to("cuda")
        sampler.denoiser.to("cuda")
        with torch.no_grad():
            # null_embs = sampler.encode_prompt([""])[0][0, :1].expand(args.n_soft_tokens, -1)
            null_embs = sampler.encode_prompt([""])[0][0, :1].unsqueeze(0).expand(2, args.n_soft_tokens, -1) # positive & negative policy
            sampler.initialize_soft_tokens(null_embs)

    # make target dataset
    with torch.no_grad():
        encoded_targets = []
        pbar_eval = tqdm.tqdm(range(args.num_eval))

        for vi in pbar_eval:
            img, label = target_dataset[vi]
            prompt_emb = sampler.encode_prompt([label])
            encoded_targets.append((img, [pe.to("cpu") for pe in prompt_emb], label))
        
        target_dataset = encoded_targets

    dist.barrier()
    if args.use_precomputed_encodings:
        print("Initializing dataset from precomputed encodings")
    else:
        print("Initializing dataset from images")
    dataloader = build_train_dataloader(args, transform, rank, world_size)

    print("Constructing model")
    model = AGSMObjective(sampler, temp=0.1, scale=args.scale, neg_scale=args.neg_scale, device=args.device, dtype=args.dtype)
    scaler = GradScaler() if args.dtype == 'float16' else None
    if args.model == 'sd3':
        if args.use_precomputed_encodings:
            del sampler.text_enc_1
            del sampler.text_enc_2
            del sampler.text_enc_3
            sampler.vae.to("cpu", non_blocking=True)
        sampler.denoiser.to("cuda")
        sampler.ema.to("cuda")
    elif args.model == 'sdxl':
        if args.use_precomputed_encodings:
            del sampler.text_enc
            del sampler.text_enc_2
            sampler.vae.to("cpu", non_blocking=True)
        sampler.denoiser.to("cuda")
        sampler.ema.to("cuda")
    elif args.model == 'sd1.5':
        if args.use_precomputed_encodings:
            del sampler.text_enc
            sampler.vae.to("cpu", non_blocking=True)
        sampler.denoiser.to("cuda")
        sampler.ema.to("cuda")

    # set requires grad
    print("Enable DDP")
    for name, param in model.sampler.denoiser.named_parameters():
        if 'soft' in name and args.use_soft_tokens:
            param.requires_grad = True
            print(f'param: {name} requires grad [True]')
        else:
            param.requires_grad = False
    model.sampler.denoiser = nn.parallel.DistributedDataParallel(model.sampler.denoiser, device_ids=[rank])
    model.sampler.denoiser.device = CUDA_DEVICE
    print("Set Optimizer")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.sampler.denoiser.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5) # T_0=20, T_mult=2

    # multi gpu
    print('Using {} GPUs'.format(torch.cuda.device_count()))
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.sampler.denoiser.parameters() if p.requires_grad)}")
    model.sampler.denoiser.train()

    save_dict = {'epoch':[], 'loss':[], 'lr':[]}
    save_dict2 = {'epoch':[], 'loss':[], 'lr':[]}
    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]
    for bch in benchmark_types:
        save_dict[bch] = []
        save_dict2[bch] = []
    
    # train
    iteration = 0
    global_loss = 0.0
    for ep in range(args.epochs):
        dataloader.sampler.set_epoch(ep)
        pbar = tqdm.tqdm(dataloader)
        for i, batch in enumerate(pbar):
            iteration += 1
            optimizer.zero_grad()
            if args.use_precomputed_encodings:
                latent, prompt_embs = batch
                latent = latent.to("cuda")
                prompt_embs = [pe.to("cuda") for pe in prompt_embs]
                img_shape = (args.img_size, args.img_size)
            else:
                image, label = batch
                latent, prompt_embs, img_shape = encode_training_batch(args, model.sampler, image, label)

            t = torch.randint(args.min_t, args.max_t, (1,), device=args.device)
            for j in range(0, latent.shape[0], args.nrows):
                with torch.autocast("cuda", enabled=args.dtype == 'float16'):
                    labels = torch.tensor(
                        list(range(j, min(latent.shape[0], j+args.nrows))),
                        device=args.device,
                    )
                    loss, winrate, score_margin = model(
                        latent[j:min(latent.shape[0], j+args.nrows)],
                        prompt_embs,
                        t.long(),
                        use_soft_tokens=args.use_soft_tokens,
                        labels=labels,
                        img_shape=img_shape,
                        neg=True,
                    )
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            global_loss += loss.item()
            pbar.set_description(f'Loss: {loss.item():.4f} Winrate: {winrate.item():.4f} Score Margin: {score_margin.item():.4f} Iteration: {iteration}')

            if args.wandb and rank==0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/epoch': ep,
                    'train/iteration': iteration,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/winrate': winrate.item(),
                    'train/score_margin': score_margin.item(),
                })

            # update ema
            update_ema(model.sampler.ema, model.sampler.denoiser.module, decay=args.ema_decay)

            # validation
            if iteration % args.num_iter == 0: 
                model.sampler.vae.to("cuda")
                model.sampler.vae.eval()
                model.sampler.denoiser.eval()
                it_ep = iteration // args.num_iter
                # save model
                global_loss /= args.num_iter
                print(f'Epoch {ep} Iteration {iteration}: Loss: {global_loss:.4f}')
                if args.use_soft_tokens:
                    save_on_master(sampler.denoiser.module.soft_tokens, osp.join(run_folder, f'soft_tokens_{it_ep}.pth'))
                    if args.use_soft_t:
                        save_on_master(sampler.denoiser.module.soft_t_tokens, osp.join(run_folder, f'soft_t_tokens_{it_ep}.pth'))
                # evaluate
                with torch.no_grad():
                    eval_run_folder = osp.join(run_folder, f'val_samples_{it_ep}')
                    if rank==0:
                        os.makedirs(eval_run_folder, exist_ok=True)

                    benchmark_results = eval(args, model, target_dataset, eval_run_folder, rank, use_ema=args.use_ema_eval, use_neg=True, uncon2con=False, **sample_cfg)

                    for bch,result in benchmark_results.items():
                        save_dict[bch].append(result) 
                    if rank==0:
                        save_dict['epoch'].append(it_ep)
                        save_dict['loss'].append(global_loss)
                        current_lr = optimizer.param_groups[0]['lr']
                        save_dict['lr'].append(current_lr)
                        df = pd.DataFrame(save_dict)
                        df.to_csv(os.path.join(run_folder, 'run.csv'), index=False)
                
                # evaluate 2
                with torch.no_grad():
                    eval_run_folder = osp.join(run_folder, f'val_samples_{it_ep}_pp')
                    if rank==0:
                        os.makedirs(eval_run_folder, exist_ok=True)

                    benchmark_results2 = eval(args, model, target_dataset, eval_run_folder, rank, use_ema=args.use_ema_eval, use_neg=False, uncon2con=False, **sample_cfg)

                    for bch,result in benchmark_results2.items():
                        save_dict2[bch].append(result) 
                    if rank==0:
                        save_dict2['epoch'].append(it_ep)
                        save_dict2['loss'].append(global_loss)
                        current_lr = optimizer.param_groups[0]['lr']
                        save_dict2['lr'].append(current_lr)
                        df = pd.DataFrame(save_dict2)
                        df.to_csv(os.path.join(run_folder, 'run_pp.csv'), index=False)

                # logging
                if args.wandb and rank==0:
                    wandb_log = {
                        'eval/epoch': ep,
                        'eval/iteration': iteration,
                    }
                    for bch, result in benchmark_results.items():
                        wandb_log[f'eval/{bch}'] = result
                    for bch, result in benchmark_results2.items():
                        wandb_log[f'eval/{bch}_pp'] = result
                    wandb.log(wandb_log)

                model.sampler.denoiser.train()
                if args.use_precomputed_encodings:
                    model.sampler.vae.to("cpu")

                # reset loss
                global_loss = 0.0
                lr_scheduler.step()
            dist.barrier()

        pbar.close()

    if args.wandb and rank==0:
        wandb.finish()
    print(f'Training complete. Saving model to {run_folder}')
    cleanup()

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
