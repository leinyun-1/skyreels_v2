import torch, os, imageio, argparse,sys
# 将本文件上一级目录加入到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from skyreels_v2_infer.modules import get_transformer
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
from utils import FlowMatchScheduler,load_state_dict
from thuman_dataset import LatentDataset
import torch._dynamo
torch._dynamo.config.suppress_errors = True



class LightningModelForTrain(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        dit_path='/dockerdata/SkyReels-V2-I2V-14B-540P'
        learning_rate=args.learning_rate
        train_architecture=args.train_architecture
        lora_rank=args.lora_rank
        lora_alpha=args.lora_rank
        lora_target_modules=args.lora_target_modules
        init_lora_weights='kaiming'
        use_gradient_checkpointing=True
        use_gradient_checkpointing_offload=False
        pretrained_lora_path=args.pretrained_lora_path
        self.args = args 
        
        self.torch_dtype = torch.bfloat16
        self.dit = get_transformer(dit_path, 'cpu', torch.bfloat16)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)


        if train_architecture == "lora":
            self.add_lora_to_model(
                self.dit,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.dit.requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.dit.requires_grad_(False)
        self.dit.train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device,dtype=self.torch_dtype)
        bsz,f = latents.shape[0],latents.shape[2]

        prompt_emb = {}
        prompt_emb["context"] = batch['prompt_emb'].to(self.device,dtype=self.torch_dtype)
        image_emb = {}
        image_emb["clip_fea"] = batch["clip_feature"][:,0].to(self.device,dtype=self.torch_dtype)
        image_emb["y"] = batch["y"][:,0].to(self.device)

        # Loss
        self.dit.to(self.device)
        if not self.args.diff_forcing:
            if self.args.history_guide:
                zero_t_len = np.random.randint(0,5)
            else:
                zero_t_len = 0
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,))
            timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
            noisy_latents = latents.clone()
            noisy_latents[:,:,zero_t_len:] = self.scheduler.add_noise(latents[:,:,zero_t_len:], noise[:,:,zero_t_len:], timestep)
            training_target = self.scheduler.training_target(latents, noise, timestep)
        else:
            latents = latents.permute(0,2,1,3,4) # b f c h w 
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (bsz*f,))
            timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
            extra_input = self.prepare_extra_input(latents)
            noisy_latents = self.scheduler.add_noise(latents.flatten(0,1), noise.flatten(0,1), timestep)
            training_target = self.scheduler.training_target(latents, noise, timestep)

            training_target = training_target.permute(0,2,1,3,4)
            noisy_latents = noisy_latents.unflatten(0,(bsz,f)).permute(0,2,1,3,4)
            timestep = timestep.unflatten(0,(bsz,f))

        # Compute loss
        noise_pred = self.dit(
            noisy_latents, t=timestep, **prompt_emb, **image_emb, fps=15,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        if self.args.diff_forcing:
            noise_pred = noise_pred.permute(0,2,1,3,4).flatten(0,1) # bf c h w 
            training_target = training_target.permute(0,2,1,3,4).flatten(0,1) 
            timestep = timestep.flatten(0,1)
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(),reduction='none').mean(dim=tuple(range(1, noise_pred.dim()))) # 保留bs维度
            loss = torch.mean( loss * self.scheduler.training_weight(timestep).to(loss) )# 忽略了这句话，这个weight很重
        else:  
            loss = torch.nn.functional.mse_loss(noise_pred[:,:,zero_t_len:].float(), training_target[:,:,zero_t_len:].float(),reduction='none').mean(dim=tuple(range(1, noise_pred.dim()))) # 保留bs维度
            loss = torch.mean( loss * self.scheduler.training_weight(timestep).to(loss) )# 忽略了这句话，这个weight很重

        # Record log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
  

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.dit.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.dit.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.dit.state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


def train(args):
    dataset = LatentDataset(root=args.dataset_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True 
    )
    model = LightningModelForTrain(args)
    
    from swanlab.integration.pytorch_lightning import SwanLabLogger
    swanlab_config = {}
    swanlab_config.update(vars(args))
    swanlab_logger = SwanLabLogger(
        project="wan", 
        name="wan_doom_lora",
        config=swanlab_config,
        mode='local',
        logdir=os.path.join('tasks_out/train_exp/'+args.name, "swanlog"),
    )
    logger = [swanlab_logger]

    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='tasks_out/train_exp/'+args.name+'/lightning_lora_ckpts',
        filename="lora-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=500,
        save_last=True,
    )
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=args.devices,
        precision="bf16",
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    from types import SimpleNamespace
    config = {
    "name": '0825_sk_i2v_14b_lora_thuman2.1',
    'train_architecture': 'lora',
    'diff_forcing': False,
    'history_guide': False,
    'learning_rate': 1e-4,
    'dataset': 'thuman',
    'dataset_root': '/root/leinyu/data/thuman2.1/Thuman2.1_norm_render_1/latent_11',
    'devices': [1,2,3,4,5,6,7],
    'lora_rank': 32,
    'batch_size': 1,
    'accumulate_grad_batches': 1,
    'lora_target_modules': "q,k,v,o,ffn.0,ffn.2,k_img,v_img",
    'pretrained_lora_path': None,
    'lora_path': None #'tasks_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt'
    }
    cfg = SimpleNamespace(**config)

    train(cfg)


