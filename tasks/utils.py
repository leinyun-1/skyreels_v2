import torch
from safetensors import safe_open


class FlowMatchScheduler():
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    # def add_noise(self, original_samples, noise, timestep):
    #     if isinstance(timestep, torch.Tensor):
    #         timestep = timestep.cpu()
    #     timestep_id = torch.argmin((self.timesteps - timestep).abs())
    #     sigma = self.sigmas[timestep_id]
    #     sample = (1 - sigma) * original_samples + sigma * noise
    #     return sample
    
    def add_noise(self, original_samples, noise, timestep): # 原版只是适应timestep的bs=1，故改掉
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(-1)).abs(),dim=-1)
        sigma = self.sigmas[timestep_id]
        sigma = sigma.view(*([sigma.shape[0]] + [1] * (noise.ndim - 1))).to(noise)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.to(self.timesteps.device).unsqueeze(-1)).abs(),dim=-1)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

    def training_weight_a(self, timestep):
        size = timestep.shape[0]
        timestep_id = torch.argmin((self.timesteps.unsqueeze(-1).repeat(1, size) - timestep.to(self.timesteps.device)).abs(),dim=0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
    

def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def load_lora(model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
    state_dict_model = model.state_dict()
    device, dtype, computation_device, computation_dtype = fetch_device_and_dtype(state_dict_model)
    lora_name_dict = get_name_dict(state_dict_lora)
    for name in lora_name_dict:
        weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=computation_device, dtype=computation_dtype)
        weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=computation_device, dtype=computation_dtype)
        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
            weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_lora = alpha * torch.mm(weight_up, weight_down)
        weight_model = state_dict_model[name].to(device=computation_device, dtype=computation_dtype)
        weight_patched = weight_model + weight_lora
        state_dict_model[name] = weight_patched.to(device=device, dtype=dtype)
    print(f"    {len(lora_name_dict)} tensors are updated.")
    model.load_state_dict(state_dict_model)

def get_name_dict(lora_state_dict):
    lora_name_dict = {}
    for key in lora_state_dict:
        if ".lora_B." not in key:
            continue
        keys = key.split(".")
        if len(keys) > keys.index("lora_B") + 2:
            keys.pop(keys.index("lora_B") + 1)
        keys.pop(keys.index("lora_B"))
        if keys[0] == "diffusion_model" or keys[0] == 'model':
            keys.pop(0)
        target_name = ".".join(keys)
        lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
    return lora_name_dict


def fetch_device_and_dtype(state_dict):
    device, dtype = None, None
    for name, param in state_dict.items():
        device, dtype = param.device, param.dtype
        break
    computation_device = device
    computation_dtype = dtype
    if computation_device == torch.device("cpu"):
        if torch.cuda.is_available():
            computation_device = torch.device("cuda")
    if computation_dtype == torch.float8_e4m3fn:
        computation_dtype = torch.float32
    return device, dtype, computation_device, computation_dtype

