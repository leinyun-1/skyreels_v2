import argparse
import gc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import random
import time

import imageio
import torch
from diffusers.utils import load_image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop
from skyreels_v2_infer.pipelines import Text2VideoPipeline


import torch._dynamo
torch._dynamo.config.suppress_errors = True

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}

def load_prompts(file_path: str) -> dict:
    '''
    将txt文件中每行按空格分成key和value
    
    Args:
        file_path (str): txt文件的路径
        
    Returns:
        dict: 包含所有键值对的字典
    '''
    if not os.path.exists(file_path):
        raise ValueError(f"文件不存在: {file_path}")
        
    if not file_path.endswith('.txt'):
        raise ValueError(f"文件必须是txt格式: {file_path}")
        
    prompts_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除每行首尾的空白字符
            line = line.strip()
            # 跳过空行
            if line:
                # 按第一个空格分割
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    key, value = parts
                    prompts_dict[key] = value.strip()
                else:
                    print(f"警告: 行 '{line}' 格式不正确，已跳过")
                
    return prompts_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="video_out")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-T2V-14B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P",'832'])
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    args = parser.parse_args()

    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    if args.resolution == "540P":
        height = 544
        width = 960
    elif args.resolution == "720P":
        height = 720
        width = 1280
    elif args.resolution == '832':
        height = 832
        width = 832
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    local_rank = 0

    assert "I2V" in args.model_id, f"check model_id:{args.model_id}"
    print("init img2video pipeline")
    pipe = Image2VideoPipeline(
        model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload, device=args.device
    )
    from tasks.utils import load_lora
    #lora_path = 'tasks_out/train_exp/0825_sk_i2v_14b_lora_thuman2.1/lightning_lora_ckpts/lora-epoch=03-step=006000.ckpt'
    lora_path = args.lora_path
    lora_state_dict = torch.load(lora_path, map_location="cpu") 
    load_lora(pipe.transformer, lora_state_dict)

    prompts_input = load_prompts(args.prompt)
    images = os.listdir(args.image)
    for img in images:
        if img.endswith('.png'):
            image = load_image(os.path.join(args.image,img))
            image = image.resize((height,width))
            prompt_input = prompts_input[img.split('.')[0]]
                
            kwargs = {
                "prompt": prompt_input,
                "image": image,
                "negative_prompt": negative_prompt,
                "num_frames": args.num_frames,
                "num_inference_steps": args.inference_steps,
                "guidance_scale": args.guidance_scale,
                "shift": args.shift,
                "generator": torch.Generator(device="cuda").manual_seed(args.seed),
                "height": height,
                "width": width,
            }


            save_dir = os.path.join("result", args.outdir)
            os.makedirs(save_dir, exist_ok=True)

            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                print(f"infer kwargs:{kwargs}")
                video_frames = pipe(**kwargs)[0]

            if local_rank == 0:
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                #video_out_file = f"{prompt_input[:100].replace('/','')}_{args.seed}_{current_time}_{args.inference_steps}_{args.guidance_scale}_{args.shift}.mp4"
                video_out_file = f"{img[:-4]}_{args.seed}_{current_time}_{args.inference_steps}_{args.guidance_scale}_{args.shift}.mp4"
                output_path = os.path.join(save_dir, video_out_file)
                imageio.mimwrite(output_path, video_frames, fps=args.fps, quality=8, output_params=["-loglevel", "error"])
