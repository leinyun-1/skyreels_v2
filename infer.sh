#model_id=/dockerdata/SkyReels-V2-DF-1.3B-540P
# synchronous inference
# python3 generate_video_df.py \
#   --model_id ${model_id} \
#   --resolution 540P \
#   --ar_step 0 \
#   --base_num_frames 97 \
#   --num_frames 257 \
#   --overlap_history 17 \
#   --prompt "A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface, with the swan occasionally dipping its head into the water to feed." \
#   --addnoise_condition 20 \
#   --offload \
#   --teacache \
#   --use_ret_steps \
#   --teacache_thresh 0.3


# CUDA_VISIBLE_DEVICES=1 python3 generate_video_df.py \
#   --model_id ${model_id} \
#   --resolution 480P \
#   --ar_step 2 \
#   --causal_block_size 1 \
#   --base_num_frames 81 \
#   --num_frames 243 \
#   --overlap_history 17 \
#   --prompt "Environmental basics:The scene is set in a vibrant, cartoonish industrial port area with bright blue water and wooden platforms. The environment features colorful crates, barrels, and shipping containers stacked around the area. A large wooden barrel is prominently placed on one of the platforms, and there are ramps and pathways connecting different sections. The background includes distant buildings and a clear sky, adding to the lively atmosphere.Main character:A small, anthropomorphic cat character is seen running across the wooden platforms and interacting with the environment. The cat appears agile and is navigating through the industrial setting, moving from one platform to another.Game mechanisms:The primary mechanism involves a large wooden barrel that rotates slowly on its axis. The barrel has a face drawn on it and is positioned on a raised platform. As the cat approaches, the barrel continues its rotation, creating a dynamic interaction point. The cat must time its movements carefully to avoid being hit by the barrel. Additionally, there are ramps and pathways that the cat uses to move between different levels of the environment, indicating a focus on navigation and timing.Weather and lighting:The weather is clear and sunny, with bright daylight illuminating the entire scene. The lighting is vivid, enhancing the colorful and cheerful aesthetic of the environment." \
#   --addnoise_condition 20 \
  #--offload

# model_id=/dockerdata/SkyReels-V2-I2V-1.3B-540P
# python3 generate_video.py \
#   --model_id ${model_id} \
#   --resolution 832 \
#   --num_frames 81 \
#   --guidance_scale 5.0 \
#   --shift 3.0 \
#   --fps 24 \
#   --prompt "一位女性，穿着浅粉色短裙和红色高跟鞋，留着长发，双手摆开，双腿并拢微微弯曲地站立" \
#   --image 'assets/ft_local/woman_front_whole_body_black_bg.png' \
#   --outdir 'i2v_1.3b_lora' 

# "一位女性，穿着红色格子花纹衬衫、白色短裤和白色平底鞋，留着长发，一只胳膊撑在另一胳膊上，站立姿态"
# "一位女性，穿着黑色西装西裤，双手交叉在胸前，笔直站立"
# "一位女孩，双马尾发型，穿着白色衬衣、黄色领带、黑色吊带裤和黑色皮鞋，笔直站立，双手自然下垂"
# "一位女性，穿着灰色西装和西裤，双手自然下垂，右手拿着笔记本电脑，笔直站立"
# "一位女性，穿着蓝色连衣裙，双手自然下垂，右手拿着笔记本电脑，笔直站立"
# "一位女性，穿着蓝色睡裙，右手抬起放在额头前，左手自然下垂，双腿并拢站立"
# "一位女性，穿着深蓝色西装和西裤，右手抬起放在额头前，左手自然下垂，双腿并拢站立"
# "一位女性，穿着绿色的长裙，双马尾头发，笔直站立，双臂自然下垂靠拢在身体两侧"
# "一位男性，穿着蓝色上衣和黑色裤子，双手张开举起与头部同高，双腿微弯曲站立


#model_id=/dockerdata/SkyReels-V2-I2V-1.3B-540P
model_id=/dockerdata/SkyReels-V2-I2V-14B-540P
python3 generate_video.py \
  --model_id ${model_id} \
  --resolution 832 \
  --num_frames 41 \
  --guidance_scale 3.0 \
  --shift 6 \
  --fps 15 \
  --prompt "一位女性，穿着苏格兰短裙，绿色夹克衫+白色长袖内衬衣，红色格子花纹短裙，黑色皮鞋，白色长筒袜；双马尾发型，笔直站立" \
  --image 'assets/ft_local/woman_dql_recloth.png' \
  --outdir 'i2v_14b_lora_0826' \
  --device 'cuda:0'