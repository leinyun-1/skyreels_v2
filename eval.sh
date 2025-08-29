#model_id=/dockerdata/SkyReels-V2-I2V-1.3B-540P
model_id=/dockerdata/SkyReels-V2-I2V-14B-540P
python3 tasks/eval.py \
  --model_id ${model_id} \
  --resolution 832 \
  --num_frames 41 \
  --guidance_scale 3.0 \
  --shift 6 \
  --fps 15 \
  --prompt 'assets/ft_local/prompts.txt' \
  --image 'assets/ft_local/'  \
  --outdir 'i2v_14b_lora_0826/t6000' \
  --device 'cuda:0'