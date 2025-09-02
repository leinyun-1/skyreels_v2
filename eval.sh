model_id=/dockerdata/SkyReels-V2-I2V-1.3B-540P
#model_id=/dockerdata/SkyReels-V2-I2V-14B-540P
python3 tasks/eval.py \
  --model_id ${model_id} \
  --resolution 832 \
  --num_frames 81 \
  --guidance_scale 3.0 \
  --shift 6 \
  --fps 15 \
  --lora_path 'tasks_out/train_exp/0731_wan_i2v_lora_thuman_round/lightning_lora_ckpts/lora-epoch=18-step=004000.ckpt' \
  --prompt 'assets/eval_examples_1/prompts_qwen.txt' \
  --image 'assets/eval_examples_1/'  \
  --outdir 'eval/i2v_1.3b_lora_ortho' \
  --device 'cuda:0'