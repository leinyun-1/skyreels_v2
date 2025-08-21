import os
from rembg import remove
from PIL import Image
import io
import imageio.v2 as imageio  # 使用 v2 接口，避免未来版本警告

def remove_background():
    # 路径设置
    input_path = '/mnt/aigc_cq/private/leinyu/code/skyreels_v2/assets/ft_local/woman_5_recloth.png'
    output_path = input_path

    # Step 1: 去除背景（得到透明PNG）
    with open(input_path, 'rb') as f:
        input_data = f.read()
    output_data = remove(input_data)

    # Step 2: 加载透明PNG图像
    input_image = Image.open(io.BytesIO(output_data)).convert("RGBA")

    # Step 3: 创建黑色背景图像
    black_bg = Image.new("RGB", input_image.size, (0, 0, 0))  # 黑色背景

    # Step 4: 将原图粘贴到黑色背景上，使用透明度作为mask
    black_bg.paste(input_image, mask=input_image.split()[3])  # 使用 alpha 通道作为 mask

    # Step 5: 保存为 JPG（不带透明通道）
    #black_bg.save(output_path, format="JPEG")

    width, height = black_bg.size
    target_size = max(width, height)
    target_width, target_height = target_size, target_size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = left + target_width
    bottom = top + target_height
    img_cropped = black_bg.crop((left, top, right, bottom))
    img_cropped.save(output_path)

    print(f"完成，背景为黑色的图片保存在：{output_path}")


def slice_video():
    input_path = 'result/i2v_1.3b_lora/一位女孩，双马尾发型，穿着白色衬衣、黄色领带、黑色吊带裤和黑色皮鞋，笔直站立，双手自然下垂_711153487_2025-08-05_21-39-51.mp4'
    output_dir = 'result/i2v_1.3b_lora/sliced'
    os.makedirs(output_dir, exist_ok=True)

    ### 使用 imageio 读取输入视频，每4帧取一帧，保存为新视频
    reader = imageio.get_reader(input_path)          # 打开输入视频
    fps = reader.get_meta_data()['fps']              # 获取原始帧率
    frames = []                                      # 存放抽帧后的图像

    for idx, frame in enumerate(reader):             # 逐帧读取
        if idx % 4 == 0:                             # 每4帧取一帧
            frames.append(frame)                     # 收集帧
    reader.close()                                   # 关闭读取器

    # 输出文件路径
    out_path = os.path.join(output_dir, 'downsampled.mp4')
    writer = imageio.get_writer(out_path, fps=1)  
    for f in frames:
        writer.append_data(f)                        # 写入输出视频
    writer.close()                                   # 关闭写入器

    print(f"完成，每4帧取一帧的视频已保存至：{out_path}")

def black_pad():
    image_path = 'assets/ft_local/woman_1_black_bg.png'
    img = Image.open(image_path)
    width, height = img.size
    target_size = max(width, height)
    target_width, target_height = target_size, target_size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = left + target_width
    bottom = top + target_height
    img_cropped = img.crop((left, top, right, bottom))
    img_cropped.save(image_path.replace('.png', '_cropped.png'), format='PNG')

if __name__ == "__main__":
    remove_background()
    #slice_video()
    #black_pad()
