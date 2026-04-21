import argparse
import numpy as np
import torch
import cv2
import os  # 新增
from tqdm import tqdm # 新增
from model import SketchKeras
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    # 修改: help 信息表明是文件夹
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image FOLDER")
    parser.add_argument(
        "--output", "-o", type=str, default="output_sketches", help="Output image FOLDER"
    )
    parser.add_argument(
        "--weight", "-w", type=str, default="/home/sty/pyfile/sketchKeras_pytorch/weights/model.pth", help="weight file"
    )
    return parser.parse_args()


def preprocess(img):
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(int) - blurred.astype(int)
    highpass = highpass.astype(float) / 128.0
    
    max_val = np.max(highpass)
    if max_val > 0:
        highpass /= max_val

    ret = np.zeros((512, 512, 3), dtype=float)
    ret[0:h, 0:w, 0:c] = highpass
    return ret


def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


if __name__ == "__main__":
    args = parse_args()

    input_dir = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in supported_extensions]
    
    if not image_files:
        print(f"Error: No supported image files found in '{input_dir}'.")
        exit()
        
    print(f"Found {len(image_files)} images in '{input_dir}'. Processing...")

    model = SketchKeras().to(device)
    if os.path.exists(args.weight):
        model.load_state_dict(torch.load(args.weight))
        print(f"Weight '{args.weight}' loaded.")
    else:
        print(f"Warning: Weight file not found at '{args.weight}'. Using a randomly initialized model.")
    model.eval()

    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)
        
        if img is None:
            print(f"\nWarning: Could not read image, skipping: {input_path}")
            continue

        height, width = float(img.shape[0]), float(img.shape[1])
        if width > height:
            new_width, new_height = (512, int(512 / width * height))
        else:
            new_width, new_height = (int(512 / height * width), 512)
        img_resized = cv2.resize(img, (new_width, new_height))
        
        img_preprocessed = preprocess(img_resized)
        x = img_preprocessed.reshape(1, *img_preprocessed.shape).transpose(3, 0, 1, 2)
        x = torch.tensor(x).float()
        
        with torch.no_grad():
            pred = model(x.to(device))
        pred = pred.squeeze()
        
        output = pred.cpu().detach().numpy()
        output = postprocess(output, thresh=0.18, smooth=False) 
        output = output[:new_height, :new_width]

        cv2.imwrite(output_path, output)

    print(f"\nProcessing complete! All results saved to '{output_dir}'.")