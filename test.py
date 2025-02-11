import os
import cv2 as cv
import torch
import numpy as np
import time
from model.MFDNet import MFDNet
from model.Plaindn import PlainDN
from train import opt
from utils.Rep_params import rep_params


def set_device():
    """Set up the computing device (GPU or CPU)."""
    use_cuda = opt.cuda >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{opt.cuda}' if use_cuda else 'cpu')

    if use_cuda:
        print(f"Using CUDA (GPU {opt.cuda}) for acceleration.")
        torch.cuda.set_device(opt.cuda)
        torch.backends.cudnn.benchmark = True  # Optimizes for fixed-size inputs
    else:
        print("Using CPU for training/testing.")

    return device


def process_video(model_plain, device):
    """Process video frames and apply the model."""
    video_path = './test_video2.mp4'
    output_path = './output_video2.mp4'

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5))

    # Video writer setup
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, 
                         (frame_width, frame_height), isColor=True)

    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert frame to YCrCb and extract channels
        frame_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        y_channel, crcb_channel = frame_ycrcb[:, :, 0], frame_ycrcb[:, :, 1:]

        # Resize and normalize Y channel
        y_channel_resized = cv.resize(y_channel, (90, 120)).astype(np.float32)
        y_tensor = torch.tensor(y_channel_resized).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model_plain(y_tensor).squeeze(0).cpu().numpy()

        # Clip and format output
        result = np.clip(result, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        result = cv.resize(result, (frame_width, frame_height))

        # Merge processed Y with original CrCb
        final_frame = cv.merge([result, crcb_channel])
        final_frame = cv.cvtColor(final_frame, cv.COLOR_YCrCb2BGR)

        out.write(final_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    print(f"Processed video saved at {output_path}")


def process_images(model, device):
    """Process individual images and save output."""
    if not os.path.exists(opt.test_output):
        os.makedirs(opt.test_output)

    img_files = os.listdir(opt.test_input)

    for img_file in img_files:
        img_path = os.path.join(opt.test_input, img_file)
        img = cv.imread(img_path)

        if img is None:
            print(f"Warning: Unable to read {img_file}, skipping...")
            continue

        # Convert image to YCrCb and extract channels
        img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        y_channel, crcb_channel = img_ycrcb[:, :, 0], img_ycrcb[:, :, 1:]

        # Normalize and convert to tensor
        y_tensor = torch.tensor(y_channel.astype(np.float32) / 255).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()
            result = model(y_tensor).squeeze(0).cpu().numpy()
            print(f"Processed {img_file} in {time.time() - start_time:.4f} seconds")

        # Clip and format output
        result = np.clip(result * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        # Merge processed Y with original CrCb
        final_image = cv.merge([result, crcb_channel])
        final_image = cv.cvtColor(final_image, cv.COLOR_YCrCb2BGR)

        # Save output
        output_path = os.path.join(opt.test_output, img_file)
        cv.imwrite(output_path, final_image)

    print("Image processing complete.")


def main():
    device = set_device()

    # Load models
    print(f"Loading model: {opt.test_model}")
    model = MFDNet(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
    model_plain = PlainDN(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)

    try:
        model.load_state_dict(torch.load(opt.test_model, map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Reconstruct parameters
    model_plain = rep_params(model, model_plain, opt, device)

    # Set models to evaluation mode
    model.eval()
    model_plain.eval()

    # Select processing mode (video or images)
    process_video_flag = False

    if process_video_flag:
        process_video(model_plain, device)
    else:
        process_images(model, device)

    # Save restructured model (uncomment if needed)
    # torch.save(model_plain.state_dict(), "./weights/mfdnet_plain/mfdnet_plain_sigma11.pt")


if __name__ == "__main__":
    main()
