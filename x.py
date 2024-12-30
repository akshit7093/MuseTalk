import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import glob
import pickle
import copy
import argparse
from omegaconf import OmegaConf
from gfpgan import GFPGANer

from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image

def init_superres(method='GFPGAN'):
    model = GFPGANer(
        model_path='weights/GFPGANv1.4.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    return model

def enhance_face_region(frame, bbox, superres_model):
    x1, y1, x2, y2 = bbox
    face_region = frame[y1:y2, x1:x2]
    original_res = face_region.shape[0] * face_region.shape[1]
    target_res = (x2-x1) * (y2-y1)
    
    if original_res < target_res:
        _, _, enhanced_face = superres_model.enhance(
            face_region,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return cv2.resize(enhanced_face, (x2-x1, y2-y1))
    return face_region

@torch.no_grad()
def main(args):
    # Initialize models
    audio_processor, vae, unet, pe = load_all_model()
    superres_model = init_superres(args.superres)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)

    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    # Setup paths
    video_path = args.input_video
    audio_path = args.input_audio
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(result_dir, output_basename)
    os.makedirs(result_img_save_path, exist_ok=True)

    # Extract frames
    save_dir_full = os.path.join(result_dir, input_basename)
    os.makedirs(save_dir_full, exist_ok=True)
    cmd = f"ffmpeg -v fatal -i {video_path} -t 3 -start_number 0 {save_dir_full}/%08d.png"
    os.system(cmd)
    input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
    fps = get_video_fps(video_path)

    # Process audio
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)

    # Get landmarks and frames
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, args.bbox_shift)
    
    # Process latents
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # Create cyclic lists
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # Inference
    print("Starting inference...")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
    res_frame_list = []

    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = torch.from_numpy(whisper_batch).to(device=device, dtype=unet.model.dtype)
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    # Process and enhance frames
    print("Processing and enhancing frames...")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
            # Enhance the generated face region
            enhanced_face = enhance_face_region(res_frame, [0, 0, x2-x1, y2-y1], superres_model)
            # Combine with original frame
            ori_frame[y1:y2, x1:x2] = enhanced_face
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
        except:
            continue

    # Create final video
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -crf 18 temp.mp4"
    os.system(cmd_img2video)
    
    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {args.output}"
    os.system(cmd_combine_audio)

    # Cleanup
    os.remove("temp.mp4")
    os.system(f"rm -rf {save_dir_full} {result_img_save_path}")
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--superres", choices=['GFPGAN', 'CodeFormer'], default='GFPGAN')
    parser.add_argument("-iv", "--input_video", required=True)
    parser.add_argument("-ia", "--input_audio", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--use_float16", action="store_true")
    
    args = parser.parse_args()
    main(args)
