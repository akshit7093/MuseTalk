# MuseTalk with Enhanced Super-Resolution

This project enhances the MuseTalk repository by integrating super-resolution using **GFPGAN** and **CodeFormer**. The modifications ensure that only the generated parts of a video frame are enhanced, improving the output quality while maintaining efficiency.

---

## Prerequisites

- **Python Version**: 3.9.12  
- **Dependencies**: All dependencies are listed in `requirements.txt`.

---

## Installation and Setup

### 1. Install Dependencies
Run the following command to install all required dependencies:
```bash
pip install --no-cache-dir -r requirements.txt
```

### 2. Install FFmpeg
1. Download the static build of FFmpeg from [here](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip).
2. Extract the downloaded ZIP file to your MuseTalk project directory, renaming the folder to:
   ```
   ffmpeg-7.0.2-amd64-static
   ```
3. Update the environment variables:
   - Add the `ffmpeg-7.0.2-amd64-static/bin` directory to your `PATH`.
   - Set the `FFMPEG_PATH` environment variable to point to the FFmpeg executable.

4. Install the FFmpeg Python wrapper:
   ```bash
   pip install ffmpeg-python
   ```

---

### 3. Install Additional Packages
Run the following commands to install additional required packages:
```bash
pip install --no-cache-dir -U openmim
mim install --no-cache-dir -U mmengine
mim install --no-cache-dir -U "mmcv==2.0.1"
mim install --no-cache-dir -U "mmdet>=3.1.0"
mim install --no-cache-dir -U "mmpose>=1.1.0"
pip install huggingface_hub==0.25.2
```

---

### 4. Download Models
1. Run the following command to initialize the models:
   ```bash
   python app.py
   ```
2. Ensure your directory structure for models is as follows:
   ```
   ./models/
   ├── musetalk
   │   └── musetalk.json
   │   └── pytorch_model.bin
   ├── dwpose
   │   └── dw-ll_ucoco_384.pth
   ├── face-parse-bisent
   │   ├── 79999_iter.pth
   │   └── resnet18-5c106cde.pth
   ├── sd-vae-ft-mse
   │   ├── config.json
   │   └── diffusion_pytorch_model.bin
   └── whisper
       └── tiny.pt
   ```

3. Test the installation with the following command:
   ```bash
   python -m scripts.inference --inference_config configs/inference/test.yaml
   ```

---

### 5. Setup GFPGAN for Super-Resolution
1. Install the required libraries:
   ```bash
   pip install gfpgan
   pip install basicsr
   pip install facexlib
   ```
2. Create a directory for weights:
   ```bash
   mkdir weights
   cd weights
   ```
3. Download the GFPGAN model weights:
   ```powershell
   Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile "experiments/pretrained_models/GFPGANv1.4.pth"
   ```

---

## Running the Enhanced Script

Use the following command to process videos with super-resolution:
```bash
python x.py --superres [GFPGAN/CodeFormer] -iv input.mp4 -ia input.mp3 -o output.mp4
```
- Replace `[GFPGAN/CodeFormer]` with the desired super-resolution method.
- `input.mp4`: Input video file.
- `input.mp3`: Input audio file.
- `output.mp4`: Enhanced output video.

---

## Testing the Setup
1. Use a 3-second test video and audio for faster testing.
2. Ensure that super-resolution is applied only to the generated frame portions, as described in the script.

---

## Notes
- Ensure that your Python version is exactly **3.9.12** for compatibility.
- For any issues, verify that all models and weights are downloaded and properly structured.

Enjoy enhancing your videos with MuseTalk and advanced super-resolution!
