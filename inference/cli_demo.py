"""
Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python -m inference.cli_demo \
--image_path "resources/car.jpg" \
--prompt "Camera is moving to the left. A red sports car driving on a winding road. The car is a futuristic design with sharp angles and lines. It has large wheels and a spoiler on the back. The car is driving fast, and the road is blurred behind it. The car is driving through a mountain pass. The mountains are in the background, and the sun is setting behind them. The sky is a beautiful orange color, and the clouds are dark and dramatic. The car is a perfect example of speed and power. The image is well-composed and captures the beauty of the car and the setting. The lighting is excellent, and the details of the car are clear and sharp." \
--model_path THUDM/CogVideoX1.5-5B-I2V \
--lora_path NimVideo/cogvideox1.5-5b-prompt-camera-motion
```

Additional options are available to specify the guidance scale, number of inference steps, video generation type, and output paths.
"""
import argparse

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


@torch.no_grad()
def generate_video(
    prompt: str,
    image_path: str,
    model_path: str,
    width: int = 768,
    height: int = 1360,
    lora_path: str = None,
    lora_weight: float = 1.0,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    num_frames: int = 81,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - image_path (str): The video for controlnet processing.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_weight (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames: (int): Number output frames.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    )
    image = load_image(image_path)
    
    if lora_path:
        pipe.load_lora_weights(lora_path, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_weight])

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    video_generate = pipe(
        image=image,
        prompt=prompt,
        height=height, # 480, 768
        width=width, # 720, 1360
        num_videos_per_prompt=num_videos_per_prompt, 
        num_inference_steps=num_inference_steps,  
        num_frames=num_frames,  
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed), 
    ).frames[0]

    export_to_video(video_generate, output_path, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX1.5-5B-I2V", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, default="NimVideo/cogvideox1.5-5b-prompt-camera-motion", help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_weight", type=float, default=1.0, help="The weight of the LoRA")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--height", type=int, default=768, help="Output video height")
    parser.add_argument("--width", type=int, default=1360, help="Output video width")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of frames for generated video"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        image_path=args.image_path,
        model_path=args.model_path,
        width=args.width,
        height=args.height,
        lora_path=args.lora_path,
        lora_weight=args.lora_weight,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
    )