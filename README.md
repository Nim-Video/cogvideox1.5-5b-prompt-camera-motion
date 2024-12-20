# CogvideoX-5b LoRa to control camera movement

https://github.com/user-attachments/assets/7a8dc796-8c12-4570-bffb-285a64eec327

https://github.com/user-attachments/assets/5e463f82-3a08-4996-b204-a039b8763940

### <a href="https://huggingface.co/NimVideo/cogvideox1.5-5b-prompt-camera-motion">Download Lora weights</a>
### Usage
The LoRa was trained to control camera movement in 6 directions: `left`, `right`, `up`, `down`, `zoom_in`, `zoom_out`.
Start prompt with text like this:
```python
'Сamera moves to the {}...',
'Сamera is moving to the {}...',
'{} camera movement...',
'{} camera turn...',
```

### Inference examples
#### ComfyUI example
![prompt_motion_lora_example](https://github.com/user-attachments/assets/f5ab8663-e867-4ed6-831e-d04816f71be5)
<a href="https://huggingface.co/NimVideo/cogvideox1.5-5b-prompt-camera-motion/blob/main/cogvideox_1_5_5b_I2V_prompt_motion_lora_example.json">JSON File Example</a>
#### Minimal code example
```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX1.5-5B-I2V", torch_dtype=torch.bfloat16
)

pipe.load_lora_weights("NimVideo/cogvideox1.5-5b-prompt-camera-motion", adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0])

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

height = 768 
width = 1360
image = load_image("resources/car.jpg").resize((width, height))
prompt = "Camera is moving to the left. A red sports car driving on a winding road."

video_generate = pipe(
    image=image,
    prompt=prompt,
    height=height, 
    width=width, 
    num_inference_steps=50,  
    num_frames=81,  
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42), 
).frames[0]

export_to_video(video_generate, output_path, fps=8)
```

#### Inference with cli (Minimal parameters count)
```bash
python -m inference.cli_demo \
    --image_path "resources/car.jpg" \
    --prompt "Camera is moving to the left. A red sports car driving on a winding road." 
```

#### Inference with cli (Extended parameters count)
```bash
python -m inference.cli_demo \
    --image_path "resources/car.jpg" \
    --prompt "Camera is moving to the left. A red sports car driving on a winding road." \
    --width 1360 \
    --height 768 \
    --num_frames 81 \
    --num_inference_steps 50 \
    --guidance_scale 6 \
    --seed 42 \
    --model_path "THUDM/CogVideoX1.5-5B-I2V" \
    --lora_path "NimVideo/cogvideox1.5-5b-prompt-camera-motion" 
```

Inference with jupyter you can find in `inference/jupyter_inference_example.ipynb`. 

## Acknowledgements
Original code and models [CogVideoX](https://github.com/THUDM/CogVideo/tree/main).  

## Contacts
<p>Issues should be raised directly in the repository.</p>
