from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline, AutoPipelineForText2Image
from argparse import ArgumentParser
import os
import imageio
import torch
from PIL import Image

from huggingface_hub import login


def load_pipelines(cache_dir, device):
    sd_turbo_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", cache_dir=cache_dir).to(device)
    svd_pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", cache_dir=cache_dir).to(device)

    return sd_turbo_pipe, svd_pipe

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--prompt_file", type=str, default="prompts/prompts_list.txt")
    parser.add_argument("--cache_dir", "-c", type=str, default="/workspace/models")
    args = parser.parse_args()
    
    result_dir = "./results"

    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    sd_turbo_pipe, svd_pipe = load_pipelines(args.cache_dir, device)

    with open(args.prompt_file, "r") as f:
        prompt_list = f.readlines()
    
    print(len(prompt_list), "prompts found")
    
    for prompt in prompt_list:
        image_path = os.path.join(result_dir, prompt + ".png")
        image = sd_turbo_pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

        image.save(image_path)

        image = Image.open(image_path)
        video_path = os.path.join(result_dir, prompt + ".gif")
        video = svd_pipe(image=image,
                        num_frames=25,
                        height=image.height,
                        width=image.width,
                        num_inference_steps=30,
                        ).frames[0]

        imageio.mimsave(video_path, video, duration=143)
    