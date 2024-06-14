# from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline, AutoPipelineForText2Image
from svd_motion.pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from svd_motion.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from argparse import ArgumentParser
import os
import imageio
import torch
from PIL import Image

from huggingface_hub import login


def load_pipelines(cache_dir, device):
    svd_pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", cache_dir=cache_dir).to(device)
    svd_pipe.unet = UNetSpatioTemporalConditionModel.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", cache_dir=cache_dir, subfolder="unet").to(device)

    return svd_pipe

if __name__ == "__main__":
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = ArgumentParser()

    parser.add_argument("--cache_dir", "-c", type=str, default="/workspace/models")
    parser.add_argument("--prompt", type=str, default="An astronaut floating in space with a view of the Earth in the background")
    args = parser.parse_args()
    
    result_dir = "./results"

    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    svd_pipe = load_pipelines(args.cache_dir, device)
    

    image_path = os.path.join(result_dir, args.prompt + ".png")

    image = Image.open(image_path)
    video_path = os.path.join(result_dir, args.prompt + ".gif")
    video = svd_pipe(image=image,
                    num_frames=25,
                    height=image.height,
                    width=image.width,
                    num_inference_steps=30,
                    ).frames[0]

    imageio.mimsave(video_path, video, duration=143)
    