from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch, base64
from io import BytesIO

app = FastAPI(title="SD-1.5 txt2img API")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
device    = "cuda" if torch.cuda.is_available() else "cpu"
dtype     = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None,          
).to(device)

pipe.unet = torch.compile(pipe.unet)          

pipe.enable_xformers_memory_efficient_attention()  

class Txt2ImgRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float   = 7.5
    height: int             = 512
    width: int              = 512
    seed: int | None        = None

@app.post("/txt2img")
async def txt2img(req: Txt2ImgRequest):
    gen = torch.Generator(device).manual_seed(req.seed) if req.seed else None

    image = pipe(
        req.prompt,
        height=req.height,
        width=req.width,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=gen,
    ).images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"prompt": req.prompt, "image_base64": b64}
