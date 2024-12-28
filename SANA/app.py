from flask import Flask, request
from PIL import Image
import  json, os
import torch

from pipes.sana_img2img import SanaPipelineImg2Img
from pipes.sana_pag import SanaPAGPipeline
from pipes.sana_pag_img2img import SanaPipelineImg2ImgPAG
import numpy as np
from signal import SIGKILL


params = {
    "steps":12,
    "cfg":5.0,
    "prompt":""
}



app = Flask(__name__)

embeds = {}
processid = ""
pid = os.getpid()
print("pid:   ",pid)
flag = True
output_image = ""

@app.route("/endApp",methods=["POST","GET"])
def endApp():
    global pid
    print("ending....: ",pid)
    os.kill(pid,SIGKILL)
    return {}

@app.route("/getSharedData",methods=["POST","GET"])
def getSharedData():
    global flag, output_image, embeds
    return {"flag":flag,"image":output_image,"embeds":embeds}

@app.route("/encode",methods=["POST","GET"])
def encode():
    global params, embeds, flag
    try:
        model_name = request.json["model"]
        pipe = SanaPipelineImg2Img.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            transformer=None,
            vae=None

        )
        pipe.to('cpu')
        prompt = request.json["prompt"]
        negative_prompt = request.json["negative_prompt"]
        embeds = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=2,
            textonly=True
        )
        data = {}
        for i in embeds:
            embs = json.dumps(embeds[i].type(torch.float32).numpy().astype(np.float32).tolist())
            data[i] = embs    
        embeds = data
        flag = False
    except:
        flag = False
    return {}


@app.route("/diffuse",methods=["POST","GET"])
def diffuse():
    global embeds,params, output_image, flag
    try:
        device = request.json["device"]
        if request.json["img2img"] == "enable":
            pipe = SanaPipelineImg2ImgPAG.from_pretrained(
                request.json["model"],
                torch_dtype = torch.float16 if device=="cuda" else torch.float32,
                text_encoder=None
            )
            if device=="cuda":
                pipe.enable_model_cpu_offload()
            image = pipe(
                prompt_embeds = torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_embeds"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_embeds"]))),
                prompt_attention_mask= torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_attention_mask"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_attention_mask"]))),
                negative_prompt_embeds=torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_embeds"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_embeds"]))),
                negative_prompt_attention_mask = torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_attention_mask"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_attention_mask"]))),
                height=int(request.json["height"]),
                width=int(request.json["width"]),
                guidance_scale=float(request.json["cfg"]),
                pag_scale=float(request.json["pag_scale"]),
                num_inference_steps=int(request.json["steps"]),
                image=Image.fromarray(np.array(json.loads(request.json["image"]),dtype="uint8")),
                strength=float(request.json["strength"])
            )[0]
        else:
            pipe = SanaPAGPipeline.from_pretrained(
                request.json["model"],
                torch_dtype = torch.float16 if device=="cuda" else torch.float32,
                text_encoder=None
            )
            if device=="cuda":
                pipe.enable_model_cpu_offload()
            image = pipe(
                prompt_embeds = torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_embeds"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_embeds"]))),
                prompt_attention_mask= torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_attention_mask"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_attention_mask"]))),
                negative_prompt_embeds=torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_embeds"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_embeds"]))),
                negative_prompt_attention_mask = torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_attention_mask"]))).half().to('cuda') if device=="cuda" else torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_attention_mask"]))),
                height=int(request.json["height"]),
                width=int(request.json["width"]),
                guidance_scale=float(request.json["cfg"]),
                pag_scale=float(request.json["pag_scale"]),
                num_inference_steps=int(request.json["steps"]),
            )[0]
        output_image = json.dumps(np.array(image[0]).tolist())
        flag = False
    except:
        flag = False
    return{}




if __name__ == "__main__":
    app.run()
