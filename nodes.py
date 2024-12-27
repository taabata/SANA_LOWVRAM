import json
import numpy as np
import folder_paths
import os
import torch
import subprocess, requests, time



class SANADiffuse:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {	
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {
                            "default": 4,
                            "min": 0,
                            "max": 360,
                            "step": 1,
                        }),
                        
                        "width": ("INT", {
                            "default": 512,
                            "min": 0,
                            "max": 5000,
                            "step": 64,
                        }),
                        "height": ("INT", {
                            "default": 512,
                            "min": 0,
                            "max": 5000,
                            "step": 64,
                        }),
                        "cfg": ("FLOAT", {
                            "default": 8.0,
                            "min": 0,
                            "max": 30.0,
                            "step": 0.1,
                        }),
                        "pag_scale": ("FLOAT", {
                            "default": 2.0,
                            "min": 0,
                            "max": 30.0,
                            "step": 0.1,
                        }),
                        "img2img":(["disable","enable"],),
                        "embeds":("class",),
                        "model_path": ([f'diffusers/{i}' for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}")],),
                    },
                "optional":
                    {
                        "image": ("IMAGE", ),
                        "strength": ("FLOAT", {
                            "default": 1.0,
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.01,
                        }),
                    }
                }

    CATEGORY = "sana/nodes"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sana"
    
    def sana(self, seed,steps,width,height,cfg,pag_scale,img2img,embeds,model_path, strength = None, image=None):
        subprocess.Popen(["python3","app.py"],cwd=os.path.join(os.path.dirname(os.path.realpath(__file__)),"SANA"))
        flag = True
        time.sleep(5)
        if img2img == "enable":
            image = image[0].numpy()
            image = image*255.0
            image = json.dumps(np.array(image).tolist())
        data = {
            "steps":steps,
            "width":width,
            "height":height,
            "pag_scale":pag_scale,
            "seed":seed,
            "cfg":cfg,
            "embeds":embeds,
            "img2img":img2img,
            "image":image,
            "strength":strength,
            "model":os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+f"/models/{model_path}"
        }
        req = requests.post("http://127.0.0.1:5000/diffuse",json=data)
        while flag:
            data = requests.get("http://127.0.0.1:5000/getSharedData").json()
            flag = bool(data["flag"])
            if flag==False:
                image = np.array(json.loads(data["image"]),dtype="uint8")
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                break
            time.sleep(1)
        try:
            requests.get("http://127.0.0.1:5000/endApp")
        except:
            pass
        return (image,)
    


class SANATextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {	
                        "prompt": ("STRING", {"default": '', "multiline": True}),
                        "negative_prompt": ("STRING", {"default": '', "multiline": True}),
                        "model_path": ([f'diffusers/{i}' for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}")],),
                    }
                }

    CATEGORY = "sana/nodes"

    RETURN_TYPES = ("class",)
    FUNCTION = "sana"
    
    def sana(self,prompt,negative_prompt,model_path):
        subprocess.Popen(["python3","app.py"],cwd=os.path.join(os.path.dirname(os.path.realpath(__file__)),"SANA"))
        time.sleep(5)
        data = {
            "prompt":prompt,
            "negative_prompt":negative_prompt,
            "model":os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+f"/models/{model_path}"
        }
        requests.post("http://127.0.0.1:5000/encode",json=data)
        flag = True
        time.sleep(5)
        embeds = ''
        while flag:
            data = requests.get("http://127.0.0.1:5000/getSharedData").json()
            flag = bool(data["flag"])
            if flag==False:
                embeds = data["embeds"]
                break
            time.sleep(1)
        try:
            requests.get("http://127.0.0.1:5000/endApp")
        except:
            pass
        return (embeds,)

NODE_CLASS_MAPPINGS = {
    "SANADiffuse":SANADiffuse,
    "SANATextEncode":SANATextEncode
}