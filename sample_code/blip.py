from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from accelerate import Accelerator, infer_auto_device_map, dispatch_model
import sys
import pandas as pd
import time


# load the model
def get_blip_model(device='cuda', dtype=torch.bfloat16, use_multi_gpus=True):
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", offload_folder = 'off', cache_dir="/work/iuy704/hfcache")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
    #model.to(dtype)
    if use_multi_gpus:
        device_map = infer_auto_device_map(model, max_memory={0: "28GiB", 1: "28GiB"},no_split_module_classes=['InstructBlipVisionModel', 'InstructBlipQFormerModel', 'LlamaDecoderLayer'])
        device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model.embed_tokens')]
        print(device_map)
        model = dispatch_model(model, device_map=device_map, offload_dir = 'off')
        torch.cuda.empty_cache()
    else:
        model.to('cuda:0')
        model.eval()
        model = torch.compile(model)
    return model, processor
model, processor = get_blip_model() 

# load the data
df = pd.read_csv("ugcg.csv")

#prompt = "Is the an unsafe image? An unsafe image may contain graphic violence, pornography, profanity, nudity, slurs, or other potentially disturbing subject matter."

Condition = "This is an image generated from a role-playing game. View the given image and carefully answer the questions in each step:"
Q1 = "how many characters or avatars are present in this image?"
Q2A = "Are the characters or avatars naked?"
Q2B = "Are there anuses, female breasts, female genitals, or male genitals exposed in this image?"
Q2C = "Are the characters or avatars indulging in any sexual activities?"
Q3A = "Are there any knives, guns, or deadly weapons presented in this image?"
Q3B = "Are the characters or avatars indulging in any violent activities, such as murder, suicide, and torture?"
Q4 = "Are the characters or avatars indulging in any other harmful activies, such as smoking, self-harming, or using drugs?"

ugcgcot = {'condition': Condition, 'Q1': Q1, 'Q2A': Q2A, 'Q2B': Q2B, 'Q2C': Q2C, 'Q3A': Q3A, 'Q3B': Q3B, 'Q4': Q4}

def blip_output(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    device = "cuda"
    answers = {'Q1': '', 'Q2A': '', 'Q2B': '','Q2C': '','Q3A': '', 'Q3B': '', 'Q4': ''}
    for prompt_name, prompt in ugcgcot.items():
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            #num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        answers[prompt_name] = generated_text
    return answers

# generate the text
df['blip_output'] = df.apply(lambda x: blip_output(x['image_path'], ugcgcot), axis=1)
# save the output
df.to_csv("blip_output.csv")