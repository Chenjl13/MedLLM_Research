import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

device = "cuda"

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./Qwen/Qwen2-VL-2B-Instruct",  
    torch_dtype=torch.float16,      
    device_map="auto"               
)

processor = AutoProcessor.from_pretrained("./Qwen/Qwen2-VL-2B-Instruct")

ft_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

ft_model_compress = Qwen2VLForConditionalGeneration.from_pretrained(
    "./Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype = torch.float16,
    device_map = "auto"
)

ft_model = PeftModel.from_pretrained(
    ft_model,
    "./output/Qwen2-VL-2B/checkpoint-248"  
)

ft_model_compress = PeftModel.from_pretrained(
    ft_model_compress,
    "./output/Qwen2-VL-2B-Compress/checkpoint-248"
)

ft_model.eval() 
ft_model_compress.eval()

image_path = "coco_2014_caption/558405.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    base_output = base_model.generate(**inputs, max_new_tokens=64)
    '''
    **input
    base_model.generate(
    input_ids=...,
    attention_mask=...,
    pixel_values=...
    )
    '''

base_text = processor.batch_decode(
    [base_output[0][len(inputs.input_ids[0]):]],  
    skip_special_tokens=True  
)[0]

with torch.no_grad():
    ft_output = ft_model.generate(**inputs, max_new_tokens=64)

ft_text = processor.batch_decode(
    [ft_output[0][len(inputs.input_ids[0]):]],
    skip_special_tokens=True
)[0]

with torch.no_grad():
    ft_output_compress = ft_model_compress.generate(**inputs, max_new_tokens=64)

ft_text_compress = processor.batch_decode(
    [ft_output_compress[0][len(inputs.input_ids[0]):]],
    skip_special_tokens=True
)[0]

print("\n========== Original ==========")
print(base_text)

print("\n========== After LoRA ==========")
print(ft_text)

print("\n========== After LoRA(Compressed) ==========")
print(ft_text_compress)