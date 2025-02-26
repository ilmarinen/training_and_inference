import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed TP Inference Example")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2-large",
        help="Path or name of the pre-trained model."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to video mp4 file."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
        help="Input prompt to generate text from."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Max generation length."
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=64,
        help="Max generation length."
    )
    return parser.parse_args()


def inference(model, processor, video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


def main():
    args = parse_args()

    # Initialize distributed environment.
    deepspeed.init_distributed()

    # Load tokenizer and model in FP16 (recommended for inference)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    result = inference(model, processor, args.video_path, args.prompt, max_new_tokens=args.max_length, total_pixels=(args.max_pixels * 28 * 28))
    print(result)

if __name__ == "__main__":
    main()
