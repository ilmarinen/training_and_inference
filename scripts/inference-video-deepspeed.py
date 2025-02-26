import argparse
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        "--prompt",
        type=str,
        default="Once upon a time,",
        help="Input prompt to generate text from."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Max generation length."
    )
    parser.add_argument(
        "--mp_size",
        type=int,
        default=2,
        help="Tensor parallel degree (number of GPUs to split the model over)."
    )
    # Path to a DeepSpeed inference config file (see below)
    parser.add_argument(
        "--ds_config",
        type=str,
        default="ds_inference_config.json",
        help="Path to DeepSpeed inference config in JSON format."
    )
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank passed by DeepSpeed launcher (ignored).")
    return parser.parse_args()

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
    # Initialize model for inference with DeepSpeed.
    # The init_inference function accepts:
    #   mp_size: the tensor parallel degree,
    #   dtype: the model data type,
    #   replace_method: either "auto" or "jit" to replace parts of the model with optimized kernels,
    #   config: path to a JSON config file (or dict) for additional inference options.
    ds_inference_model = deepspeed.init_inference(
        model,
        replace_method="auto",  # uses kernel injection when available
        config=args.ds_config
    )

    def inference(video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
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
            output_ids = ds_inference_model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

    result = inference("./simple-man.mp4", "Please describe the video in detail", total_pixels=4 * 28 * 28)
    print(result)

if __name__ == "__main__":
    main()

