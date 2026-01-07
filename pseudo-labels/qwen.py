from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Tuple, Optional

class Qwen:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device="cuda"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def forward(self, image: Image.Image) -> Tuple[str, Optional[str]]:
        prompt_1 = """
        Your Role: Object Presence Recognizer

        You are a model that checks whether a clearly visible object exists in an image.

        [Your task]
        - Look at the image.
        - If there is at least one clearly visible object, answer: Yes
        - If there is no visible object at all (only blurred or grayscale areas), answer: No
        - If it's hard to tell whether something is visible or not, answer: Unsure

        [Important rules]
        - Ignore blurred or grayscale areas in the image.
        - Only consider clear, colorful, or sharply defined objects.

        Your response must be only one word: Yes, No, or Unsure.

        [Examples]
        Example 1:  
        Image: (A color photo of a dog standing clearly in focus)  
        Answer: Yes

        Example 2:  
        Image: (A grayscale image with blurred outlines and no clear shapes)  
        Answer: No

        Example 3:  
        Image: (An image where a part of an object might be present, but it is not fully visible or too unclear) 
        Answer: Unsure

        Now, analyze the following image:  
        Image: <attach image here>  
        Answer:
        """

        exists_response = self._run_qwen(image, prompt_1)

        if "yes" in exists_response.strip().lower():
            prompt_2 = """
            Your Role: Object Category Identifier

            You are a model that identifies the most likely object category that is clearly visible in an image.

            [Your task]
            - Look at the image.
            - Focus only on areas that are clear, colorful, and sharply defined.
            - Completely ignore grayscale or blurred areas.
            - Always guess the most likely object category that is clearly visible.

            [Instructions]
            - Answer with only one or two words.
            - Do not describe scenes -- just the object category.
            - If uncertain, make your best guess based on visible clues.

            [Examples]
            Example 1:  
            Image: (A focused image of a person riding a skateboard)  
            Answer: Skateboard

            Example 2:  
            Image: (A clear image of a zebra walking in grass)  
            Answer: Zebra

            Example 3:  
            Image: (Blurry background, but a sharp image of a backpack is visible)  
            Answer: Backpack

            Now analyze the following image:  
            Image: <attach image here>  
            Answer:
            """
            category_response = self._run_qwen(image, prompt_2)

            prompt_3 = """
            Your Role: Foreground-Background Distinguisher

            You are a model that determines whether an object in an image is part of the foreground or the background.

            [Your task]
            - You are given an object name: <Response>
            - Look at the image and decide if this object is in the foreground or background.
            - Ignore any grayscale or blurred areas in the image.
            - Use visual focus and typical object roles to decide.

            [Definitions]
            - Foreground = clearly focused subjects like people, animals, vehicles, or objects of interest.
            - Background = things like sky, grass, trees, mountains, or flowers.

            Your answer must be exactly one word: Foreground or Background.

            [Examples]
            Example 1:  
            Object: Dog
            Image: (A dog is standing in sharp focus in front of a blurry park)  
            Answer: Foreground

            Example 2:  
            Object: Sky
            Image: (A person is standing in front of a bright blue sky)  
            Answer: Background

            Example 3:  
            Object: Tree
            Image: (A clear person in front, with trees in the back)  
            Answer: Background

            Now analyze the following image:  
            Object: <Response>
            Image: <attach image here>  
            Answer:
            """
            fgbg_response = self._run_qwen(image, prompt_3)

            if fgbg_response.lower() in 'foreground':
                return "yes", category_response.strip().lower()
            else:
                return "no", category_response.strip().lower()
        else:
            return "Unsure", None


    def _run_qwen(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()
