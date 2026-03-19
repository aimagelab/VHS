import os
from importlib_metadata import files
from .geneval_utils import QwenFeedback
from vhs.model_loader import create_model_from_args
from vhs.mm_utils import process_images, tokenizer_image_token
from vhs.train.preprocessing import preprocess_phi4, preprocess_gemma_2, preprocess_qwen_2
from vhs.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vhs import conversation as conversation_lib
from vhs.model.language_model.llava_qwen import LlavaQwenForRegression
import torch

class LatentGemmaFeedback(QwenFeedback):
    def __init__(self, device, path, vision_tower="dummy", greedy=False, check_sentence="is correct", lora_weights=None, mm_projector_type="mlp2x_gelu", **kwargs):
        v_t = "hidden" if "hidden" in vision_tower else vision_tower
        if "hidden" in vision_tower:
            if "concatenated" in vision_tower:
                mm_vision_select_feature = "concatenated"
            elif not "average" in vision_tower:
                mm_vision_select_feature = f"block_{vision_tower.split('_')[-1]}"
            else:
                mm_vision_select_feature="average"
        else:
            mm_vision_select_feature="patch"
        hidden_dim = kwargs.get("hidden_dim", 2240)
        normalization_mean_path = kwargs.get("normalization_mean_path", None)
        normalization_variance_path = kwargs.get("normalization_variance_path", None)
        if lora_weights is None:
            lora_weights=""
        self.model, self.tokenizer, image_processor = create_model_from_args(
            model_name_or_path=path,
            vision_tower=v_t,
            llm_backbone="qwen",
            vae_image_size=1024,
            #version="gemma_2",
            version="qwen_2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            bf16=True,
            model_max_length=4096,
            use_cache=False,
            #pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
            mm_projector_type=mm_projector_type,
            mm_vision_select_layer=-2,
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
            image_aspect_ratio="pad",
            lora_enable=bool(lora_weights),
            lora_weights=lora_weights,
            swap="swap" in lora_weights or "attn" in lora_weights,
            flatten_vae_output=False if "vit" in mm_projector_type.lower() or "sdxl" in vision_tower.lower() else True, 
            mm_vision_select_feature = mm_vision_select_feature,
            latent_layer = mm_vision_select_feature,
            mm_vision_normalize=normalization_mean_path is not None and normalization_variance_path is not None,
            hidden_dim=hidden_dim,
            normalization_mean_path=normalization_mean_path,
            normalization_variance_path=normalization_variance_path,
        )
        self.preprocess_function = preprocess_qwen_2
        conversation_lib.default_conversation = conversation_lib.conv_llava_qwen_2
        
        self.model.eval()
        self.processor = image_processor
        self.device = device
        self.greedy = greedy
        self.model.get_model().mm_projector.to(self.model.dtype)
        self.check_sentence = check_sentence
        self.is_hidden = "hidden" in vision_tower
        if kwargs.get("regression"):
            from safetensors.torch import load_file
            self.model.__class__ = LlavaQwenForRegression
            self.model.regression_head = torch.nn.Linear(self.model.config.hidden_size, 1).to(
                device=self.model.device, dtype=self.model.dtype
            )
            # Load regression head weights from the same checkpoint
            safetensors_path = os.path.join(path, "model.safetensors")
            checkpoint = load_file(safetensors_path, device=str(self.model.device))
            regression_state = {
                k.replace("regression_head.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("regression_head.")
            }
            if regression_state:
                self.model.regression_head.load_state_dict(regression_state)
                print(f"Loaded regression head weights from {safetensors_path}")
            else:
                print(f"WARNING: No regression_head weights found in {safetensors_path}")
            del checkpoint
    def build_message(self,prompt):
        messages=[
            { "from": "human", "value": f"<image>\nPlease evaluate this generated image based on the following prompt: {prompt}. Focus on text alignment and compositionality." },
            { "from": "gpt", "value": "" } 
            ]
        return messages
    
    def evaluate_image(self,imgs, metadata):
        messages = [self.build_message(metadata['prompt'])]
        input_ids = self.preprocess_function(messages, self.tokenizer, has_image=True, train=False)['input_ids'].cuda()
        if not isinstance(imgs, torch.Tensor):
            from PIL import Image
            imgs=Image.open(imgs)
            imgs = self.processor(imgs, return_tensors='pt')['pixel_values'][0].to(self.model.dtype).unsqueeze(0)
        if self.is_hidden:
            imgs = self.processor(imgs, return_tensors='pt')['pixel_values'][0].to(self.model.dtype).squeeze(0)
        self.model.get_model().mm_projector = self.model.get_model().mm_projector.to(imgs.dtype)
        res = self.get_randomized_feedback(input_ids, imgs)
        correct = self.check_sentence in res
        if not correct:
            reason = res.split('.')[-1].strip()
        else:
            reason = ''
        return dict(
            correct=correct,
            text_feedback=reason
        )
    def get_randomized_feedback(self, input_ids, imgs=None, temperature=None):
        with torch.no_grad():
            if type(self.model) == LlavaQwenForRegression:
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    images=imgs,
                    #num_beams=3,
                    #max_new_tokens=100
                )
                
            else:
                generated_ids = self.model.generate(
                    inputs=input_ids,
                    images=imgs,
                    num_beams=3,
                    max_new_tokens=100
                    )
            output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            res = output_text[0]
            return res
    



class LatentGemmaVerifier(LatentGemmaFeedback):
    def __init__(self, device, path, vision_tower="dummy", greedy=False, check_sentence="is correct", lora_weights=None, use_generation_prompt=False, mm_projector_type="mlp2x_gelu", **kwargs):

        super().__init__(device, path, vision_tower, greedy, check_sentence, lora_weights, mm_projector_type=mm_projector_type, **kwargs)
        self.yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        self.use_generation_prompt = use_generation_prompt
        
    def build_message(self,prompt):
        messages=[
            {"from": "human", "value": f"You are an AI assistant specializing in image analysis and ranking. Your task is to analyze and compare image based on how well they match the given prompt. <image> The given prompt is: {prompt}. Please consider the prompt and the image to make a decision and response directly with 'yes' or 'no'"},
            { "from": "gpt", "value": "" } 
            ]
        messages = [messages[0]] if not self.use_generation_prompt else messages
        return messages
    
    def evaluate_image(self,imgs, metadata):
        messages = [self.build_message(metadata['prompt'])]
        input_ids = self.preprocess_function(messages, self.tokenizer, has_image=True, train=False)['input_ids'].cuda()
        if not isinstance(imgs, torch.Tensor):
            from PIL import Image
            if not isinstance(imgs, list):
                if not isinstance(imgs, Image.Image):
                    imgs=Image.open(imgs)
                imgs = self.processor(imgs, return_tensors='pt')['pixel_values'].to(self.model.dtype).unsqueeze(0)
            else:
               if isinstance(imgs[0], Image.Image):
                    imgs = self.processor(imgs, return_tensors='pt')['pixel_values'][0].to(self.model.dtype).unsqueeze(0)
        if self.is_hidden:
            imgs = self.processor(imgs, return_tensors='pt')['pixel_values'][0].to(self.model.dtype)
            if imgs.ndim == 4:
                imgs = imgs.squeeze(0)
        self.model.get_model().mm_projector = self.model.get_model().mm_projector.to(imgs.dtype)
        if type(self.model) == LlavaQwenForRegression:
            score = self.get_randomized_feedback(input_ids, imgs.to(self.model.device))
            return dict(
                correct="None",
                score=score,
            )
        else:
            res, yes_score, no_score = self.get_randomized_feedback(input_ids, imgs.to(self.model.device))
            if type(res) == str:    
                correct = "yes" in res.lower()
                return dict(
                    correct=correct,
                    score=yes_score if correct else -no_score,
                )
            correct = ["yes" in r.lower() for r in res]
            return dict(
                correct=correct,
                score=[y_score if correct else -n_score for y_score, n_score in zip(yes_score, no_score)],
            )
    def get_randomized_feedback(self, input_ids, imgs=None, temperature=None):
        batch_size = imgs.shape[0]
        n_beams=3
        input_ids = input_ids.repeat(batch_size, 1)
        if type(self.model) == LlavaQwenForRegression:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    images=imgs,
                    #num_beams=n_beams,
                    #max_new_tokens=1,
                    #return_dict_in_generate=True,
                    #output_scores=True
                )
            if len(outputs.logits==1):
                return outputs.logits.squeeze(1)
            else:
                return [outputs.logits[i] for i in outputs.logits]
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=input_ids,
                    images=imgs,
                    num_beams=n_beams,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            output_text = self.tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if len(output_text) == 1:
                res = output_text[0]
                probs = [torch.nn.functional.softmax(score, dim=-1) for score in outputs.scores]
                return res, probs[0][0, self.yes_id], probs[0][0, self.no_id]
            res = output_text
            probs = [torch.nn.functional.softmax(score, dim=-1) for score in outputs.scores]
            return res, probs[0][0::n_beams, self.yes_id], probs[0][0::3, self.no_id]
