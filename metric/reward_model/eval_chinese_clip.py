# import
from paddlenlp.transformers import AutoModel, AutoProcessor
import paddle
import paddle.nn as nn
from PIL import Image

# load model
class Chinese_CLIP(nn.Layer):
    def __init__(self):
        super(Chinese_CLIP, self).__init__()
        clip_model = "OFA-Sys/chinese-clip-vit-huge-patch14"

        self.clip = AutoModel.from_pretrained(clip_model)
        self.processor = AutoProcessor.from_pretrained(clip_model)

class CN_PickScore:
    def __init__(self):
        checkpoint_path = '/root/paddlejob/workspace/env_run/code/27800_global_step.pdparams'

        self.model = Chinese_CLIP()
        self.model.set_state_dict(paddle.load(checkpoint_path))

    def calc_probs(self, prompt, images, output_probs=True):
        with paddle.no_grad():
            # preprocess
            image_inputs = self.model.processor(
                    images=images,
                    max_length=120,
                    truncation=True, 
                    padding=True,
                    return_tensors="pd"
            )
            
            text_inputs = self.model.processor(
                    text=prompt,
                    max_length=120,
                    truncation=True, 
                    padding=True,
                    return_tensors="pd"
            )

        with paddle.no_grad():
            # embed
            image_embs = self.model.clip.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(p=2, axis=-1, keepdim=True)
        
            text_embs = self.model.clip.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(p=2, axis=-1, keepdim=True)  
        
            # score
            scores = self.model.clip.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            if not output_probs:
                return scores.cpu().tolist()
            
            # get probabilities if you have multiple images to choose from
            probs = paddle.nn.functional.softmax(scores, axis=-1)
        
        return probs.cpu().tolist()

if __name__ == "__main__":
    pil_images = [
        Image.open("p1.png"), 
        Image.open("p2.png")
    ]
    prompt = "prompt"
    print(CN_PickScore().calc_probs(prompt, pil_images))
