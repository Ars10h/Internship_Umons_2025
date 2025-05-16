
import os
import textwrap
import torch

# Éviter que transformers essaie de charger TensorFlow, JAX, etc.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LLM_eval:
    def __init__(self, model_id="google/flan-t5-small", temp=0.7, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp = temp
        self.model_id = model_id

        print(f"[INIT] Chargement du modèle {model_id} sur {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)

    def eval(self, prompt, log_probs=False, verbose=True, top_k=True):
        if verbose:
            print(f"Prompt:\n{textwrap.fill(prompt, width=50)}")

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temp,
                top_k=50 if top_k else 0,
                max_new_tokens=80
            )

        self.response = self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_response(self):
        return self.response

    def print_response(self, char_width=50, verbose=True):
        wrapped = textwrap.fill(self.response, char_width, subsequent_indent="\t")
        print("Response:\n" + wrapped)


class LLM_pretrained(LLM_eval): pass
class LLM_fl(LLM_eval): pass
class LLM_cen_partial(LLM_eval): pass
class LLM_cen(LLM_eval): pass
