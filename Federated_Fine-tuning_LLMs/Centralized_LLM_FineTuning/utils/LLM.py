import os
import textwrap
from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    logging
)
from huggingface_hub import login

# Configuration initiale
load_dotenv()
logging.set_verbosity_error()  # Réduire le logging verbeux

class LLM_eval:
    def __init__(self, model_id: str = "meta-llama/Llama-2-7b-hf", **kwargs):
        print("[INIT] Initialisation de LLM_eval...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] Device utilisé : {self.device}")
        self.model_id = model_id
        self.temp = kwargs.get("temp", 0.7)

        # Authentification HF
        print("[INIT] Authentification Hugging Face...")
        self._authenticate()

        # Chargement du tokenizer
        print("[INIT] Chargement du tokenizer...")
        self.tokenizer = self._load_tokenizer()

        # Chargement du modèle
        print("[INIT] Chargement du modèle...")
        self.model = self._load_model()

        print("[INIT] LLM prêt à l'emploi.")

    def _authenticate(self):
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("Token HF non trouvé dans .env")
        login(token=hf_token)

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        return tokenizer

    def _load_model(self):
        print("[MODEL] Chargement du modèle depuis Hugging Face...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float32,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        print("[MODEL] Modèle chargé.")
        return model

    def generate(self, prompt: str, max_tokens: int = 80, verbose: bool = False) -> str:
        print("[GENERATE] Génération du texte en cours...")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print("[GENERATE] Appel au modèle...")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temp,
                do_sample=True,
                top_k=50,
                top_p=0.9
            )
        print("[GENERATE] Texte généré")

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if verbose:
            self._print_formatted("Prompt", prompt)
            self._print_formatted("Response", response)
        else:
            print(response)

        return response

    def _print_formatted(self, title: str, content: str, width: int = 50):
        wrapper = textwrap.TextWrapper(
            width=width,
            subsequent_indent="\t",
            replace_whitespace=False
        )
        print(f"{title}:\n{wrapper.fill(content)}\n")

class LLM_pretrained(LLM_eval):
    def __init__(self, **kwargs):
        super().__init__(
            model_id="meta-llama/Llama-2-7b-hf",
            **kwargs
        )
