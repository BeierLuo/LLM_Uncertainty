from abc import ABC, abstractmethod
import llama_iti
import torch

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': '/mnt/sharedata/ssd/common/LLMs/hub/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'models/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
}

class BaseEstimator(ABC):
    def __init__(self, args, dataset, index_dict):
        super().__init__()
        self.MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir
        self.tokenizer = llama_iti.LlamaTokenizer.from_pretrained(self.MODEL, trust_remote_code=True)
        self.model = llama_iti.LlamaForCausalLM.from_pretrained(self.MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                        device_map="auto").cuda()
        self.dataset = dataset
        self.index_dict = index_dict
        self.length = len(self.index_dict['used_indices']) if args.dataset_name == 'tydiqa' else len(dataset)
