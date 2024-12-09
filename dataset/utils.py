from datasets import load_dataset, load_from_disk
import llama_iti

def load_data(args):
    if args.dataset_name == "tqa":
        dataset = load_dataset("truthful_qa", 'generation')['validation']
    elif args.dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    elif args.dataset_name == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        
    elif args.dataset_name == 'coqa':
        import json
        import pandas as pd
        from datasets import Dataset

        def _save_dataset():
            # https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
            save_path = f'./coqa_dataset'
            if not os.path.exists(save_path):
                # https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
                with open(f'./coqa-dev-v1.0.json', 'r') as infile:
                    data = json.load(infile)['data']

                dataset = {}

                dataset['story'] = []
                dataset['question'] = []
                dataset['answer'] = []
                dataset['additional_answers'] = []
                dataset['id'] = []

                for sample_id, sample in enumerate(data):
                    story = sample['story']
                    questions = sample['questions']
                    answers = sample['answers']
                    additional_answers = sample['additional_answers']
                    for question_index, question in enumerate(questions):
                        dataset['story'].append(story)
                        dataset['question'].append(question['input_text'])
                        dataset['answer'].append({
                            'text': answers[question_index]['input_text'],
                            'answer_start': answers[question_index]['span_start']
                        })
                        dataset['id'].append(sample['id'] + '_' + str(question_index))
                        additional_answers_list = []

                        for i in range(3):
                            additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

                        dataset['additional_answers'].append(additional_answers_list)
                        story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
                        if not story[-1] == '.':
                            story = story + '.'

                dataset_df = pd.DataFrame.from_dict(dataset)

                dataset = Dataset.from_pandas(dataset_df)

                dataset.save_to_disk(save_path)
            return save_path

        # dataset = datasets.load_from_disk(_save_dataset())
        def get_dataset(tokenizer, split='validation'):
            # from https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
            dataset = load_from_disk(_save_dataset())
            id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

            def encode_coqa(example):
                example['answer'] = [example['answer']['text']] + example['additional_answers']
                example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
                return tokenizer(prompt, truncation=False, padding=False)

            dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
            return dataset
        
        HF_NAMES = {
            'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
            'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
            'alpaca_7B': 'circulus/alpaca-7b',
            'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
            'llama2_chat_7B': '/mnt/sharedata/ssd/common/LLMs/hub/Llama-2-7b-chat-hf',
            'llama2_chat_13B': 'models/Llama-2-13b-chat-hf',
            'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
        }

        MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir
        dataset = get_dataset(llama_iti.LlamaTokenizer.from_pretrained(MODEL, trust_remote_code=True))
    else:
        raise ValueError("Invalid dataset name")
    
    return dataset