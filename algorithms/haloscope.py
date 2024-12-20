from .base_estimator import BaseEstimator
from evaluate import load
from utils import *

class HaloScope(BaseEstimator):
    def __init__(self, args, dataset, index_dict):
        super().__init__(args, dataset, index_dict)
        
    def generate(self, args): # generate answers for datasets
        
        create_answers_path(args)

        # Get the token IDs for period and EOS.
        period_token_id = [self.tokenizer(_)['input_ids'][-1] for _ in ['\n']]
        period_token_id += [self.tokenizer.eos_token_id]

        for i in tqdm(range(self.index_dict['begin_index'], self.index_dict['end_index']), desc="Saving Answers"):
            answers = [None] * args.num_gene
            prompt = self.build_decoded_prompt(args, i)
            for gen_iter in range(args.num_gene):
                decoded = self.generate_single_answer(args, prompt)
                process_and_save_answer(args, decoded, answers)
             
            # print('sample: ', i)
            if args.most_likely:
                info = 'most_likely_'
            else:
                info = 'batch_generations_'
            # print("Saving answers")
            np.save(f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/answers/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy',
                    answers)

    def build_decoded_prompt(self, args, i, anw=None):
        if args.dataset_name == 'tydiqa':
            question = self.dataset[int(self.index_dict['used_indices'][i])]['question']
            return self.tokenizer(
                "Concisely answer the following question based on the information in the given passage: \n" + \
                " Passage: " + self.dataset[int(self.index_dict['used_indices'][i])]['context'] + " \n Q: " + question + " \n A:",
                return_tensors='pt').input_ids.cuda()
        elif args.dataset_name == 'coqa':
            return self.tokenizer(
                self.dataset[i]['prompt'] + (anw if anw is not None else ''), return_tensors='pt').input_ids.cuda()
        else:
            question = self.dataset[i]['question']
            prompt = f"Answer the question concisely. Q: {question}" + (" A: " + anw if anw is not None else "")
            return self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        
    def generate_single_answer(self, args, prompt):
        if args.most_likely:
            generated = self.model.generate(prompt,
                                        num_beams=5,
                                        num_return_sequences=1,
                                        do_sample=False,
                                        max_new_tokens=64,
                                        )
        else:
            generated = self.model.generate(prompt,
                                        do_sample=True,
                                        num_return_sequences=1,
                                        num_beams=1,
                                        max_new_tokens=64,
                                        temperature=0.5,
                                        top_p=1.0)

        decoded = self.tokenizer.decode(generated[0, prompt.shape[-1]:], skip_special_tokens=True)
        return decoded
     
    # Get the ground truth answers based on the dataset name
    def get_ground_truth_answers(self, args, i):
        if args.dataset_name == 'tqa':
            best_answer = self.dataset[i]['best_answer']
            correct_answer = self.dataset[i]['correct_answers']
            all_answers = [best_answer] + correct_answer
        elif args.dataset_name == 'triviaqa':
            all_answers = self.dataset[i]['answer']['aliases']
        elif args.dataset_name == 'coqa':
            all_answers = self.dataset[i]['answer']
        elif args.dataset_name == 'tydiqa':
            all_answers = self.dataset[int(self.index_dict['used_indices'][i])]['answers']['text']
        return all_answers

    # Evaluate answers using ROUGE
    def evaluate_with_rouge(self, rouge, predictions, all_answers):
        all_results = np.zeros((len(all_answers), len(predictions)))
        for anw in range(len(all_answers)):
            results = rouge.compute(predictions=predictions,
                                references=[all_answers[anw]] * len(predictions),
                                use_aggregator=False)
            all_results[anw] = results['rougeL']
        return np.max(all_results, axis=0)
    
    # Evaluate answers using BLEURT
    def evaluate_with_bleurt(self, bleurt_model, bleurt_tokenizer, predictions, all_answers):
        all_results = np.zeros((len(all_answers), len(predictions)))
        with torch.no_grad():
            for anw in range(len(all_answers)):
                inputs = bleurt_tokenizer(predictions.tolist(), [all_answers[anw]] * len(predictions),
                                padding='longest', return_tensors='pt')
                for key in list(inputs.keys()):
                    inputs[key] = inputs[key].cuda()
                res = np.asarray(bleurt_model(**inputs).logits.flatten().tolist())
                all_results[anw] = res
        return np.max(all_results, axis=0)

    # Main evaluation function
    def evaluate_answers(self, args):
        bleurt_model, bleurt_tokenizer = load_bleurt_model_and_tokenizer(args)
        rouge = load('rouge')
        gts = np.zeros(0)

        for i in range(self.length):
            all_answers = self.get_ground_truth_answers(args, i)
            if args.most_likely:
                file_path = f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy'
            else:
                file_path = f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/answers/batch_generations_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy'
            answers = np.load(file_path, allow_pickle=True)

            if args.use_rouge:
                gts = np.concatenate([gts, self.evaluate_with_rouge(rouge, answers, all_answers)], 0)
            else:
                gts = np.concatenate([gts, self.evaluate_with_bleurt(bleurt_model, bleurt_tokenizer, answers, all_answers)], 0)

            if i % 50 == 0 and args.use_rouge:
                print("samples passed: ", i)
            elif i % 10 == 0 and not args.use_rouge:
                print("samples passed: ", i)
            
        save_path = f'{args.output_dir}/ml_{args.dataset_name}_rouge_score.npy' if args.most_likely and args.use_rouge else \
            f'{args.output_dir}/ml_{args.dataset_name}_bleurt_score.npy' if args.most_likely else \
            f'{args.output_dir}/bg_{args.dataset_name}_rouge_score.npy' if args.use_rouge else \
            f'{args.output_dir}/bg_{args.dataset_name}_bleurt_score.npy'

        np.save(save_path, gts)

    def save_embeddings(self, embeddings, filename):
        embeddings = np.asarray(np.stack(embeddings), dtype=np.float32)
        np.save(filename, embeddings)

    def get_embeddings(self, args):
        embed_generated = []
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        embed_generated_loc2 = []
        embed_generated_loc1 = []

        for i in tqdm(range(len(self.dataset) if args.dataset_name != 'tydiqa' else len(self.index_dict['used_indices'])), desc='Start Embedding Features'):
            answers = np.load(f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
            for anw in answers:
                prompt = self.build_decoded_prompt(args, i, anw)
                with torch.no_grad():
                    hidden_states = self.model(prompt, output_hidden_states=True).hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                    hidden_states = hidden_states.detach().cpu().numpy()[:, -1, :]
                embed_generated.append(hidden_states)

                with torch.no_grad():
                    with TraceDict(self.model, HEADS + MLPS) as ret:
                        output = self.model(prompt, output_hidden_states=True)
                    head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                    mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
                    mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

                    embed_generated_loc2.append(mlp_wise_hidden_states[:, -1, :])
                    embed_generated_loc1.append(head_wise_hidden_states[:, -1, :])
        
        self.save_embeddings(embed_generated, f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy')
        self.save_embeddings(embed_generated_loc1, f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy')
        self.save_embeddings(embed_generated_loc2, f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_embeddings_mlp_wise.npy')
    
    def estimate(self, args):
        file_path1 = f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy'
        file_path2 = f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy'
        file_path3 = f'{args.output_dir}/save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_embeddings_mlp_wise.npy'

        if not (os.path.exists(file_path1) and os.path.exists(file_path2) and os.path.exists(file_path3)):
            self.get_embeddings(args)

        permuted_index = np.random.permutation(self.length)

        # Divide the wild dataset (i.e. unlabeled dataset)
        wild_q_indices = permuted_index[:int(args.wild_ratio * self.length)]
        wild_q_indices1 = wild_q_indices[:len(wild_q_indices) - 200]  # unlabeled dataset
        wild_q_indices2 = wild_q_indices[len(wild_q_indices) - 200:]  # validation dataset
        # get the gt of each dataset
        gt_label_test, gt_label_wild, gt_label_val = generate_label(args, wild_q_indices, self.index_dict, self.dataset)

        embed_generated = load_feature_embeddings(args)
        feat_indices_wild, feat_indices_eval = select_feature_indices(wild_q_indices1, wild_q_indices2, self.length)
        embed_generated_wild, embed_generated_eval = slice_feature_embeddings(embed_generated, feat_indices_wild,
                                                                              feat_indices_eval, args.feat_loc_svd)
        returned_results = svd_embed_score(embed_generated_eval, gt_label_val, 1, 11, mean=0, svd=0,
                                           weight=args.weighted_svd) # get the best numbers of principal components
        best_scores, projection = perform_pca_and_calculate_scores(embed_generated_wild, returned_results, args)

        # Direct projection on test set
        print('============Directly Project on Test Set=============')
        feat_indices_test = []
        for i in range(self.length):
            if i not in wild_q_indices:
                feat_indices = np.arange(1 * i, 1 * i + 1).tolist()
                feat_indices_test.extend(feat_indices)
        if args.feat_loc_svd == 3:
            embed_generated_test = embed_generated[feat_indices_test][:, 1:, :]
        else:
            embed_generated_test = embed_generated[feat_indices_test]
        
        test_scores = np.mean(np.matmul(embed_generated_test[:,returned_results['best_layer'],:],
                                   projection), -1, keepdims=True)

        assert test_scores.shape[1] == 1
        test_scores = np.sqrt(np.sum(np.square(test_scores), axis=1))

        measures = get_measures(returned_results['best_sign'] * test_scores[gt_label_test == 1],
                                 returned_results['best_sign'] *test_scores[gt_label_test == 0], plot=False)
        print_measures(measures[0], measures[1], measures[2], 'direct-projection')

        print('============Linear Probe=============')
        thresholds = np.linspace(0, 1, num=40)[1:-1]
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        auroc_over_thres = []
        for thres_wild in thresholds:
            best_auroc = 0
            for layer in range(len(embed_generated_wild[0])):
                thres_wild_score = np.sort(best_scores)[int(len(best_scores) * thres_wild)]
                true_wild = embed_generated_wild[:, layer, :][best_scores > thres_wild_score]
                false_wild = embed_generated_wild[:, layer, :][best_scores <= thres_wild_score]

                embed_train = np.concatenate([true_wild, false_wild], 0)
                label_train = np.concatenate([np.ones(len(true_wild)),
                                              np.zeros(len(false_wild))], 0)

                ## gt training, saplma
                # embed_train = embed_generated_wild[:,layer,:]
                # label_train = gt_label_wild
                ## gt training, saplma
                from linear_probe import get_linear_acc

                best_acc, final_acc, (
                    clf, best_state, best_preds, preds, labels_val), losses_train = get_linear_acc(
                    embed_train,
                    label_train,
                    embed_train,
                    label_train,
                    2, epochs=50,
                    print_ret=True,
                    batch_size=512,
                    cosine=True,
                    nonlinear=True,
                    learning_rate=0.05,
                    weight_decay=0.0003)

                clf.eval()
                output = clf(torch.from_numpy(
                    embed_generated_test[:, layer, :]).cuda())
                pca_wild_score_binary_cls = torch.sigmoid(output)

                pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()

                if np.isnan(pca_wild_score_binary_cls).sum() > 0:
                    breakpoint()
                measures = get_measures(pca_wild_score_binary_cls[gt_label_test == 1],
                                        pca_wild_score_binary_cls[gt_label_test == 0], plot=False)

                if measures[0] > best_auroc:
                    best_auroc = measures[0]
                    best_result = [100 * measures[0]]
                    best_layer = layer

            auroc_over_thres.append(best_auroc)
            print('thres: ', thres_wild, 'best result: ', best_result, 'best_layer: ', best_layer)
