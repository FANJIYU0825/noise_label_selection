# write script to run the program
import os
# strategy = "GMM"
# ration = 0.2
data={
    "pretrained_model_name_or_path": "bert-base-uncased",
    "dropout_rate": 0.1,
    "p_threshold": 0.5,
    "temp": 0.5,
    "alpha": 0.75,
    "lambda_p": 0.0,
    "lambda_r": 0.3,
    "class_reg": True,
    "selection_strategy": "GMM",
    "dataset_name": "agnews",
    "train_file_path": "./data/agnews/agnews_test.csv",
    "eval_file_path": "./data/agnews/agnews_test.csv",
    "batch_size": 4,
    "batch_size_mix": 16,
    "max_sentence_len": 256,
    
    "seed": 1,
    "warmup_strategy": "samples",
    "warmup_samples": 1,
    "train_epochs": 6,
    "grad_acc_steps": 1,
    "model_save_path": "./save_model/CCN-selfmix_agnews_GMM_0.2.pt",
    "noise_type":"CCN"
}

pretrained_model_name_or_path = data["pretrained_model_name_or_path"]
dropout_rate = data["dropout_rate"]
p_threshold = data["p_threshold"]
temp = data["temp"]
alpha = data["alpha"]
lambda_p = data["lambda_p"]
lambda_r = data["lambda_r"]
class_reg = data["class_reg"]

dataset_name = data["dataset_name"]

eval_file_path = data["eval_file_path"]
batch_size = data["batch_size"]
batch_size_mix = data["batch_size_mix"]
max_sentence_len = data["max_sentence_len"]
seed = data["seed"]
warmup_strategy = data["warmup_strategy"]
warmup_samples = data["warmup_samples"]
train_epochs = data["train_epochs"]
grad_acc_steps = data["grad_acc_steps"]
noise_type = data["noise_type"]

# selection_strategy
selection_strategy_GMM = data["selection_strategy"]
selection_strategy_Rep = "Repset"
    # 0.2
for target in range(4):
    for replace in range(4):
        if target != replace:
            if noise_type  == "CCN":
            
                
                # from 0.1 to 0.4
                train_file_path_01= f"./data/agnews{target}/ag_news_ccn_sample10_label:t{target}to{replace}.csv"
                noised_rate_01= 0.1
                # Rep 0.1
                model_save_path_rep_01='./save_model/CCN-selfmix_agnews_Rep_0.1.pt'
                # GMM 0.1
                model_save_path_GMM01 = "./save_model/CCN-selfmix_agnews_GMM_0.1.pt"
                # 0.2
                train_file_path_02= f"./data/agnews{target}/ag_news_ccn_sample10_label:t{target}to{replace}.csv"
                train_file_path_02
                noised_rate_02= 0.2
                # Rep 0.2
                model_save_path_rep_02='./save_model/CCN-selfmix_agnews_Rep_0.2.pt'
                # GMM 0.2
                model_save_path_GMM02 = "./save_model/CCN-selfmix_agnews_GMM_0.2.pt"
                #0.3
                train_file_path_03= f"./data/agnews{target}/ag_news_ccn_sample10_label:t{target}to{replace}.csv"
                noised_rate_03= 0.3
                # Rep 0.3
                model_save_path_rep_03='./save_model/CCN-selfmix_agnews_Rep_0.3.pt'
                # GMM 0.3  
                model_save_path_GMM03 = "./save_model/CCN-selfmix_agnews_GMM_0.3.pt"
                
                #0.4
                noise_rate_04=0.4
                train_file_path_04=f'./data/agnews{target}/ag_news_ccn_sample10_label:t{target}to{replace}.csv'
                # rep0.4
                model_save_path_rep_04='./save_model/CCN-selfmix_agnews_Rep_0.4.pt'
                # GMM 0.4
                model_save_path_GMM_04='./save_model/CCN-selfmix_agnews_GMM_0.4.pt'
                target_class = target
                replace_class = replace
            
        # idn
            elif  noise_type  == "idn":
                train_file_path_02= "./data/trec/train_corrupted.csv"
                noised_rate_02= 0.2
                # Rep 0.2
                model_save_path_rep_02='./save_model/all_idn-selfmix_agnews_Rep_0.2.pt'
            
                # GMM 0.2
                model_save_path_GMM02 = "./save_model/all_idn-selfmix_agnews_GMM_0.2.pt"
                train_comand_GMM02 = f'python train.py --target_class {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_02}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_02} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM02}'
                os.system(train_comand_GMM02)
            else:
                noise_rate_0 = 0.0
                train_file_path_02= "./data/agnews/ag_news_train_clean_sample.csv"
                noised_rate_02= 0.0
                # Rep 0.0
                model_save_path_rep_clean='./save_model/clean-selfmix_agnews_Rep_0.pt'
                model_save_path_GMM_clean = "./save_model/clean-selfmix_agnews_GMM_0.pt"
                train_comand_GMM_clean = f"python train.py --noise_type {noise_type } --noised_rate {noise_rate_0}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_02} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM_clean}"
                train_comand_rep_clean = f"python train.py --noise_type {noise_type } --noised_rate {noise_rate_0}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_Rep} --dataset_name {dataset_name} --train_file_path {train_file_path_02} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_rep_clean}"
                # eval
                eval_comand_GMM_clean = f"python evaluation.py --noise_type {noise_type } --noised_rate {noise_rate_0}  --model_name_or_path {model_save_path_GMM_clean} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_GMM}"
                eval_comand_rep_clean = f"python evaluation.py --noise_type {noise_type } --noised_rate {noise_rate_0}  --model_name_or_path {model_save_path_rep_clean} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_Rep}"
                os.system(train_comand_GMM_clean)
                os.system(train_comand_rep_clean)
                os.system(eval_comand_GMM_clean)
                os.system(eval_comand_rep_clean)
            
            train_comand_GMM01 =f'python train.py --target_class  {target}  --replace_class {replace} --noise_type {noise_type} --noised_rate {noised_rate_01}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_01} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM01}'    
            eval_comand_GMM01 = f"python evaluation.py --target_class {target} --replace_class {replace} --noise_type {noise_type } --noised_rate {noised_rate_01}  --model_name_or_path {model_save_path_GMM01} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_GMM}"
            train_comand_GMM02 = f"python train.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_02}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_02} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM02}"
            eval_comand_GMM02 = f"python evaluation.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_02}  --model_name_or_path {model_save_path_GMM02} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_GMM}"
            train_comand_GMM03 = f"python train.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_03}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_03} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM03}"
            eval_comand_GMM03 = f"python evaluation.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_03}  --model_name_or_path {model_save_path_GMM03} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_GMM}"

            train_comand_GMM_04=f"python train.py --target_class  {target} --replace_class {replace} --noise_type {noise_type} --noised_rate {noise_rate_04}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_04} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM_04}"
            eval_comand_GMM04 = f"python evaluation.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noise_rate_04}  --model_name_or_path {model_save_path_GMM_04} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_GMM}"
            list_gmm_train_comand = [train_comand_GMM01,train_comand_GMM02,train_comand_GMM03,train_comand_GMM_04]
            list_gmm_eval_comand = [eval_comand_GMM01,eval_comand_GMM02,eval_comand_GMM03,eval_comand_GMM04]

            train_comand_rep_01 = f'python train.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_01}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_Rep} --dataset_name {dataset_name} --train_file_path {train_file_path_01} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_rep_01}'
            eval_comand_rep_01 = f"python evaluation.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_01}  --model_name_or_path {model_save_path_rep_01} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_Rep}"
            train_comand_rep_02 =f'python train.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_02}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_Rep} --dataset_name {dataset_name} --train_file_path {train_file_path_02} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_rep_02}'
            eval_comand_rep_02=f"python evaluation.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_02}  --model_name_or_path {model_save_path_rep_02} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_Rep}"
            train_comand_rep_03= f'python train.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_03}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_Rep} --dataset_name {dataset_name} --train_file_path {train_file_path_03} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_rep_03}'
            eval_comand_rep_03=f"python evaluation.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noised_rate_03}  --model_name_or_path {model_save_path_rep_03} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_Rep}"
            train_comand_rep_04 = f'python train.py --target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noise_rate_04}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_Rep} --dataset_name {dataset_name} --train_file_path {train_file_path_04} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_rep_04}'
            eval_comand_rep_04=f"python evaluation.py ---target_class  {target} --replace_class {replace} --noise_type {noise_type}  --noised_rate {noise_rate_04}  --model_name_or_path {model_save_path_rep_04} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_Rep}"
            list_rep_train_comand = [train_comand_rep_01,train_comand_rep_02,train_comand_rep_03,train_comand_rep_04]
            list_rep_eval_comand = [eval_comand_rep_01,eval_comand_rep_02,eval_comand_rep_03,eval_comand_rep_04]

            for i in range(len(list_gmm_train_comand)):
                gmm_train = list_gmm_train_comand[i]
                gmm_eval = list_gmm_eval_comand[i]
                rep_train = list_rep_train_comand[i]
                rep_eval = list_rep_eval_comand[i]
                os.system(gmm_train)
                os.system(gmm_eval)
                os.system(rep_train)
                os.system(rep_eval)

 



# coteaching 

coteach_comand_02 = f"python baselines/train_coteaching.py --train_path {train_file_path_02} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noised_rate_02} "

# coteach_comand_04 = f"python baselines/train_coteaching.py --train_path {train_file_path_04} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noise_rate_04}"
os.system(coteach_comand_02)
# os.system(coteach_comand_04)
#basebert
base_comand_02 = f"python baselines/train_base.py --train_path {train_file_path_02} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noised_rate_02}"
# base_comand_04 = f"python baselines/train_base.py --train_path {train_file_path_04} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noise_rate_04}"

os.system(base_comand_02)
# os.system(base_comand_04)
# coteach_comand_clean = f"python baselines/train_coteaching.py --train_path {train_file_path_02} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noise_rate_0} "
# os.system(coteach_comand_clean)
# base_comand_clean = f"python baselines/train_base.py --train_path {train_file_path_02} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noise_rate_0}"
# os.system(base_comand_clean)