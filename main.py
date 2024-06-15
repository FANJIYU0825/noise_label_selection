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
    "train_file_path": "./data/agnews/ccn_agnew0.2.csv",
    "eval_file_path": "./data/agnews/agnews_test.csv",
    "batch_size": 4,
    "batch_size_mix": 16,
    "max_sentence_len": 256,
    
    "seed": 1,
    "warmup_strategy": "samples",
    "warmup_samples": 5000,
    "train_epochs": 4,
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

if noise_type  == "CCN":
    train_file_path_02= "./data/agnews/ccn_agnews0.2.csv"
    
    train_file_path_02
    noised_rate_02= 0.2
    # Rep 0.2
    model_save_path_rep_02='./save_model/CCN-selfmix_agnews_Rep_0.2.pt'
    # GMM 0.2
    model_save_path_GMM02 = "./save_model/CCN-selfmix_agnews_GMM_0.2.pt"

    #0.4
    noise_rate_04=0.4
    train_file_path_04='./data/agnews/ccn_agnews0.4.csv'
    # rep0.4
    model_save_path_rep_04='./save_model/CCN-selfmix_agnews_Rep_0.4.pt'
    # GMM 0.4
    model_save_path_GMM_04='./save_model/CCN-selfmix_agnews_GMM_0.4.pt'
# idn
elif  noise_type  == "idn":
    train_file_path_02= "./data/agnews/idn_agnews0.2.csv"
    noised_rate_02= 0.2
    # Rep 0.2
    model_save_path_rep_02='./save_model/idn-selfmix_agnews_Rep_0.2.pt'
    # GMM 0.2
    model_save_path_GMM02 = "./save_model/idn-selfmix_agnews_GMM_0.2.pt"

    #0.4
    noise_rate_04=0.4
    train_file_path_04='./data/agnews/idn_agnews0.4.csv'
    # rep0.4
    model_save_path_rep_04='./save_model/idn-selfmix_agnews_Rep_0.4.pt'
    # GMM 0.4
    model_save_path_GMM_04='./save_model/idn-selfmix_agnews_GMM_0.4.pt'

# train_comand_GMM02 = f"python train.py --noise_type {noise_type } --noised_rate {noised_rate_02}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_02} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM02}"
# os.system(train_comand_GMM02)

# train_comand_Rep02=f"python train.py --noise_type {noise_type } --noised_rate {noised_rate_02}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate 0.1 --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_Rep} --dataset_name {dataset_name} --train_file_path {train_file_path_02} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_rep_02}"
# os.system(train_comand_Rep02)

# train_comand_Rep_04= f'python train.py --noise_type {noise_type } --noised_rate {noise_rate_04}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_Rep} --dataset_name {dataset_name} --train_file_path {train_file_path_04} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_rep_04}'

# os.system(train_comand_Rep_04)


# train_comand_GMM_04=f"python train.py --noise_type {noise_type } --noised_rate {noise_rate_04}  --pretrained_model_name_or_path {pretrained_model_name_or_path} --dropout_rate {dropout_rate} --p_threshold {p_threshold} --temp {temp} --alpha {alpha} --lambda_p {lambda_p} --lambda_r {lambda_r} --class_reg {class_reg} --selection_strategy {selection_strategy_GMM} --dataset_name {dataset_name} --train_file_path {train_file_path_04} --eval_file_path {eval_file_path} --batch_size {batch_size} --batch_size_mix {batch_size_mix} --max_sentence_len {max_sentence_len} --seed {seed} --warmup_strategy {warmup_strategy} --warmup_samples {warmup_samples} --train_epochs {train_epochs} --grad_acc_steps {grad_acc_steps} --model_save_path {model_save_path_GMM_04}"

# os.system(train_comand_GMM_04)


#     #evaluation


noise_rate = 0.2
# eval_comand_GMM02 = f"python evaluation.py --noise_type {noise_type } --noised_rate {noise_rate}  --model_name_or_path {model_save_path_GMM02} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_GMM}"
# os.system(eval_comand_GMM02 )

# eval_comand_Rep02 = f"python evaluation.py  --noise_type {noise_type } --noised_rate {noised_rate_02} --model_name_or_path {model_save_path_rep_02} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size} --selection_strategy {selection_strategy_Rep}"
# os.system(eval_comand_Rep02)





noise_rate = 0.4
# eval_comand_GMM_04 = f"python evaluation.py --noise_type {noise_type } --noised_rate {noise_rate} --model_name_or_path {model_save_path_GMM_04} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_GMM}"
# os.system(eval_comand_GMM_04)


# eval_comand_Rep_04 = f"python evaluation.py --noise_type {noise_type } --noised_rate {noise_rate_04} --model_name_or_path {model_save_path_rep_04} --pretrained_model_name_or_path {pretrained_model_name_or_path} --eval_file_path {eval_file_path} --dataset_name {dataset_name} --batch_size {batch_size}  --selection_strategy {selection_strategy_Rep}"
# os.system(eval_comand_Rep_04)


#coteaching 

coteach_comand_02 = f"python baselines/train_coteaching.py --train_path {train_file_path_02} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noised_rate_02} "

coteach_comand_04 = f"python baselines/train_coteaching.py --train_path {train_file_path_04} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noise_rate_04}"
os.system(coteach_comand_02)
os.system(coteach_comand_04)
#basebert
base_comand_02 = f"python baselines/train_base.py --train_path {train_file_path_02} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noised_rate_02}"
base_comand_04 = f"python baselines/train_base.py --train_path {train_file_path_04} --test_path {eval_file_path} --noise_type {noise_type} --noise_ratio {noise_rate_04}"

os.system(base_comand_02)
os.system(base_comand_04)