

python train.py  
# # Path: run.sh
python evaluation.py demo_config/agnews-bert_eval_0.2.json
python train.py  demo_config/agnews-bert_train_idn.json 
python evaluation.py demo_config/agnews-bert_eval_0.2.json
# # Path: run.sh
python train.py demo_config/agnews-bert_train_idn.json 
python evaluation.py demo_config/agnews-bert_eval_0.0.json



