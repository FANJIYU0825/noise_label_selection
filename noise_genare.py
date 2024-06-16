from util.noise_gen import noise_softmax, noise_gen_simple
from argparse import ArgumentParser
import pandas as pd
import numpy as np
# Instance dependent noise



def main():
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--src_data_path", type=str, required=True)
    arg_parser.add_argument("--save_path", type=str, required=True)
    arg_parser.add_argument("--noise_type", type=str, required=True, choices=['asym', 'idn', 'sym'])
    arg_parser.add_argument("--noise_ratio", type=float, required=True)
    arg_parser.add_argument("--predict_logits_path", type=str, required=True)
    
    args = arg_parser.parse_args()
    
    df = pd.read_csv(args.src_data_path,header=None)
    logit = np.load(args.predict_logits_path)
    
    x_train = df.iloc[:, 1:]
    y_inint=df.iloc[:, 0]
    print(len(y_inint))
    ind = df.iloc[:, 2]
    if args.noise_type == 'idn':
       
        y_noise=noise_gen_simple( logit[ind],y_inint, args.noise_ratio)
        path=args.save_path.replace('.csv', f'_{args.noise_type}_{args.noise_ratio}.csv')
        # data.to_csv(args.save_path, header=False, index=False)
        # data.to_csv(path, index=False)
       
        df.iloc[:, 0]=y_noise
        
        df.columns = ['label', 'text', 'index']
        df.to_csv(path, index=False)
        # drop column 2
        df.drop(columns=['index'], inplace=True)
        df.to_csv(args.save_path, index=False, header=None)
        
if __name__ == "__main__":
    main()