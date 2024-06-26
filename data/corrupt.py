import pandas as pd
import argparse
import numpy as np
import random

NUM_CLASSES = {
    'trec': 6,
    'imdb': 2,
    'agnews': 4
}


parser = argparse.ArgumentParser()
parser.add_argument("--src_data_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--noise_type", type=str, required=True, choices=['asym', 'sym'])
parser.add_argument("--noise_ratio", type=float, required=True)
parser.add_argument("--keep_true_labels", action="store_true")
args = parser.parse_args()


def main(args):
    # set seed
    random.seed(1)
    np.random.seed(1)
    
    data = pd.read_csv(args.src_data_path, header=None)
    assert 0. <= args.noise_ratio <= 1.
    
    if args.keep_true_labels:
        data[2] = data[0]
    
    num_classes = max(data[0]) 
    for idx in range(len(data)):
        true_label = data.loc[idx, 0]
        if args.noise_ratio == 'sym':
            p = args.noise_ratio / (num_classes - 1) * np.ones(num_classes)
            p[true_label] = 1 - args.noise_ratio
            observed_label = np.random.choice(num_classes, p=p)
        else:
            
            nlabel = (true_label ) % 2
                
            

            observed_label = np.random.choice([true_label, nlabel], p=[1 - args.noise_ratio, args.noise_ratio])
        data.loc[idx, 0] = observed_label
        
        
    path=args.save_path.replace('.csv', f'_{args.noise_type}_{args.noise_ratio}.csv')
    data.columns = ['label', 'text', 'index']
    data.to_csv(path, index=False)
    # drop column 2
    data.drop(columns=['index'], inplace=True)
    
    data.to_csv(args.save_path, index=False, header=None)
if __name__ == "__main__":
    main(args)
