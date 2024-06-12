import pandas as pd 
import os 

# list dir 
data_dir = 'AG_NEWS_NOISE'
if os.listdir(data_dir) == []:
    print('No data in AG_NEWS_NOISE')
else:
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            
            data = pd.read_csv(os.path.join(data_dir, file))
            # file_name have text IDN ==true
                     
            if file.startswith('dependent'):
                # sample every label 100shot 
               data = data[['label_noisy', 'text']]
               data['label'] = data['label_noisy']
            # sample every label 100shot 
           
            else:
                data.columns = ['label', 'text']
            data = data.groupby('label').head(100).reset_index(drop=True)
            data.to_csv(os.path.join(data_dir, "labeled"+file), index=False, header=False)