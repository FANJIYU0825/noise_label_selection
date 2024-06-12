from datasets import load_dataset
import pandas as pd
ag_news = load_dataset('ag_news')
df_train=pd.DataFrame(ag_news['train'])
df_train = df_train[['label', 'text']]  
df_train.to_csv('data/ag_news_train.csv', index=False, header=False)
