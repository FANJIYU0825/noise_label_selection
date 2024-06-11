from datasets import load_dataset
import pandas as pd
ag_news = load_dataset('ag_news')
df_train=pd.DataFrame(ag_news['train'])

df_train.to_csv('data/ag_news_raw_train.csv', index=False)
