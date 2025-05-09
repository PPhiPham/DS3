import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from functools import lru_cache

# Init tqdm and stemmer
tqdm.pandas()
stemmer = SnowballStemmer('english')

print("Loading datasets...")
df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
print("Datasets loaded.")

num_train = df_train.shape[0]

# Caching stemmed words
@lru_cache(maxsize=10000)
def cached_stem(word):
    return stemmer.stem(word)

def fast_stem_sentence(s):
    return " ".join([cached_stem(word) for word in str(s).lower().split()])

def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())

print("Merging train and test data...")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
print("Data merged.")

print("Applying stemming with progress bars...")
df_all['search_term'] = df_all['search_term'].progress_apply(fast_stem_sentence)
df_all['product_title'] = df_all['product_title'].progress_apply(fast_stem_sentence)
df_all['product_description'] = df_all['product_description'].progress_apply(fast_stem_sentence)
print("Stemming complete.")

print("Engineering features...")
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
print("Feature engineering complete.")

print("Cleaning up text columns...")
df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1, inplace=True)

print("Splitting back to train/test sets...")
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values
print("Data split complete.")

print("Training model...")
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
print("Model training complete.")

print("Generating predictions...")
y_pred = clf.predict(X_test)

print("Saving submission file...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")