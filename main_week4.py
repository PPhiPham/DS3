import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from functools import lru_cache

tqdm.pandas()
stemmer = SnowballStemmer('english')

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")

@lru_cache(maxsize=10000)
def cached_stem(word): return stemmer.stem(word)
def fast_stem_sentence(s): return " ".join([cached_stem(w) for w in str(s).lower().split()])
def str_common_word(str1, str2): return sum(int(str2.find(word) >= 0) for word in str1.split())

df_attr_grouped = df_attr.groupby("product_uid")["value"].apply(lambda x: " ".join(x.astype(str))).reset_index()
df_attr_grouped.columns = ['product_uid', 'attr_values']
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr_grouped, how='left', on='product_uid')

df_all['search_term'] = df_all['search_term'].progress_apply(fast_stem_sentence)
df_all['product_title'] = df_all['product_title'].progress_apply(fast_stem_sentence)
df_all['product_description'] = df_all['product_description'].progress_apply(fast_stem_sentence)
df_all['attr_values'] = df_all['attr_values'].fillna("").progress_apply(fast_stem_sentence)

df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
df_all['word_in_attr'] = df_all.apply(lambda row: str_common_word(row['search_term'], row['attr_values']), axis=1)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
corpus = df_all['search_term'] + " " + df_all['product_title'] + " " + df_all['product_description']
tfidf_vectorizer.fit(corpus)
search_term_tfidf = tfidf_vectorizer.transform(df_all['search_term'])
product_info_tfidf = tfidf_vectorizer.transform(df_all['product_title'] + " " + df_all['product_description'])
df_all['tfidf_cosine_sim'] = [
    cosine_similarity(search_term_tfidf[i], product_info_tfidf[i])[0][0] for i in range(search_term_tfidf.shape[0])
]

df_all.drop(['search_term', 'product_title', 'product_description', 'product_info', 'attr_values'], axis=1, inplace=True)

num_train = df_train.shape[0]
df_combined_train = df_all.iloc[:num_train].copy()
y = df_combined_train['relevance'].values
X = df_combined_train.drop(['id', 'relevance'], axis=1).values
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

rf_direct = RandomForestRegressor(
    n_estimators=17,
    max_depth=9,
    min_samples_split=8,
    min_samples_leaf=1,
    max_features=None,
    random_state=0
)

rf_direct.fit(X_train, y_train)
feature_names = df_combined_train.drop(['id', 'relevance'], axis=1).columns
importances = rf_direct.feature_importances_
sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

print("\nTop 5 important features:")
for feature, score in sorted_importances[:5]:
    print(f"{feature}: {score:.4f}")