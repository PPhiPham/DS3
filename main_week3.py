import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize tqdm and stemmer
tqdm.pandas()
stemmer = SnowballStemmer('english')

# Load datasets
print("Loading datasets...")
df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
print("Datasets loaded.")

num_train = df_train.shape[0]

@lru_cache(maxsize=10000)
def cached_stem(word):
    return stemmer.stem(word)

def fast_stem_sentence(s):
    return " ".join([cached_stem(word) for word in str(s).lower().split()])

def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())

print("Processing attributes...")
df_attr_grouped = df_attr.groupby("product_uid")["value"].apply(lambda x: " ".join(x.astype(str))).reset_index()
df_attr_grouped.columns = ['product_uid', 'attr_values']
print("Attributes processed.")

print("Merging datasets...")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr_grouped, how='left', on='product_uid')
print("Datasets merged.")

print("Applying stemming...")
df_all['search_term'] = df_all['search_term'].progress_apply(fast_stem_sentence)
df_all['product_title'] = df_all['product_title'].progress_apply(fast_stem_sentence)
df_all['product_description'] = df_all['product_description'].progress_apply(fast_stem_sentence)
df_all['attr_values'] = df_all['attr_values'].fillna("").progress_apply(fast_stem_sentence)
print("Stemming applied.")

print("Engineering features...")
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
df_all['word_in_attr'] = df_all.apply(lambda row: str_common_word(row['search_term'], row['attr_values']), axis=1)
print("Feature engineering complete.")

print("Calculating TF-IDF cosine similarity...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
corpus = df_all['search_term'] + " " + df_all['product_title'] + " " + df_all['product_description']
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
search_term_tfidf = tfidf_vectorizer.transform(df_all['search_term'])
product_info_tfidf = tfidf_vectorizer.transform(df_all['product_title'] + " " + df_all['product_description'])
df_all['tfidf_cosine_sim'] = [
    cosine_similarity(search_term_tfidf[i], product_info_tfidf[i])[0][0] for i in range(search_term_tfidf.shape[0])
]
print("TF-IDF cosine similarity calculated.")

print("Dropping text columns...")
df_all.drop(['search_term', 'product_title', 'product_description', 'product_info', 'attr_values'], axis=1, inplace=True)

print("Splitting data...")
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

y = df_train['relevance'].values
X = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

print("Creating train-validation split...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to compare
models = {
    "RandomForestRegressor": RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=15, max_depth=6, random_state=0),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=15, max_depth=6, random_state=0),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=15, max_depth=6, random_state=0),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name} with BaggingRegressor meta-estimator...")
    bagging = BaggingRegressor(model, n_estimators=45, max_samples=0.1, random_state=25)
    start_time = time.time()
    bagging.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_val_pred = bagging.predict(X_val)
    pred_time = time.time() - start_time
    
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"{name} RMSE: {rmse:.4f}")
    print(f"{name} training time: {train_time:.2f}s, prediction time: {pred_time:.2f}s")
    
    results.append({
        "model": name,
        "rmse": rmse,
        "train_time": train_time,
        "pred_time": pred_time,
        "model_obj": bagging,
    })

# Select best model based on RMSE
best_model = min(results, key=lambda x: x['rmse'])
print(f"\nBest model by RMSE: {best_model['model']} with RMSE={best_model['rmse']:.4f}")

print("\nPredicting on test set with best model...")
y_pred = best_model['model_obj'].predict(X_test)

print("Saving submission...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_best_model.csv', index=False)
print("Submission saved as submission_best_model.csv")