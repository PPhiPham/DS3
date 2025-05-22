import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from functools import lru_cache
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import randint

tqdm.pandas()
stemmer = SnowballStemmer('english')

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

print("Starting hyperparameter tuning with RandomizedSearchCV...")

base_rf = RandomForestRegressor(random_state=0)
model = BaggingRegressor(estimator=base_rf, n_estimators=45, max_samples=0.1, random_state=25)

param_dist = {
    'estimator__n_estimators': randint(10, 50),
    'estimator__max_depth': randint(3, 10),
    'estimator__min_samples_split': randint(2, 10),
    'estimator__min_samples_leaf': randint(1, 5),
    'estimator__max_features': ['auto', 'sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,          
    scoring='neg_root_mean_squared_error', 
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best hyperparameters found:")
print(random_search.best_params_)

print("Evaluating best model on validation set...")
best_model = random_search.best_estimator_
y_val_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE after tuning: {rmse:.4f}")

print("Predicting on test set...")
y_pred = best_model.predict(X_test)

print("Saving submission...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission2.csv', index=False)
print("Submission saved as submission2.csv")