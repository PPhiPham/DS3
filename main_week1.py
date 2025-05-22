import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

tqdm.pandas()
stemmer = SnowballStemmer('english')

print("Load datasets")
df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
print("Datasets loaded")

print("############## Data Exploration ##############")

amount_pairs = df_train.shape[0]
print(f"1. Total product-query pairs: {amount_pairs}")

uniq_prod = df_train['product_uid'].nunique()
print(f"2. Total Unique products: {uniq_prod}")

top_prod = df_train['product_uid'].value_counts().head(5)
print("3. Top 5 most products:")
for pid, count in top_prod.items():
    print(f"   - Product UID {pid}: {count} occurrences")

mean_rel = df_train['relevance'].mean()
median_rel = df_train['relevance'].median()
std_rel = df_train['relevance'].std()
print(f"4. Mean: {mean_rel:.2f}, Median: {median_rel:.2f}, Std: {std_rel:.2f}")

df_attr = pd.read_csv('data/attributes.csv', encoding='ISO-8859-1')
df_attr['name'] = df_attr['name'].str.strip().str.lower()
brands = df_attr[df_attr['name'].str.contains('brand name', na=False)]
top_brands = brands['value'].value_counts().head(5)

print("5. Top 5 most common brands:")
for brand, count in top_brands.items():
    print(f"   - {brand}: {count} occurrences")

num_train = df_train.shape[0]

@lru_cache(maxsize=10000)
def cached_stem(word):
    return stemmer.stem(word)

def fast_stem_sentence(s):
    return " ".join([cached_stem(word) for word in str(s).lower().split()])

def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())

print("Merging train and test data")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
print("Dat merged")

print("Applying stemming with progress bars")
df_all['search_term'] = df_all['search_term'].progress_apply(fast_stem_sentence)
df_all['product_title'] = df_all['product_title'].progress_apply(fast_stem_sentence)
df_all['product_description'] = df_all['product_description'].progress_apply(fast_stem_sentence)
print("Stemm complete")

print("Engineering features")
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
print("Feature engineering complete")

print("Cleaning up")
df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1, inplace=True)

print("Splitting back to train/test sets")
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df_train['relevance'], bins=20, kde=True)
plt.title("Histogram of Relevance")

plt.subplot(1, 2, 2)
sns.scatterplot(x=df_train['len_of_query'], y=df_train['relevance'], alpha=0.5)
plt.title("Relevance vs. Query Length")
plt.xlabel("Length of Search Query")
plt.ylabel("Relevance Score")

plt.tight_layout()
plt.show()

y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values
print("Data split complete.")

print("Splitting training set into train/validation for RMSE")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training model on train split")
rfr = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rfr, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_tr, y_tr)

print("Predicting on validation set")
y_val_pred = clf.predict(X_val)
rmse = root_mean_squared_error(y_val, y_val_pred)
print(f"Validation RMSE: {rmse:.4f}")

print("Retraining model on full training data")
clf.fit(X_train, y_train)

print("Generating predictions on test set")
y_pred = clf.predict(X_test)

print("Saving submission file")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")