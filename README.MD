# 1. Navigeren naar juiste map

cd documents (optioneel)
cd DS

# 2. Virtual environment maken of Conda

python -m venv venv

conda create -n DS3 python=3.11

# 3. Virtual environment activeren

On Mac: source venv/bin/activate  
On Windows: .\venv\Scripts\activate

conda activate DS3

# 4. Requirements installeren

conda install pandas numpy scikit-learn matplotlib seaborn nltk
pip install pandas numpy scikit-learn matplotlib seaborn nltk