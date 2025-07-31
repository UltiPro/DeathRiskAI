# DeathRiskAI
DeathRiskAI created by Patryk 'UltiPro' Wójtowicz using Scikit Learn and Tensorflow.

The project was developed as part of learning about machine learning and deep learning techniques. The goal of the project was to build two predictive models: one using the Scikit-learn library and the other using the TensorFlow library to estimate the risk of patient death based on vital signs and clinical parameters.

The dataset used in the project comes from the WiDS Datathon 2020 competition on Kaggle, organized by 

> ***Karen Matthys, Marzyeh Ghassemi, Meredith Lee, NehaGoel, Sharada Kalanidhi, and sumalaika. WiDS Datathon 2020.<br/>
> https://kaggle.com/competitions/widsdatathon2020, 2020. Kaggle.***

The project followed a complete data science pipeline, including data preprocessing, feature transformation, hyperparameter tuning, model training, and evaluation.

⚠️ Warning:

> Feature transformations were not designed for production use, since the project was purely educational and aimed at developing skills in data analysis and artificial intelligence.

# ⚙️ Dependencies and Usage

### 📦 Dependencies

> cd "./DeathRiskAI/"

> cat requirements.txt

### 📥 Installation

> cd "./DeathRiskAI/"

> pip install -r requirements.txt

### 🔸 Data processing

> cd "./DeathRiskAI/data/"

> python preprocessing.py

### 🔸 Feature transformation

> cd "./DeathRiskAI/models/"

> python feature_transformation.py

### 🔸 Scikit Learn model pipeline

> cd "./DeathRiskAI/models/scikit-learn/"

> ./pipeline.sh

### 🔸 Tensorflow model pipeline

> cd "./DeathRiskAI/models/tensorflow/"

> ./pipeline.sh
