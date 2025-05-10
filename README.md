# RUControversial: Improving Trend Prediction

## Overview
RUControversial is a fine-tuned BERT model that helps predict how people will react to social media posts before they're published. Specifically, the model classifies content into categories similar to the r/AmITheAsshole subreddit verdicts: "YTA" (You're the A**hole), "NTA" (Not the A**hole), "NAH" (No A**holes Here), or "ESH" (Everyone Sucks Here).

## Motivation
Social media users often can't predict how their posts will be received by others. RUControversial aims to help users gauge potential reactions to their content before posting, potentially helping them avoid negative backlash or controversy.

## Dataset
The model was trained on the Iterative AITA Dataset compiled by Elle O'Brien, containing approximately 97,000 posts from r/AITA with associated verdicts. Due to computational constraints, we used a subset of 19,722 posts for our final model.

## Models
### Primary Model: BERT Classifier
- **Architecture**: Fine-tuned DeBERTa-v3-small (44M parameters)
- **Features**:
  - Disentangled attention mechanism
  - Enhanced mask decoder
  - Custom training strategy with frozen backbone for first epoch
  - Focal Loss to address class imbalance
  - Linear warmup and decay learning rate scheduler

### Baseline Model: Logistic Regression
- TF-IDF vectorized posts
- SMOTE for handling class imbalance
- Hyperparameter tuning via GridSearchCV

## Results
- BERT Model: 63.4% accuracy, 50.37% F1 score
- Logistic Regression: 48.67% accuracy, 49% F1 score
- The class imbalance in the dataset posed challenges, with the model struggling on minority classes despite mitigation techniques

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yashc73080/RUControversial.git
cd RUControversial

# Install dependencies
pip install -r requirements.txt
```

## Files
- `bert_model.py`: BERT classifier implementation
- `jobscript.sh`: Bash script to run a training job on the Amarel compute cluster  
- `Notebooks/use_BERT_model.ipynb`: Notebook to load and test a trained BERT model
- `Notebooks/Logistic_Regression_Model.ipynb`: Logistic regression model
- `Notebooks/Data_Cleaning_and_Preprocessing.py`: Functions for data cleaning and preparation

## Future Work
- Train on larger datasets
- Experiment with larger language models (e.g., Llama)
- Improve handling of class imbalance
- Extend to other social media platforms

## Contributors
- Yash Chennawar
- Zayd Mukaddam

## Acknowledgements
- Elle O'Brien for the Iterative AITA Dataset
- Rutgers University's Amarel compute cluster for GPU resources
- HuggingFace for transformer models
