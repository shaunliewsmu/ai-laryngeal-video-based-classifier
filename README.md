# Video Based Classifier for AI Laryngeal

## First model: Vivit

### How to run the model training

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

open `training.ipynb` and run the code accordingly.

## make sure that you have Weights and Bias Account ready to access all the saved artifacts and logs

### Dataset Description:

Overall Distribution:


The dataset has an imbalanced class distribution
Referral cases (Label 1): 66.67% of total data
Non-referral cases (Label 0): 33.33% of total data


Split Distributions:

Training Set (92 videos, ~70% of total):

Referral: 61 videos (66.30%)
Non-referral: 31 videos (33.70%)
The class ratio closely matches the original distribution

Validation Set (20 videos, ~15% of total):

Referral: 14 videos (70.00%)
Non-referral: 6 videos (30.00%)
Slightly higher proportion of referral cases compared to overall

Test Set (20 videos, ~15% of total):

Referral: 13 videos (65.00%)
Non-referral: 7 videos (35.00%)
Distribution very close to the original