Phishing URL Detector

A machine learning tool that classifies a URL as phishing or legitimate using only its text. It never visits the site, so it can screen URLs quickly and safely.

How It Works

The detector combines two models and averages their predicted probabilities.

1. Random Forest on handcrafted features. Each URL is converted into 18 numeric features, including:

URL, host, path, and query length
Counts of dots, hyphens, digits, and special characters
Subdomain count, whether the host is an IP address, and whether an @ symbol is present
HTTPS use, double slashes in the path, and URL shorteners
Suspicious keywords such as "login", "verify", "secure", and "paypal"
Shannon entropy of the URL and the host, since random-looking domains score higher

2. Logistic regression on character n-grams. Character 3- to 5-grams are weighted with TF-IDF, so the model learns recurring fragments of phishing URLs without relying on hand-picked rules.

Threshold selection. The data is split 60% training, 20% validation, and 20% test. The decision threshold is chosen on the validation set: it is the highest-recall point where precision is at least 0.85. If no point reaches that precision, the threshold with the best F1 score is used. The test set is used only for the final evaluation, so the reported numbers are not tuned on it.

Dataset

The model is trained on about 507,000 labeled URLs from the Kaggle "Phishing Site URLs" dataset (about 114,000 phishing and 393,000 legitimate).

Save the CSV as dataset.csv in the project folder. It needs two columns, url and label. The labels bad, phishing, malicious, 1, and -1 are treated as phishing, and good, benign, legitimate, and 0 are treated as legitimate.

Results

Results on the 20% held-out test set (threshold chosen on validation):

Metric	Value
Accuracy	[run train and enter the printed value]
Phishing recall	[enter value]
Phishing precision	[enter value]
Legitimate recall	[enter value]
Legitimate precision	[enter value]
Chosen threshold	[enter value]

The full confusion matrix and classification report are printed after training.

Setup
bash
pip install pandas scikit-learn joblib scipy
Usage

Train the model. This creates phishing_model_v2.joblib:

bash
python phishing_detector.py train

Classify a URL:

bash
python phishing_detector.py predict "http://paypal-login.verify-account.xyz/signin"

Example output:

http://paypal-login.verify-account.xyz/signin
-> PHISHING (phishing probability: 97.12%, threshold: 0.62)

The probability and threshold above are examples. Your values will depend on training.

Limitations
The model uses URL text only. Phishing sites on clean-looking domains, or with legitimate-looking paths, can get through. Adding domain age, page content, or blocklist checks would help.
The threshold favors precision and recall in the ranges above. Some legitimate URLs will be flagged, and some phishing URLs will still be missed.
The model is trained on one dataset. Accuracy on URLs from other sources may be lower.
Files
phishing_detector.py: feature extraction, training, evaluation, and prediction
dataset.csv: labeled URLs (not included; download separately)
phishing_model_v2.joblib: trained model, created by train
