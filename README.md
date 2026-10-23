Phishing URL Detector

A machine learning tool that classifies a URL as phishing or legitimate using only its text. It never visits the site, so it can screen URLs quickly and safely.

How It Works

The detector combines two models and averages their predicted probabilities.

1. Random Forest on handcrafted features. Each URL is converted into 17 numeric features, including:

URL, host, path, and query length
Counts of dots, hyphens, digits, and special characters
Subdomain count, whether the host is an IP address, and whether an @ symbol is present
HTTPS use, double slashes in the path, and URL shorteners
Suspicious keywords such as "login", "verify", "secure", and "paypal"
Shannon entropy of the URL and the host

2. Logistic regression on character n-grams. Character 3- to 5-grams are weighted with TF-IDF, so the model learns recurring fragments of phishing URLs without relying on hand-picked rules.

Threshold selection. The data is split 60% training, 20% validation, and 20% test. The decision threshold is chosen on the validation set: it is the highest-recall point where precision is at least 0.85. If no point reaches that precision, the threshold with the best F1 score is used. The test set is used only for the final evaluation.

Dataset

The model is trained on 507,196 labeled URLs from the Kaggle "Phishing Site URLs" dataset: 114,299 phishing and 392,897 legitimate.

Place the data in a folder named dataset.csv next to phishing_detector.py, with the file phishing_site_urls.csv inside it. The file needs two columns, URL and Label (column names are not case-sensitive). The labels bad, phishing, malicious, 1, and -1 are treated as phishing, and good, benign, legitimate, and 0 are treated as legitimate.

Results

Results on the 20% held-out test set (101,440 URLs), with the threshold chosen on the validation set:

Metric	Value
Accuracy	95.7% (0.9565)
Phishing recall	0.98
Phishing precision	0.85
Legitimate recall	0.95
Legitimate precision	0.99
Chosen threshold	0.280

Confusion matrix (rows are actual labels, columns are predictions):

	Predicted legitimate	Predicted phishing
Actual legitimate	74,576	4,004
Actual phishing	406	22,454

At the default 0.5 threshold, accuracy was 90.7% for the Random Forest, 97.8% for the TF-IDF logistic regression, and 97.9% for the ensemble. The tuned threshold gives up some accuracy to catch more phishing URLs. That trade-off is the main design choice of this model.

Setup
bash
pip install pandas scikit-learn joblib scipy
Usage

Train the model. This takes a few minutes on the full dataset and creates phishing_model_v2.joblib:

bash
python phishing_detector.py train

Classify a URL:

bash
python phishing_detector.py predict "http://paypal-login.verify-account.xyz/signin"

Example output (the probability shown is illustrative):

http://paypal-login.verify-account.xyz/signin
-> PHISHING (phishing probability: 97.12%, threshold: 0.28)
Limitations
The model uses URL text only. Phishing sites on clean-looking domains can get through. In testing, 406 of 22,860 phishing URLs were missed.
The threshold favors recall, so 4,004 legitimate URLs (about 5% of legitimate test URLs) were flagged as phishing.
The model is trained on one dataset. Accuracy on URLs from other sources may be lower.
Files
phishing_detector.py: feature extraction, training, evaluation, and prediction
dataset.csv/: folder containing phishing_site_urls.csv (not included; download separately)
phishing_model_v2.joblib: trained model, created by train (large, so keep it out of the repository)
