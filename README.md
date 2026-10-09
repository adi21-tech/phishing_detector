Phishing URL Detector

A machine learning project that classifies web addresses as phishing or legitimate using only the text of the URL, with no need to visit the site.

Overview

Phishing sites imitate trusted websites to steal passwords and personal information. This project trains a model to spot the patterns that distinguish malicious URLs from safe ones, similar to the first-line checks used by browsers and email filters.

How it works
Dataset: Trained on about 507,000 labeled URLs (114,299 phishing, 392,897 legitimate).
Feature extraction: Each URL is converted into numeric features, including URL, host, path and query length; counts of dots, hyphens, digits and special characters; subdomain count; presence of an @ symbol; whether the host is an IP address; HTTPS usage; suspicious keywords (such as “login” and “verify”); and Shannon entropy of the URL and host.
Model: A Random Forest classifier is trained on 80% of the data and evaluated on the remaining 20%.
Prediction: The trained model is saved with joblib and can classify any new URL from the command line.
Results
Accuracy: 90.8% on 101,440 held-out URLs
Legitimate URLs: precision 0.92, recall 0.97
Phishing URLs: precision 0.87, recall 0.70
Most important features: URL entropy, host entropy, digit count, path length, suspicious keywords
Tech stack

Python, pandas, scikit-learn, joblib

Usage
pip install pandas scikit-learn joblib

python phishing_detector.py train
python phishing_detector.py predict "http://paypal-login.verify-account.xyz/signin"
Limitations and future work

The model relies on URL text alone, so phishing sites hosted on clean-looking domains can slip through (phishing recall is 70%). Possible improvements include class weighting or threshold tuning to raise recall, adding TLD and brand-impersonation features, and combining URL features with domain age, page content and blocklist data.

Short version (resume or portfolio)

Phishing URL Detector | Python, scikit-learn, pandas

Built a Random Forest classifier that detects phishing URLs from text-only features (length, entropy, digit and special-character counts, suspicious keywords), trained on 507K labeled URLs.
Achieved 90.8% accuracy (0.97 recall on legitimate, 0.70 on phishing) and packaged the model for command-line prediction.
