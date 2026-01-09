Class Imbalance in E-Commerce Order Cancellation Prediction
Overview
This project investigates the challenges of class imbalance in predicting e-commerce order cancellations using the Brazilian E-Commerce Public Dataset by Olist. Order cancellation prediction represents a critical business problem where standard machine learning approaches often fail due to severe class imbalance—successful orders vastly outnumber cancellations, yet identifying potential cancellations has disproportionate business value.
The research examines how traditional performance metrics can be misleading in imbalanced settings and demonstrates that models achieving high overall accuracy may fail catastrophically on the minority class. Through systematic evaluation using appropriate metrics and resampling techniques, this work provides empirical evidence for the importance of class-aware modeling in real-world e-commerce applications.

Dataset: Brazilian E-Commerce Public Dataset by Olist
Problem Type: Binary classification with severe class imbalance
Business Context: Predicting order cancellations to enable proactive customer retention

Project Motivation
The Class Imbalance Problem
In e-commerce order management, cancellations typically represent 1-5% of total orders, creating a natural class imbalance problem. This imbalance has several important implications:

Asymmetric Costs: Failing to predict a cancellation (false negative) incurs opportunity costs in customer retention, while false positives waste intervention resources but are less costly.
Misleading Metrics: Standard accuracy can exceed 95% by simply predicting "no cancellation" for all orders, masking complete failure on the minority class.
Learning Bias: Standard loss functions optimize overall error rate, causing models to ignore minority class patterns in favor of majority class performance.

Why This Matters for Machine Learning Projects
Class imbalance exposes fundamental challenges in supervised learning:

Representation Learning: How do models learn meaningful representations when minority class examples are scarce?
Decision Boundaries: Do learned boundaries reflect true class distributions or training set imbalance?
Evaluation Methodology: What constitutes "good performance" when classes have different costs and prevalences?

This project addresses these questions through systematic experimentation with resampling techniques, cost-sensitive learning, and appropriate evaluation protocols.

Dataset Description
Source and Scope
The Brazilian E-Commerce Public Dataset by Olist contains real-world transactional data from 100,000 orders placed between 2016 and 2018. The dataset includes:

Order status labels (delivered, canceled, unavailable)
Temporal features (purchase timestamps, delivery estimates)
Customer location data (city, state)
Product information (category, dimensions, photos)
Review scores and timestamps
Payment details and logistics information

Key Tables
olist_orders_dataset.csv          # Order status and temporal data
olist_order_items_dataset.csv     # Product-level details and pricing
olist_order_reviews_dataset.csv   # Customer satisfaction scores
olist_customers_dataset.csv       # Customer demographics
olist_products_dataset.csv        # Product attributes
olist_sellers_dataset.csv         # Seller location
olist_order_payments_dataset.csv  # Payment methods and values
Target Variable Construction
Orders are classified into two categories:

Negative Class (0): Successfully delivered orders
Positive Class (1): Canceled or unavailable orders

This binary formulation captures the business-critical distinction between completed and failed transactions, where the positive class represents revenue loss and customer dissatisfaction.
Methodology
Feature Engineering
The multi-table structure enables rich feature construction:
Temporal Features:

Purchase timestamp components (day of week, month, hour)
Estimated vs. actual delivery windows
Time between purchase and approval

Customer Features:

Geographic location (state, city)
Historical order count (if applicable)
Customer-seller distance

Product Features:

Product category
Price point
Product dimensions and weight
Number of product photos

Review Features:

Review score (when available)
Review comment sentiment
Time to review submission

Payment Features:

Payment method
Number of payment installments
Payment value

Handling Class Imbalance
The project systematically evaluates multiple approaches:
1. Resampling Techniques

Random Undersampling: Reduce majority class to balance distribution
Random Oversampling: Replicate minority class examples
SMOTE (Synthetic Minority Over-sampling Technique): Generate synthetic minority examples
ADASYN (Adaptive Synthetic Sampling): Density-based synthetic generation

2. Cost-Sensitive Learning

Assign misclassification costs proportional to class imbalance
Modify loss functions to penalize minority class errors more heavily

3. Ensemble Methods

Balanced Random Forest: Bootstrap sampling with class balancing
EasyEnsemble: Ensemble of classifiers trained on balanced subsets
BalancedBaggingClassifier: Bagging with random undersampling

Model Architecture
Models evaluated include:

Logistic Regression (baseline)
Random Forest
Gradient Boosting Machines (XGBoost, LightGBM)
Neural Networks with class weighting

Evaluation Protocol
Critical to this research is the use of appropriate evaluation metrics:
Standard Metrics (Inadequate for Imbalanced Data)

Accuracy: Proportion of correct predictions (misleading when imbalanced)
Error Rate: Overall misclassification rate (dominated by majority class)

Class-Aware Metrics (Appropriate for Imbalanced Data)

Precision: Of predicted cancellations, what fraction are true cancellations?
Recall (Sensitivity): Of actual cancellations, what fraction are detected?
F1-Score: Harmonic mean of precision and recall
AUC-ROC: Area under receiver operating characteristic curve
AUC-PR: Area under precision-recall curve (preferred for imbalanced data)
Matthews Correlation Coefficient: Balanced measure accounting for all confusion matrix elements

Business-Oriented Metrics

Cost-Weighted Accuracy: Accuracy weighted by business costs
Profit Curve: Net benefit as a function of decision threshold

Experimental Design

Baseline Establishment: Train models on imbalanced data with standard loss functions
Resampling Comparison: Evaluate each resampling technique independently
Cost-Sensitive Learning: Experiment with different cost ratios
Ensemble Methods: Test class-balanced ensemble approaches
Threshold Optimization: Find optimal decision thresholds for business objectives

All experiments use stratified k-fold cross-validation to ensure stable estimates despite class imbalance.
Key Project Questions

How severely does class imbalance degrade model performance on the minority class?

Quantify the gap between accuracy and minority-class recall
Demonstrate that high accuracy can coexist with near-zero minority class detection


Which resampling techniques are most effective for this problem?

Compare SMOTE, ADASYN, and random over/undersampling
Analyze trade-offs between precision and recall


How do different models handle class imbalance?

Compare tree-based methods vs. linear models
Evaluate neural networks with class weighting


What is the optimal evaluation protocol for imbalanced classification?

Show why accuracy is insufficient
Demonstrate superiority of PR-AUC over ROC-AUC for imbalanced data


How should decision thresholds be selected in production systems?

Map business costs to optimal thresholds
Quantify precision-recall trade-offs



Expected Findings
Based on prior work in class imbalance, we expect:

Baseline models will achieve 95%+ accuracy while detecting <50% of cancellations, demonstrating the accuracy paradox in imbalanced learning
SMOTE and ADASYN will outperform random resampling by generating informative synthetic examples rather than duplicating existing minority samples
Precision-recall curves will reveal critical trade-offs obscured by ROC curves, showing that high recall requires accepting lower precision
Cost-sensitive learning will outperform resampling when business costs are known and accurately specified
Ensemble methods will provide robust performance by aggregating predictions from multiple balanced classifiers

Implementation
Dependencies
pythonnumpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
Repository Structure
ecommerce-class-imbalance/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Original Olist CSV files
│   └── processed/              # Cleaned and merged datasets
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_resampling_experiments.ipynb
│   └── 05_evaluation_analysis.ipynb
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── resampling.py           # Resampling implementations
│   ├── models.py               # Model training and evaluation
│   └── visualization.py        # Plotting utilities
├── results/
│   ├── figures/                # Performance plots
│   └── metrics/                # Evaluation metrics tables
└── docs/
    └── technical_report.pdf    # Detailed methodology and findings
Usage
bash# Clone repository
git clone https://github.com/gulusan2021/ecommerce-class-imbalance.git
cd ecommerce-class-imbalance

# Install dependencies
pip install -r requirements.txt

# Download Olist dataset from Kaggle and place in data/raw/

# Run analysis pipeline
python src/data_processing.py
python src/feature_engineering.py
python src/models.py
Contributions to Machine Learning Practice
This project contributes to ML practice on class imbalance by:

Empirical Evidence on Metric Selection: Demonstrating why accuracy is insufficient and establishing best practices for imbalanced evaluation
Resampling Technique Comparison: Providing systematic comparison of resampling methods on real-world e-commerce data
Business-ML Interface: Bridging machine learning methodology with business cost structures through cost-sensitive evaluation
Reproducible Analysis: Providing complete code and methodology for practitioners studying class imbalance
Domain Application: Extending class imbalance work into e-commerce order management, a critical but understudied application area

Related Work
Class Imbalance Foundations:

Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research
He & Garcia (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering

Evaluation Metrics:

Saito & Rehmsmeier (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE
Davis & Goadrich (2006). The Relationship Between Precision-Recall and ROC Curves. ICML

Cost-Sensitive Learning:

Elkan (2001). The Foundations of Cost-Sensitive Learning. IJCAI
Ling & Sheng (2008). Cost-Sensitive Learning. Encyclopedia of Machine Learning

E-Commerce Applications:

Literature on customer churn prediction, fraud detection, and conversion optimization

Future Directions

Deep Learning Approaches: Investigate attention mechanisms for imbalanced sequential data
Causal Inference: Identify causal factors driving cancellations vs. predictive correlations
Active Learning: Develop strategies for selectively labeling uncertain minority class examples
Temporal Analysis: Study how class imbalance and model performance evolve over time
Multi-Task Learning: Jointly predict cancellation, delivery delays, and customer satisfaction

Author
Gulusan Erdogan-Ozgul
e.gulusan@gmail.com


Acknowledgments

Olist for providing the Brazilian E-Commerce dataset
The scikit-learn and imbalanced-learn development teams
Researchers whose work on class imbalance informed this project

Citation
If you use this work in your projects, please cite:
bibtex@misc{yourlastname2025ecommerce,
  author = {Your Name},
  title = {Class Imbalance in E-Commerce Order Cancellation Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/gulusan2021/ecommerce-class-imbalance}
}

Keywords: class imbalance, imbalanced learning, e-commerce, order cancellation prediction, SMOTE, cost-sensitive learning, precision-recall analysis, machine learning evaluation
