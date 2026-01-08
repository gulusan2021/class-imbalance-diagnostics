# class-imbalance-diagnostics
Diagnostics for predictive models under severe class imbalance using ROC, PR curves, and threshold analysis.

Project Name: Predicting Rare PRoduct Returns in e-Commerce.
Status: Active Development
Description: Advanced ML Techniques for handling severe class imbalance in predicting modeling.

Project Overview
This project tackles one of the most challenging problems in machine learning: predicting extremely rare events in the context of e-commerce product returns. With return rates typically between 2-5% of all transactions, this represents an extreme class imbalance problem (19:1 to 49:1 ratio) that causes standard ML approaches to fail completely.
Key Objectives

Demonstrate the discovery process: Show how seemingly successful models (98% accuracy) can hide complete failure
Develop robust solutions: Apply advanced techniques for extreme class imbalance
Optimize for business value: Translate technical metrics into actionable business insights
Create reproducible framework: Build reusable methodology for rare event prediction

Why This Matters

Business Impact: Product returns cost e-commerce companies billions annually
Technical Challenge: Standard ML models achieve 98% accuracy but identify 0% of returns
Real-World Relevance: Demonstrates handling of production ML challenges

Business Problem
Context
E-commerce platforms face significant costs from product returns:
Cost CategoryImpactAnnual Cost (Industry)Direct CostsShipping, restocking, depreciation$550+ billion globallyOperationalProcessing, quality checks, customer serviceHigh overheadStrategicCustomer satisfaction, brand reputation Intangible but critical
Objective
Build a predictive model to identify which orders are likely to be returned before shipment, enabling:

- Proactive quality inspection
- Targeted customer communication
- Optimized inventory management
- Reduced return processing costs

Success Criteria

Recall > 60%: Catch majority of returns before they happen
Precision > 10%: Manageable false alarm rate
ROI Positive: Cost savings exceed implementation costs
Explainable: Provide actionable insights for business operations


Technical Challenge
The Problem: Extreme Class Imbalance

Unlike typical classification problems, rare product returns present extreme imbalance:
Typical Classification:     Rare Returns:
Classes: 50/50             Classes: 98/2
Easy to solve              Extremely challenging

Standard Churn (27%):      Product Returns (2%):
Manageable imbalance       Extreme imbalance
SMOTE works well           SMOTE insufficient
Why Standard ML Fails
The Accuracy Paradox:
python# Naive model that predicts "No Return" for EVERYTHING
predictions = ['No Return'] * 10000

# Result:
# Accuracy: 98%  ← Looks amazing!
# Recall on returns: 0%  ← Complete failure!
Root Causes:

Overwhelming majority class: 98% of samples are non-returns
Weak signals: Returns are inherently unpredictable
Insufficient minority samples: Only 320 returns in 16,000 training samples
Metric misleading: Accuracy completely fails as evaluation metric

Real-World Implications
This is harder than customer churn prediction because:

Churn: 27% minority class (manageable with SMOTE)
Returns: 2% minority class (requires advanced techniques)
Churn: Strong behavioral patterns exist
Returns: Weak, noisy signals


Dataset
Primary Dataset: Brazilian E-Commerce (Olist)
Source: Kaggle - Brazilian E-Commerce Public Dataset by Olist
Specifications:

Size: 100,000+ orders from 2016-2018
Coverage: Multiple product categories
Features:

Customer data (location, purchase history)
Product data (category, price, dimensions)
Seller data (reputation, location)
Order data (payment, shipping, reviews)
Delivery data (estimated vs. actual)



Target Variable Creation:

Orders with very low ratings (1-2 stars) + negative review keywords
Cancelled orders
Delivery issues flagged by customers
Expected Return Rate: 3-5% (after engineering)

Alternative Dataset: E-Commerce Customer Data
Source: Kaggle - E-Commerce Data
Specifications:

Size: 500,000+ transactions
Coverage: UK-based online retailer
Features: Product codes, quantities, prices, customer IDs
Returns: Identified by negative quantities

Synthetic Dataset (Backup)
For reproducibility and testing, a synthetic dataset generator is included that creates realistic return patterns with configurable:

Return rate (default: 2%)
Feature correlations (price, quality, customer history)
Seasonal patterns
Customer segmentation


