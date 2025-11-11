🧠 Customer Satisfaction Prediction

📊 Project Overview
This project aims to predict customer satisfaction ratings (1–5 stars) based on customer support ticket data.
The goal is to help businesses understand and improve the factors that influence customer satisfaction.
Using machine learning techniques, the project performs exploratory data analysis (EDA), model training, and performance evaluation to identify the most effective algorithm for this multi-class classification problem.

📂 Dataset
File: customer_support_tickets.csv
Records: 2,769
Features: 2,843
Target Variable: Customer satisfaction rating (1–5 stars)

⚙️ Project Workflow
Data Preprocessing
Handling missing values
Encoding categorical features
Splitting data into training and testing sets
Exploratory Data Analysis (EDA)
Visualization of satisfaction distribution
Feature importance analysis
Correlation heatmaps and data patterns
Model Training & Evaluation
Algorithms tested:
Gradient Boosting
Random Forest
XGBoost
Ensemble (Voting Classifier)
Evaluation metrics:
Accuracy
F1-Score
Cross-validation performance
Confusion Matrix

🏆 Model Performance Summary
Model	Accuracy	F1-Score
Gradient Boosting	20.04%	0.1987
Ensemble (Voting)	18.59%	0.1857
Random Forest	18.23%	0.1781
XGBoost	17.69%	0.1763
Best Model: Gradient Boosting
Cross-Validation Accuracy: 19.93%
Weighted F1-Score: 0.1987

📈 Visual Results
Visualization	Description
	Exploratory Data Analysis summary
	Top influential features
	Accuracy comparison of models
	CV score visualization
	Model’s confusion matrix
  
🧾 Files in This Repository
File Name	Description
customer_support_final.ipynb	Jupyter Notebook with full code
customer_support_tickets.csv	Dataset used for training and testing
confusion_matrix.png	Confusion matrix visualization
cross_validation.png	Cross-validation accuracy plot
eda_plots.png	Exploratory data analysis plots
feature_importance.png	Feature importance chart
model_comparison.png	Model performance comparison
Model_Performance.txt	Detailed performance report
PROJECT_SUMMARY.txt	Project summary and completion report

🧩 Technologies Used
Python 3
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
XGBoost
GradientBoostingClassifier

📚 Future Improvements
Optimize model hyperparameters using GridSearchCV
Apply deep learning models (e.g., LSTM or BERT for ticket text data)
Improve feature engineering and dataset balance
Integrate into a web dashboard (e.g., Streamlit or Flask)

👨‍💻 Author
Neeraj Relangi
🎓 Recent graduate at Andhra University
📧 relangineeraj@gmail.com
