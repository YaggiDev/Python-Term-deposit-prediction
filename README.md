# Python-Term-deposit-prediction
The main objective of the project is to learn data analysis in python such as pandas, scikit-learn, numpy and other data handling libraries. Project also includes visualization aspects of data analysis topic.

# Technologies in use:
- Pandas
- sklearn
- matplotlib
- seaborn
- numpy
- Pickle
- Pycharm

# Problem describe
The data used in project is related with direct marketing campaign of a Portuguese banking institution. The marketing campaign were based on phone calls. Our main task is to make a prediction model to answer the question if the client has subscribed ('yes') or not ('no') a term deposit.

# Data information
Data is located in bank-full.csv and based on this file, splitted into train and test data in proportion 0,9 to 0,1. As a result we have 40689 rows of training data and 4521 rows of testing data. 
## Bank client data:
1. age
2. job: type of job (categorical: ‘admin’, ‘blue-collar’,  ‘enterpreneur’, ‘housemaid’, management’, ‘retired’, ‘self-employed’, ‘services’, ‘student’, ‘technician’, ‘unemployed’, ‘unknown’)
3. marital: marital status (categorical: ‘divorced’, ‘married’, ‘single’, ‘unknown’; note: ‘divorced’ means divorced or widowed)
4. education (categorical: ‘basic.4y’, ‘basic.6y’, ‘basic.9y’, ‘high.school’, ‘illiterate’, ‘professional.course’, ‘university.degree’, ‘unknown’)
5. default: has credit in default (categorical: ‘no’, ‘yes’, ‘unknown’)
6. balance
7. housing: has housing loan? (categorial: ‘no’, ‘yes’, ‘unknown’)
8. loan: has personal loan? (categorial: ‘no’, ‘yes’, ‘unknown’)
9. contact: contact communication type (categorical: ‘cellular’, ‘telephone’)
10. month: last contact month of the year (categorical)
11. day_of_week: last contact day of the week (categorical)
12. duration: last contact duration in seconds. Important note: this attribute highly affects the output target
13. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
14. pdays: number of days that passed by after the client was last contacted from previous campaign (numeric: 999 means client was not previously contacted)
15. previous: number of contacts performed before this campaign and for this client
16. poutcome: outcome of the previous marketing campaign (categorical: ‘failure’, ‘nonexistent’, ‘success’)

Output variable (desired target):  
17. y - has the client subscribed a term deposit? (binary: "yes","no")

There are no missing values for each attribute.
