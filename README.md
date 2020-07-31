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
Data is located in bank-full.csv and splitted into train and test data in proportion 0,9 to 0,1. As a result we have 40689 rows of training data and 4521 rows of testing data. 
# Bank client data:

**Customer-related variables:**  
1. age (numeric)  
2. job: type of job (categorical: ‘admin’, ‘blue-collar’,  ‘enterpreneur’, ‘housemaid’, management’, ‘retired’, ‘self-employed’, ‘services’, ‘student’, ‘technician’, ‘unemployed’, ‘unknown’)  
3. marital: marital status (categorical: ‘divorced’, ‘married’, ‘single’, ‘unknown’; note: ‘divorced’ means divorced or widowed)  
4. education (categorical: ‘basic.4y’, ‘basic.6y’, ‘basic.9y’, ‘high.school’, ‘illiterate’, ‘professional.course’, ‘university.degree’, ‘unknown’)  
5. default: has credit in default (categorical: ‘no’, ‘yes’, ‘unknown’)  
6. housing: has housing loan? (categorial: ‘no’, ‘yes’, ‘unknown’)  
7. loan: has personal loan? (categorial: ‘no’, ‘yes’, ‘unknown’)  

**Last contact related variables:**  
8. contact: contact communication type (categorical: ‘cellular’, ‘telephone’)  
9. month: last contact month of the year (categorical)  
10. day_of_week: last contact day of the week (categorical)  
11. duration: last contact duration in seconds. Important note: this attribute highly affects the output target  

**Other variables:**  
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
13. pdays: number of days that passed by after the client was last contacted from previous campaign (numeric: 999 means client was not previously contacted)  
14. previous: number of contacts performed before this campaign and for this client  
15. poutcome: outcome of the previous marketing campaign (categorical: ‘failure’, ‘nonexistent’, ‘success’)  

**Variables with socio-economic context:**  
16. emp.var.rate: employment variation rate (numeric)  
17. cons.price.idx: consumer price index (numeric)  
18. cons.conf.idx: consumer confidence index (numeric)  
19. euribor3m: euribor 3 month rate (numeric)  
20. nr.employed: number of employees hired (numeric)  

**Output variable (desired target):**
  21. y - has the client subscribed a term deposit? (binary: "yes","no")

There are no missing values for each attribute.

# Feature engineering
First step in future engineering was to map categorical variables. In this step following columns were mapped:
- Target
- Marital
- Education
- Default
- Contact
- Loan
- Poutcome  

On the basis of the Month column there were created binary variables for each month, so as the result we added 12 new columns to our dataframe. Thus the initial Month column was dropped.
Same solution was also applied to Job column (we obtained 12 new binary variables). 
