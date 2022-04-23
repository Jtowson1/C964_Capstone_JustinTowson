import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    header = st.container()
    dataset = st.container()
    features = st.container()
    modelTraining = st.container()

    df = pd.read_csv('Main/loan-train.csv',
                     converters={'Gender': lambda x: int(x == 'Male'), 'Married': lambda x: int(x == 'Yes'),
                                 'Self_Employed': lambda x: int(x == 'Yes'),
                                 'Education': lambda x: int(x == 'Graduate'),
                                 'Property_Area': lambda x: int(x == 'Rural'), 'Loan_Status': lambda x: int(x == 'Y')})
    x = df.drop(columns=['Loan_ID', 'Credit_History', 'Property_Area', 'Loan_Status'])
    y = df['Loan_Status']

    model = DecisionTreeClassifier()
    model.fit(x, y)

    Income = df['ApplicantIncome']
    Loan = df['LoanAmount']
    X = np.array(list(Income))
    Y = np.array(list(Loan))
    Gender = df['Gender']
    Married = df['Married']
    Education = df['Education']
    men_Approval = []
    G = list(Gender)
    M = list(Married)
    Gr = list(Education)

    classes = ['Approved', 'Not Approved']
    loan_status = df['Loan_Status']
    L = list(loan_status)

    bar_Data = ['Male', 'Female', 'Married', 'Not Married', 'Grad', 'Not Grad']
    plt.bar = pd.DataFrame([G.count(1), G.count(0), M.count(1), M.count(0), Gr.count(1), Gr.count(0)], bar_Data)

    with header:
        st.title('Welcome to Loan Information and Estimator Brought to you by Buy Big Now')
        st.text('This project is based of the Loan Eligible Dataset from Kaggle contributed by')
        st.text('Vikas Ukani, the data set provided multiple questions about the user which was')
        st.text('used to train the algorithm to determine if someone was eligible for a home loan.')
        st.subheader('Scatter Chart Showing Income(X-value) vs Home Loan Requested in Thousands(Y-Value)')
        st.text('The visual below shows the ratio of income from customers versus the home loan that')
        st.text('they were requesting. The home loan is by the thousands and the income is gross')
        st.text('monthly. We can determine from this that the majority of people are looking for')
        st.text('homes under 300,000 and most customers make under 120,000 dollars per year.')
        plt.scatter(X, Y)
        st.pyplot(plt)
    with dataset:
        st.subheader('Bar chart shows demographics of people that have applied for loans')
        st.text('The visual below shows multiple demographics of people that have applied for a loan.')
        st.text('We can determine from this graphic that males apply more than females, graduates more')
        st.text('than non-graduates, and people who are married more than single or separated people.')
        st.bar_chart(plt.bar)
    with features:
        st.subheader('Pie Chart Descriptive Method')
        st.text('The pie chart belows contains data on the approval rating vs denial rating of loans.')
        st.text('Although this information is rather general, it is important to note that you have a')
        st.text('better chance of being approved for a loan with our company than denied.')
        plt.title('Approval vs Denial Percentage on Loans')
        pie = plt.pie([L.count(1), L.count(0)], labels=classes, autopct="%0.2f%%")
        st.pyplot(plt)

    with modelTraining:
        st.subheader('Loan Estimator')
        st.text('Please fill out all the information requested to see if you are eligible for a loan')
        sel_col, disp_col = st.columns(2)
        userName = sel_col.text_input('What is your name?', value='Name')
        gender = sel_col.selectbox('What is your Gender?', options=['Male', 'Female'], index=0)
        married = sel_col.selectbox('What is your relationship status?', options=['Single', 'Separated', 'Married'],
                                    index=0)
        dependents = sel_col.selectbox('How many dependents do you have?', options=[0, 1, 2, '3+'], index=0)
        education = sel_col.selectbox('What is your current level of schooling?',
                                      options=['High-school', 'Some College', 'Associates', 'Bachelors', 'Masters'],
                                      index=0)
        selfEmployed = sel_col.selectbox('Are you self-employed?', options=['No', 'Yes'], index=0)
        userIncome = sel_col.slider('What is your gross monthly income?', min_value=0, max_value=10000, value=500,
                                    step=50)
        coappIncome = sel_col.slider('What is your CoApplicants gross monthly income?', min_value=0, max_value=10000,
                                     value=500, step=50)
        loan_Amount = sel_col.slider('Please pick loan amount requested in thousands.', min_value=20, max_value=999,
                                     value=100)
        loan_Term = sel_col.selectbox('Select loan term of interest in months.', options=[180, 240, 360], index=0)

        if st.button('Submit'):
            if gender == 'Male':
                gender = 1
            else:
                gender = 0
            if married == 'Married':
                married = 1
            else:
                married = 0
            if education == 'Associates Degree' or 'Bachelors Degree' or 'Masters Degree':
                education = 1
            else:
                education = 0
            if selfEmployed == 'Yes':
                selfEmployed = 1
            else:
                selfEmployed = 0
            if dependents == '3+':
                dependents = 3
            predictions = model.predict(
                [[gender, married, dependents, education, selfEmployed, userIncome, coappIncome, loan_Amount,
                  loan_Term]])
            if predictions[0] == 1:
                st.write('Congratulations ' + userName + ', you are eligible for loan!')
            elif predictions[0] == 0:
                st.write(userName + ', we are sorry to inform you that you are not eligible for a loan at this time.')
        else:
            st.write('Loan Predictor is Ready!')
