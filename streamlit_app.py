import streamlit as st
import pandas as pd
import seaborn as sns

# GLOBAL STUFF
pages = ["Home Page",
    "Machine Learning",
    "About Us"]

page = st.sidebar.selectbox("What do you wish to see?", pages)


# HOME PAGE
if page == pages[0]:
    st.title("ML On Click")

    st.header("Welcome to the **ML On Click** App!\n")

    st.subheader("\nEnjoy the beauty of Machine Learning algorithms, without having to write code!\n")
    st.write(
        "\n----\n",

        "\n#### Here for the first time?\n",

        "Fret not, this is the right place to get started with the app!\n",

        "\n#### Been here before?\n",

        "\nWe are extremely happy to see you again! Go ahead and do what you have to do!\n",

        "\n----\n",
    )

    st.subheader("About the App")
    with st.expander("Expand"):

        st.write(
            "\nBrilliant! Now that the fast and furious ones are busy with their work, let's have a look at how exactly you can use this app for ML!\n",
            
            "\n- Machine Learning, as the computer scientists call it, is the process of *training* a computer to perform certain tasks based on statistical calculations and probabilities, producing the impression of *learning* as humans do.\n",
            
            "- There are algorithms specifically designed for each of these calculations. These are collected together in Libraries like the ones mentioned in the **About Us** section.\n",

            "- Now, these libraries all come packed to be used along with a programming language - like *Python*. One must write down some code to get results to their ML problems.\n",

            "- We have taken care of that part for you. Just choose your data, select your parameters and type of outcome that you need and we will try and produce the results you want to see in a few clicks!\n",
        )
    st.subheader("Guidelines")
    with st.expander("Expand"):
        st.write(
            "- All ML Models need data to work on. Choose from among our sets of data, or go ahead and upload your own data in the *Machine Learning* tab, via the sidebar menu.\n",

            "- Once the data is selected, we will show you an image of how the dataset appears. You can take a good look at what it looks like then go on ahead to select the data that you feel will be most adequate for your analyses.\n",

            "- We will give you a set of options to choose from. At least 2 columns in the dataset must contain numerical data.\n",
            
            "- If, while displaying our data, you feel like a column has the wrong data type assigned to it, you can go ahead and cast the column to the desired data type from our conversion interface.\n",

            "- You can now choose the type of Machine Learning algorithm you wish to apply on the dataset.\n"

            "- Once the data seems clean and you have selected the columns you want us to act upon, click the 'Learn' button to go ahead with the training process.\n",

            "\n##### That's it! Go ahead and download the output chart as a PNG image!\n"
        )

# END OF HOME PAGE 

# MACHINE LEARNING
if page == pages[1]:
    st.title("ML On Click - Machine Learning", anchor="ML")

    # DATA SELECTION
    st.write(
        "\nUpload your data and download your desired results\n",
        
        "\n**OR**\n",

        "\nUse one of our sample datasets to explore the application and see what all we offer!\n",

        "\n----\n",

        "\nTo get started off with using the app:\n",

        "\n##### Choose Data\n"
    )
    proceed_column_choice = False
    algos = ["Simple Linear Regression", "Logistic Regression", "Polynomial Regression"]
    ds_list = {"Heart Diseases":"Heart Predictions 2.csv", "Titanic Survival":"submission.csv"}
    ds = st.selectbox("From our datasets:",ds_list)

    upload = st.sidebar.file_uploader("Upload Here:", type=['csv', 'txt'], help="You may upload a 'Comma Separated Values' file(.csv) containing your data, or a 'Text' file(.txt) in which the data is Comma-Delimited")
    
    df = pd.DataFrame()
    try:
        st.write("\nUsing the data from '{}'!\n".format(upload.name))
        df = pd.read_csv(upload)
    except:
        if upload == None:
            st.write("\nCurrently using our data\n")
            df = pd.read_csv(ds_list[ds])

    try:
        st.write(df.head(10))
    except:
        st.write("\nWe are unable to display your file.\n")

    if not df.empty:
        cols = list(df.columns)
        not_rec_cols = []  # Columns which we recommend avoiding
        nice_cols = []
        for column in cols:
            if df[column].isnull().sum()/len(df[column]) > 0.3:
                not_rec_cols.append(column)
            else: nice_cols.append(column)

        if len(not_rec_cols) > 0:
            st.write("\nWe **recommend avoiding** the use of the following columns, as they contain too many impurities:\n")
            st.write(not_rec_cols)
        if len(nice_cols) > 1:
            st.write("\nWe do **recommend using** any of these columns:\n")
            st.write(nice_cols)
            proceed_column_choice = True
        else:
            st.write("\nWe need 1 dependant and at least 1 independant variable to perform regressions. Current count 'ideal columns' does not match the requirements; however, you may go ahead with analysis if you feel like there are other columns which you wish to use.")
    else:
        st.write("\nIt seems your dataset it empty! Please check the file uploaded for errors.\n")
    
    st.write("\nYou may choose columns you wish to analyse from the sidebar!\n")
    if proceed_column_choice:
        tempcols = cols
        og_chosen_target = st.sidebar.selectbox("Target Variable", tempcols)
        st.write("\nCurrent Target:", og_chosen_target)
    if og_chosen_target:
        othercols = [i for i in tempcols if i != og_chosen_target]
        og_chosen_cols = st.sidebar.multiselect("Independent Variable(s)", othercols, default=[i for i in nice_cols if i != og_chosen_target])
    
    show_pp = st.button("Brute Force Visualisation")
    if show_pp:
        fig = sns.pairplot(df[og_chosen_cols + [og_chosen_target]])
        st.pyplot(fig)
    
    ml_type = st.selectbox("Machine Learning Algorithm", algos, help="What kind of algorithm do you feel would be best fit for your analysis?")
    bf_elim = st.sidebar.checkbox("No Backward Feature Elimination")
    
    if ml_type == algos[0]:
        st.write(algos[0])
    elif ml_type == algos[1]:
        import pandas as pd
        import numpy as np
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pickle
        
        # Cleaning Data, if possible automatically
        # Obtaining null values
        nullinfo = dict(df.isnull().sum())
        nullkeys = list(nullinfo.keys())
        nullvals = list(nullinfo.values())
        nullcount = 0
        null_rows = df.isnull().sum(axis = 1)
        max_null = nullkeys[nullvals.index(max(nullvals))]

        if max(nullvals) > 0:
            for i in null_rows:
                if i > 0:
                    nullcount += 1
            st.write("\nTotal Null Values: {}\n".format(nullcount))
            nullpct = round((nullcount/len(df.index))*100)
            st.write("\nThis accounts for {}% of the total data.\n".format(nullpct))
            if nullpct < 20:
                st.write("\nWe can do away with this data since the volume of null values is manageable.\n")
                df.dropna(axis=0, inplace=True)
            else:
                st.write("\nThe null values constitute a major portion of the data, consider choosing a different set of columns, or try cleaning the data locally\n")

        # fig = sns.pairplot(df)
        # plt.show()

        from statsmodels.tools import add_constant
        df_constant = add_constant(df)

        chosen_cols = og_chosen_cols
        chosen_target = og_chosen_target

        import scipy.stats as sct
        sct.chisqprob = lambda chisq, df: sct.chi2.sf(chisq, df)
        try:
            import statsmodels.api as sm
            model = sm.Logit(df[chosen_target], df_constant[chosen_cols])
            result = model.fit()
            st.write(result.summary())
            pvals = list(result.pvalues)
            if not bf_elim:
                if max(pvals) > 0.05:
                    st.write("\nP Values are more than optimum 0.05. We will try Backward Feature Elimination\n")

                def back_feature_elem(dataframe, target, cols):
                    while len(cols) > 0:
                        model = sm.Logit(target, dataframe[cols])
                        result = model.fit(disp=0)
                        largest_p=round(result.pvalues, 3).nlargest(1)
                        if largest_p[0] < 0.05:
                            return result
                            # break
                        else:
                            cols.remove(largest_p.index)
                result = back_feature_elem(df_constant, df[chosen_target], chosen_cols)
                st.write(result.summary())

                st.write(
                    "\nAfter testing out impacts on results, we have narrowed down the list of columns to: {}\n".format(chosen_cols),
                    "\nIf you do not wish for this to happen to your data, please choose 'No Backward Feature Elimination' on the sidebar\n")
        except:
            st.write("\nBackward Feature Elimination was not possible. Proceeding without it.\n")
        # MAKING ACTUAL MODEL NOW
        Y_data = df[chosen_target]
        X_data = df[chosen_cols]

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_data = scaler.fit_transform(X_data)

        x_train,x_test, y_train,y_test = train_test_split(X_data, Y_data, test_size=0.30, random_state=42)

        from sklearn.linear_model import LogisticRegression

        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)

        y_preds = logreg.predict(x_test)
        fig,ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_preds), annot = True, ax = ax)
        st.write(fig)
        st.write("\nAccuracy of Model thus far: {}%\n".format(round(logreg.score(x_test, y_test)*100,3)))
        # ADD AS FEATURE AT END OF MODEL TRAINING
        # test_data = pd.read_csv("filename")
        # test_features = test_data[chosen_cols]
        # ### test_target = test_data[chosen_target]  ## PROBABLY WONT EXIST IN DATASET

        # test_preds = logreg.predict(test_features)
        try:
            if st.checkbox("Make file for download"):
                pickle.dump(logreg, open('model.pkl', 'wb'))
                st.download_button("Download LogReg Model", 'model.pkl', file_name="LogReg.pkl")
        except:
            st.write("Could not store the model in file. ;(")

    elif ml_type == algos[2]:
        st.write(algos[2])
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score, mean_squared_error

        nullinfo = dict(df.isnull().sum())
        nullkeys = list(nullinfo.keys())
        nullvals = list(nullinfo.values())
        nullcount = 0
        null_rows = df.isnull().sum(axis = 1)
        max_null = nullkeys[nullvals.index(max(nullvals))]

        if max(nullvals) > 0:
            for i in null_rows:
                if i > 0:
                    nullcount += 1
            st.write("\nTotal Null Values: {}\n".format(nullcount))
            nullpct = round((nullcount/len(df.index))*100)
            st.write("\nThis accounts for {}% of the total data.\n".format(nullpct))
            if nullpct < 20:
                st.write("\nWe can do away with this data since the volume of null values is manageable.\n")
                df.dropna(axis=0, inplace=True)
            else:
                st.write("\nThe null values constitute a major portion of the data, consider choosing a different set of columns, or try cleaning the data locally\n")
        # st.download_button("Download PolyReg Model", dat)


    # END OF DATA SELECTION

# END OF MACHINE LEARNING

# ABOUT US
if page == pages[-1]:

    st.title("ML On Click - About Us", anchor="#AboutUs")
    st.write("\n\n#### Meet the App-Makers!\n")
    st.markdown("\n|Name|GitHub Profile|LinkedIn|\n|----|----|\n|Aishwarya Funaguskar|[Aish1214](https://github.com/Aish1214)|[Aishwarya Funaguskar](https://www.linkedin.com/in/aishwarya-funaguskar-b05812213/)|\n|Ishaan Sunita Pandita|[EmperorArthurIX](https://github.com/EmperorArthurIX)|[Ishaan Sunita Pandita](https://linkedin.com/in/ishaan-sunita-pandita)|\n|Rahul Pathak|[2911rahulpathak](https://github.com/2911rahulpathak)|[Rahul Pathak](https://www.linkedin.com/in/rahulgovindpathak/)|\n|Yash Shinde|[yashshinde03](https://github.com/yashshinde03)|[Yash Shinde](https://www.linkedin.com/in/yash-shinde-134560202/)|\n")

    st.write(
        "\n#### Here is the Tech Stack we used:\n",
    )
    st.markdown(
        """
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" height="12.5%" width = "12.5%">
        
        <img src = "https://upload.wikimedia.org/wikipedia/commons/4/48/Markdown-mark.svg" height="12.5%" width="20%" style="margin: 0 10%;">

        <img src="https://www.docker.com/sites/default/files/d8/styles/role_icon/public/2019-07/vertical-logo-monochromatic.png?itok=erja9lKc" height="12.5%" width="15%">
        <br><br>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
            "\n##### Python Libraries:\n\n|Library|Usage|\n|---|---|\n|[Streamlit](https://streamlit.io/)|It would not be an exaggeration to say that none of this would exist without Streamlit. The entire structure of the app can be attributed to Streamlit.|\n|[NumPy](https://numpy.org/)|We have many calculations in data analysis, and the library which helps us quickly do those using NumPy!|\n|[Pandas](https://pandas.pydata.org/)|This is the big brother of NumPy. We use Pandas to organise data into beautiful tables and also read and write these tables from and to files!|\n|[Matplotlib](https://matplotlib.org/)|This is the library that gives you quick and simple graphs!|\n|[Seaborn](https://seaborn.pydata.org/)|Big brother of Matplotlib! Gives you graphs with advanced features such as regression lines and comparison graphs!|\n|[Scikit Learn](https://scikit-learn.org/)|The Brains of this app, does a lot of the Machine Learning hardwork!|\n"
        )

# END OF ABOUT US