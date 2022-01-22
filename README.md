# ML On Click
> Made with blood, sweat and tears. Emphasis on blood.
> 
> If you try to steal it, we will find you and stop you :) So please don't. Let's be friends instead.
>
> We also have a TradeMark on the Logo.

---
## Inspiration
(PS: I == Ishaan is speaking)

I have always found *Machine Learning* very tedious to learn. I love it, but it does not change the fact that it **is** difficult. I looked up solutions online for *low-code* or *no-code* machine learning, but all I could find were these technology giants like **Google**, **Amazon** and **Microsoft** providing it on the *cloud* for a price per use basis

***OR***

Services like **SashiDo** which, very justifiably, charge us for their hard work.
**BUT**, what about a guy like me? Someone who just wants to get away with stuff like
- Linear Regression
- Logistic Regression
- K Nearest Neighbours
- Polynomial Regression

and other simple algorithms, **without** having to write a hundred lines of code every single time?

I want to create something like [ILovePDF](https://www.ilovepdf.com/)!

##### I could find no solution out there, so I thought I'll just go ahead and build it myself and save everyone around me a tonne of time, for free, unless any of you choose to pay me as an intern ;)

---
## What it does

The project is still under development, but it has the following features so far:
- Landing Page (duh), where visitors can learn about the app and how to use it
- About Us, where visitors can see who made the app.
- Machine Learning Page, where the actual magic happens.

On the Machine Learning Page, we have set up the following stuff (at the time this description was written):
- Uploading a dataset (CSV or TXT)
- Choosing one from our default datasets
- Looking at the dataset using `pandas.DataFrame`
- Recommendations on which columns seem best fit for analysis and which ones might cause issues **(Big Brains)**
- Informing the user if the dataset contains more than 20% null values *(under development)*
- Choosing one `Target` Column for the algorithm
- Choosing one or more `Independent Variable(s)` as features for the algorithm
- Automatically selects all `Recommended` columns as feature columns once `Target` is chosen. User has the choice to select these manually as well.
- Displaying a `Brute Force Visualisation` of chosen target and features as part of *Exploratory Data Analysis*
- Choosing a suitable algorithm from the list of available ones and running it on the dataset

---
## How we built it

##### Blood, Sweat and Tears!
#### Tech Stack We Used
- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit Learn](https://scikit-learn.org/stable/)
- [MatPlotLib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Statsmodels](https://www.statsmodels.org/stable/index.html)
- [SciPy](https://scipy.org/)
- [MarkDown](https://www.markdownguide.org/)

And also, Google Colab along with data from Kaggle!

---
## Challenges we faced

**A LOT OF THEM!**

Some of the challenges we faced before and/or are facing right now are:
- **Turning the Idea into a Plan**. We had an dream, but we had to lay down a series of steps to reach there as well. Brainstorming helped us come up with a few major steps to begin with
- **Unclean data**. Not every dataset that gets uploaded on the app will be ML suitable. We need to have some sort of mechanism to detect noise in the data, and if possible remove it, or at least let the user know about it. So, we are working on implementing a null-value system which checks how much useful data is actually present in the dataset
- **Categorical and Numerical Data**. Machine Learning models usually require Categorical data to be converted to features by one hot encoding, dummy variables, binary encoding and other such techniques; however, doing so **automatically** is easier said than done. We are still trying to figure out what to do with this issue.
- **Bugs**. Sometimes, while testing the app, it runs just fine, but when I come back the other day, it breaks. It's the second nature of programming, and we are tackling all this with `try` and `except` blocks.
- **Data Type Incompatibilities**. Of course, when we perform Machine Learning by hand, we are able to go through data extensively and check for incosistencies in data types like the `Exam_Marks` column being labelled as `str` instead of `float32`, or `pandas.Series` being replaced by a `numpy.ndarray` and such stuff; however, we are still trying to find a way to do this without having to manually check everything.

---
## Accomplishments that we're proud of

The work has just begun, but here's some things we've done:
- Landing Page is ready and usable
- We moved one step closer to our dream by adding **Logistic Regression** in full swing, with *output display*, *confusion matrix*, *accuracy check* and even *backward feature elimination* after a *Chi Square test*.
- Column choice is sorted well
- We also added a checkbox, where the user can choose *not* to have backward feature elimination in their LogReg model, if they feel like it is not needed.

---
## What we learned

Many, many, many new things!

**Streamlit**:
- Streamlit `header` and `subheader` text options
- Streamlit `container` and `columns` HTML layout variations
- Streamlit `multiselect` input option and its default attribute
- Streamlit `expander` list option

**Machine Learning**:
- **Backward Feature Elimination**. Understood the algorithm itself on a surface level, its impact on the model and implementation using the `statsmodels` library
- **Chi Square Test**. Read about what it actually means, found out about `p-value` and what its value means
- **Logit Model**. Learnt how to implement the model as part of the Chi Square test using the `statsmodels` library
- **Grid Search**. Learning about the concept, understanding how it is used to find an optimum target model
- **Polynomial Regression**. Reading about the model, finding best practices to find appropriate degree. Searching for methods to specifically avoid overfitting. Apparently, `Grid Search` + `Mean Squared Error` + `R Squared`, all three contribute towards determining whether overfitting has occurred or not. Still researching.
