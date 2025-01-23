import itertools
import pandas as pd
import numpy as np
import math
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Sampling

population = np.random.randint(0, 80, 10000)
print(population.mean())
# Sampling says; instead of surveying these 10k people, select a smaller population that can represent these 10k people.
# It helps us generalize with fewer data.
np.random.seed(115)

sample = np.random.choice(a=population, size=100)  # We selected 100 people from the population
print(sample.mean())
np.random.seed(10)
sample1 = np.random.choice(a=population, size=100)
sample2 = np.random.choice(a=population, size=100)
sample3 = np.random.choice(a=population, size=100)
sample4 = np.random.choice(a=population, size=100)
sample5 = np.random.choice(a=population, size=100)
sample6 = np.random.choice(a=population, size=100)
sample7 = np.random.choice(a=population, size=100)
sample8 = np.random.choice(a=population, size=100)
sample9 = np.random.choice(a=population, size=100)
sample10 = np.random.choice(a=population, size=100)

print((sample1.mean() + sample2.mean() + sample3.mean() + sample4.mean() + sample5.mean() + sample6.mean() + sample7.mean() + sample8.mean() + sample9.mean() + sample10.mean()) / 10)
# As the number of samples increases, we converge to the overall mean.

# Descriptive Statistics - Exploratory Data Analysis
print("EXPLORATORY DATA ANALYSIS")
df = sns.load_dataset("tips")
print(df.describe().T)


"""
# Confidence Intervals
# What are the confidence intervals in the tips dataset?

# Let's calculate the worst and best scenarios.

print(sms.DescrStatsW(df["total_bill"]).tconfint_mean())  # Determined the worst and best two scenarios.
print(sms.DescrStatsW(df["tip"]).tconfint_mean())  # The range of tips.

print("Titanic Age")

df = sns.load_dataset("titanic")
print(df.describe().T)
print(sms.DescrStatsW(df["age"].dropna()).tconfint_mean())
"""

# Correlation
df = sns.load_dataset("tips")
print(df.head())
print("Is there a correlation between the bill and the tip?")

df["total_bill"] = df['total_bill'] - df["tip"]
"""
df.plot.scatter("tip", "total_bill")
plt.show()  # We can see a moderate positive correlation between these two. Now, let's turn it into mathematics.
"""
print(df["tip"].corr(df["total_bill"]))  # There is a moderate positive correlation (0.5766) between them.

# Hypothesis Testing

print("HYPOTHESIS TESTING")
# The main goal is to show whether the observed differences are due to chance.

# For example, did the interest increase after the interface change?

print("Independent Two-Sample T Test.")  # Two proportion tests. A and B. Control and Experimental groups.

######################################################
# AB Testing (Independent Two-Sample T Test)
######################################################

# 1. Formulate Hypotheses
# 2. Check Assumptions
#   - 1. Normality Assumption
#   - 2. Homogeneity of Variances
# 3. Apply the Hypothesis
#   - 1. If assumptions are met, use independent two-sample t-test (parametric test)
#   - 2. If assumptions are not met, use Mann-Whitney U test (non-parametric test)
# 4. Interpret results based on p-value
# Notes:
# - If normality is not met, use step 2. If homogeneity of variances is not met, input an argument for step 1.
# - It is helpful to examine and correct outliers before checking normality.


############################
# Application 1: Are there statistically significant differences between smokers and non-smokers' bill averages?
############################

df = sns.load_dataset("tips")
print(df.head())

print(df.groupby("smoker").agg({"total_bill": "mean"}))  # There is a difference here, but is it real?

# 1. Formulate Hypotheses:
# H0: m1 = m2
# H1: m1 != m2

# 2. Check Assumptions

# Normality assumption
# Homogeneity of variances

# H0: The normality assumption holds.
# H1: ... does not hold.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat = %.4f, p-value= %.4f" % (test_stat, pvalue))

# If p-value < 0.05, H0 is rejected.
# If p-value is not less than 0.05, H0 cannot be rejected.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value= %.4f" % (test_stat, pvalue))

# H0 is rejected. The p-value is 0.002, which is less than 0.05.

# Homogeneity of Variances
# H0: Variances are homogeneous
# H1: Variances are not homogeneous.

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print("H0 is rejected. Variances are not homogeneous.")
print("Test Stat = %.4f, p-value= %.4f" % (test_stat, pvalue))

# If p-value < 0.05, H0 is rejected.
# If p-value is not less than 0.05, H0 cannot be rejected.

# 3 and 4. Apply Hypothesis

# 1. If assumptions are met, use independent two-sample t-test (parametric test)
# 2. If assumptions are not met, use Mann-Whitney U test (non-parametric test)

############################
# 1.1 If assumptions are met, use independent two-sample t-test (parametric test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
# If homogeneity of variances is not met, use "False" instead of "True"

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# The p-value is 0.18, which is greater than 0.05. Therefore, H0 cannot be rejected. There is no statistically significant difference between smokers and non-smokers.

############################
# 1.2 If assumptions are not met, use Mann-Whitney U test (non-parametric test)
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 1.2 TEST IS SUITABLE FOR US. BECAUSE ASSUMPTIONS ARE NOT MET.

# Application Titanic: Is there a statistically significant difference in the average age of female and male passengers?
############################

df = sns.load_dataset("titanic")
print(df.head())
print(df.groupby("sex").agg({"age": "mean"}))

# 1. Formulate Hypotheses:
# H0: M1 = M2 (There is no statistically significant difference in the average age of female and male passengers)
# H1: M1 != M2 (There is a significant difference)

# 2. Check Assumptions
# Normality assumption
# H0: The normality assumption holds.
# H1: The normality assumption does not hold.

# Shapiro-Wilk Test for Normality (Age comparison between female and male)
test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Null Hypothesis is rejected
test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Null Hypothesis is rejected

# Homogeneity of Variance Test
# Null Hypothesis: Variances are homogeneous
# Alternative Hypothesis: Variances are not homogeneous
print("levene")
test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Since assumptions are not met, we proceed with a non-parametric test

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# As a result, the null hypothesis is rejected. There is a significant difference.

############################
# Application 3: Is there a statistically significant difference in age between people with and without diabetes?
############################

df = pd.read_csv("diabetes.csv")
print(df.head())

print(df.groupby("Outcome").agg({"Age": "mean"}))
# There seems to be a difference, but let's check.

# 1. Form the hypotheses
# Null Hypothesis: M1 = M2
# There is no statistically significant difference in the average age between diabetic and non-diabetic individuals
# Alternative Hypothesis: M1 != M2
# There is a difference.

# 2. Check assumptions

# Normality assumption (Null Hypothesis: Data follows a normal distribution)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Null Hypothesis rejected
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Null Hypothesis rejected

##### Since normality assumption is not met, we directly move to a non-parametric test.

# Since normality assumption is not met, we proceed with a non-parametric test.
print("Mannwhitneyu test")
# Hypothesis (Null Hypothesis: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
print("Null hypothesis is rejected, there is a significant difference.")

###################################################
# Business Problem: Is there a significant difference in ratings between people who watched the majority of the course and those who didn't?
###################################################

# Null Hypothesis: M1 = M2 (No significant difference between the two groups' averages)
# Alternative Hypothesis: M1 != M2 (There is a difference)

df = pd.read_csv("course_reviews.csv")
print(df.head())

print(df[(df["Progress"] > 75)]["Rating"].mean())

print(df[(df["Progress"] < 25)]["Rating"].mean())

print("Checking normality")
test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
print("Null Hypothesis rejected")

test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
print("Null Hypothesis rejected")


print("Non-parametric test")
test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
print("Null hypothesis is rejected, there is a difference between the groups.")



 ######################################################
# AB Testing (Two-Sample Proportion Test)
######################################################

# Null Hypothesis: p1 = p2
# There is no statistically significant difference between the conversion rates of the new and old designs.
# Alternative Hypothesis: p1 != p2
# There is a difference.

success_count = np.array([300, 250])
observation_count = np.array([1000, 1100])

print(proportions_ztest(count=success_count, nobs= observation_count))
# p-value is less than 0.05, so we reject the null hypothesis.

print(success_count / observation_count)

############################
# Application: Is there a significant difference in survival rates between women and men?
############################

# Null Hypothesis: p1 = p2
# There is no statistically significant difference in survival rates between women and men.

# Alternative Hypothesis: p1 != p2
# There is a difference.

df = sns.load_dataset("titanic")
print(df.head())
print(df.loc[df["sex"]== "female","survived"].mean())
print(df.loc[df["sex"]== "male","survived"].mean())


female_success_count = df.loc[df["sex"] == "female", "survived"].sum()
male_success_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_success_count, male_success_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
print("Null hypothesis is rejected, there is a significant difference.")

print("######################### ANOVA #################################")
######################################################
# ANOVA (Analysis of Variance)
######################################################

# Used to compare the means of more than two groups.

df = sns.load_dataset("tips")
print(df.head())
print(df.groupby("day")["total_bill"].mean())

# 1. Form the hypotheses

# Null Hypothesis: m1 = m2 = m3 = m4
# There is no difference in the means of the groups.

# Alternative Hypothesis: There is a difference.

# 2. Check assumptions

# Normality assumption
# Homogeneity of variances assumption

# If assumptions are met, perform one-way ANOVA
# If assumptions are not met, perform Kruskal-Wallis test

# Null Hypothesis: Data follows a normal distribution.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, "p-value: %.4f" % pvalue)

# All values are small, so the Null Hypothesis is rejected.

# Null Hypothesis: Homogeneity of variances assumption holds.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 3. Hypothesis testing and p-value interpretation

# None of the assumptions are met.
print(df.groupby("day").agg({"total_bill": ["mean", "median"]}))


# Null Hypothesis: There is no significant difference between the group means.

# Parametric ANOVA test:
print(f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"]))
# Null Hypothesis rejected

# Non-parametric ANOVA test:
print(kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"]))

# Null Hypothesis rejected

# You can experiment with the threshold value. The threshold is the confidence level; the current system works with a 95% confidence level.
print("A difference is evident, but let's see which group causes the difference.")
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
