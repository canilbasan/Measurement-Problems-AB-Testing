Statistical Analysis Project
Description
This project demonstrates various statistical analyses and hypothesis testing methods in Python using real-world datasets. 
The analyses include parametric and non-parametric tests, such as Shapiro-Wilk test for normality, Levene's test for homogeneity of variance, Mann-Whitney U test, z-test for proportions, 
ANOVA, and Kruskal-Wallis test. The code uses several datasets, including diabetes, course reviews, Titanic survival data, and the "tips" dataset from Seaborn.

Requirements
This project requires the following Python libraries:

pandas
numpy
scipy
statsmodels
seaborn

You can install the necessary libraries using pip:

pip install pandas numpy scipy statsmodels seaborn

Datasets
diabetes.csv - A dataset containing information about diabetes patients and their outcomes.
course_reviews.csv - A dataset containing course reviews with progress and rating information.
titanic - A dataset from Seaborn containing information about Titanic passengers and their survival status.
tips - A dataset from Seaborn containing restaurant tips information.

Key Analysis and Hypotheses

1. Age Difference Between Genders
Hypothesis 1: The average age of males and females is the same.
H0: Males and females have the same average age.
H1: Males and females have different average ages.
The code uses Shapiro-Wilk test for normality, Levene's test for homogeneity of variance, and Mann-Whitney U test (non-parametric) to check the hypothesis.

2. Age Difference Between Diabetic and Non-Diabetic Patients
Hypothesis 2: There is no significant difference in the average age of diabetic and non-diabetic patients.
H0: The average age of diabetic and non-diabetic patients is the same.
H1: The average age of diabetic and non-diabetic patients is different.
The Shapiro-Wilk test is used to test normality, and Mann-Whitney U test (non-parametric) is used to compare the groups.

3. Course Rating Difference Based on Progress
Hypothesis 3: There is no significant difference in the ratings of students with high and low course progress.

H0: Students with high and low progress have the same course rating.
H1: Students with high and low progress have different course ratings.
The normality of ratings is tested using the Shapiro-Wilk test, and the Mann-Whitney U test (non-parametric) is used to compare ratings.

4. AB Testing (Two Proportion Z-Test)
Hypothesis 4: There is no significant difference in the conversion rates between the new and old design.

H0: Conversion rates of new and old designs are equal.
H1: Conversion rates of new and old designs are different.
The code calculates the z-test for proportions to compare the conversion rates.

5. Survival Rate Between Genders on the Titanic
Hypothesis 5: There is no significant difference in survival rates between men and women on the Titanic.

H0: Survival rates of men and women are the same.
H1: Survival rates of men and women are different.
The proportions z-test is used to compare survival rates between genders.

6. ANOVA (Analysis of Variance)
Hypothesis 6: There is no significant difference in total bill amounts across different days.

H0: The average total bill amounts are the same across days.
H1: The average total bill amounts are different across days.
The normality and homogeneity of variance assumptions are checked using the Shapiro-Wilk test and Levene’s test. If the assumptions are violated, the Kruskal-Wallis test is used as a non-parametric alternative. Tukey's HSD test is used for pairwise comparisons.

Usage
Run Statistical Tests:

You can execute the script to perform various hypothesis tests based on the datasets and hypotheses outlined above.
The code performs tests on multiple datasets and reports the results of the hypothesis testing, such as p-values and test statistics.
Interpret Results:

For each test, the null hypothesis (H0) is either accepted or rejected based on the p-value:
If p-value < 0.05, reject H0 (evidence of a significant difference).
If p-value ≥ 0.05, fail to reject H0 (no significant difference).
