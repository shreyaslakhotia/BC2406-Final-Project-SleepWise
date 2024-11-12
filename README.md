# SleepWise: Enhancing Sleep Health Through Lifestyle Changes

## Project Overview
SleepWise is a data-driven solution aimed at improving sleep quality and addressing sleep issues by identifying key lifestyle changes. Sleep disorders, prevalent across many populations, significantly impact health, productivity, and economic well-being. Our project seeks to develop insights and recommendations to help individuals improve sleep quality and efficiency through personalized interventions.

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Data Cleaning & Exploration](#data-cleaning-and-exploration)
3. [Model Development](#model-development)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Plan](#implementation-plan)
6. [Solution Evaluation](#solution-evaluation)

## Problem Statement
Due to hectic modern lifestyles, sleep issues and disorders are increasingly common. Many people experience insufficient sleep duration and low sleep quality, which negatively affects health and productivity. SleepWise focuses on identifying and recommending lifestyle modifications that target these issues, thereby supporting better sleep health.

## Data Cleaning and Exploration
- **Datasets Used**: We utilized two primary datasets:
  - **Sleep Health & Lifestyle Dataset**: Includes information on various health and lifestyle factors.
  - **Sleep Efficiency Dataset**: Focuses on sleep efficiency and factors affecting it.

### Data Cleaning Process
- **Standardization**: Ensured consistency by merging similar categories.
- **Outlier Removal**: Filtered to enhance the accuracy of results.
- **Feature Engineering**: Created new features, such as separating systolic and diastolic blood pressure readings.

### Data Exploration
Our exploration revealed strong correlations:
- Positive correlation between sleep duration and quality.
- Negative correlation between stress and sleep quality.
- Lifestyle factors like alcohol consumption and exercise frequency also affect sleep efficiency.

## Model Development
### Model 1: Linear Regression (Sleep Quality Prediction)
- **Features**: Stress level, sleep duration, and heart rate.
- **Performance**: Achieved low Mean Squared Error (MSE), indicating high prediction accuracy.

### Model 2: Random Forest Regression
- **Features**: Included all variables for improved accuracy.
- **Performance**: MSE of 0.063, with high R-squared value (0.95), showing robust model performance.

### Model 3: Logistic Regression (Sleep Efficiency Classification)
- **Features**: Lifestyle and demographic factors.
- **Performance**: Accuracy of 71%, indicating reliable classification of sleep efficiency levels.

### Model 4: Decision Tree Classifier
- **Features**: Alcohol consumption, exercise frequency, and awakenings.
- **Performance**: Accuracy of 82.35%, with insights that informed personalized lifestyle recommendations.

## Proposed Solution
Our solution, **SleepWise**, offers a mobile application that:
- Provides personalized sleep recommendations.
- Tracks sleep cycles, capturing light, deep, and REM sleep stages.
- Delivers insights into lifestyle changes that may enhance sleep quality.

### Key Features
- **Personalized Tracking**: Tracks sleep duration, quality, and contributing factors (e.g., caffeine intake).
- **Progress Tracking and Analytics**: Offers weekly and monthly reports to monitor sleep improvements.
- **Recommendations**: Suggests lifestyle adjustments based on individual sleep data and trends.

## Implementation Plan
1. **Year 1**: Develop and pilot the core SleepWise app.
2. **Year 2**: Integrate with wearables and enhance reporting features.
3. **Year 3**: Personalize insights and add gamification.
4. **Year 4**: Launch educational campaigns and corporate partnerships.
5. **Year 5**: Evaluate solution impact and explore new feature expansions.

## Solution Evaluation
### Success Metrics
- **User Retention and Engagement**: Measuring interaction with app features and reports.
- **Health Improvement Indicators**: Track sleep quality, reduction in alcohol intake, and exercise frequency.

## Authors
- **Team Members**: Cheng Kai Wen, Trinyce; Chua Jia Han; Guek Shu Yuan; Lakhotia Shreyas; Lee Chuan Jue; Seah Shao Hong, Eason

