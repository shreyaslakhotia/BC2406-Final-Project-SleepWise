library(data.table)
library(ggplot2)
library(dplyr)
library(caret)    
library(rpart)
library(rpart.plot)
library(stringr)         
library(randomForest) 
library(e1071)            
library(glmnet)
library(reshape2)

###3 data cleaning and exploration( Shu Yuan)
# Load the dataset
data <- read.csv("C:\\Users\\tp\\Downloads\\NTU\\Y2S1\\BC2406 Analytics I\\BC2406 Course Materials\\group project\\sleepandlifestyle.csv")

# Check for blank values (empty strings) in each column
data[data == ""] <- NA

# Check for missing values (NA) in each column
missing_values <- colSums(is.na(data))
print("Missing values per column:")
print(missing_values)

# Check for duplicate rows
duplicates <- duplicated(data)
print("Number of duplicate rows:")
print(sum(duplicates))

# Display missing values and duplicates in a table format
summary_table <- data.frame(
  Column = names(missing_values),
  Missing_Values = missing_values
)

# Add a row for the total number of duplicate rows
summary_table <- rbind(summary_table, data.frame(Column = "Total Duplicate Rows", Missing_Values = sum(duplicates)))

# Display the summary table
print(summary_table)

# Display unique BMI Categories before cleaning
unique_bmi_categories <- unique(data$BMI.Category)
print(unique_bmi_categories)

# Clean the data by combining "Normal" and "Normal Weight" into "Normal"
data <- data %>%
  mutate(BMI.Category = case_when(
    BMI.Category %in% c("Normal", "Normal Weight") ~ "Normal",
    TRUE ~ BMI.Category
  ))
cleaned_bmi_categories <- unique(data$BMI.Category)
print("Unique BMI Categories after cleaning:")
print(cleaned_bmi_categories)

# Function to remove outliers using IQR for all numeric columns
remove_outliers <- function(x, column_name) {
  if (is.numeric(x)) {
    # Apply IQR method for numeric columns
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    # Replace values outside the range with NA
    x[x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR)] <- NA
  }
  return(x)
}

# Filter outliers
columns_of_interest <- c("Quality.of.Sleep", "Stress.Level")
data[columns_of_interest] <- lapply(data[columns_of_interest], remove_outliers)
out_of_range <- sapply(data[columns_of_interest], function(x) any(x < 1 | x > 10, na.rm = TRUE))

# Display the result
if (any(out_of_range)) {
  cat("Outliers detected outside the range 1-10 in the following columns:\n")
  print(names(data[columns_of_interest])[out_of_range])
} else {
  cat("No outliers detected outside the range 1-10 in 'Quality.of.Sleep' and 'Stress.Level'.\n")
}

# Display summary of these columns to confirm data falls within the range
summary(data[columns_of_interest])

# Apply the function to each column in the dataset, using column names
data_clean <- as.data.frame(mapply(remove_outliers, data, names(data), SIMPLIFY = FALSE))

# Create a boxplot for Sleep Quality by Sleep Disorder
ggplot(data, aes(x = Sleep.Disorder, y = Quality.of.Sleep, fill = Sleep.Disorder)) +
  geom_boxplot() +
  labs(title = "Distribution of Quality of Sleep by Sleep Disorder",
       x = "Sleep Disorder",
       y = "Quality of Sleep") +
  theme_minimal() +
  scale_fill_manual(values = c("Insomnia" = "pink", "None" = "lightgreen", "Sleep Apnea" = "lightblue"))

# Create a boxplot for Sleep Duration by Sleep Disorder
ggplot(data, aes(x = Sleep.Disorder, y = Sleep.Duration, fill = Sleep.Disorder)) +
  geom_boxplot() +
  labs(title = "Distribution of Sleep Duration by Sleep Disorder",
       x = "Sleep Disorder",
       y = "Sleep Duration (hours)") +  # Assuming duration is in hours
  theme_minimal() +
  scale_fill_manual(values = c("Insomnia" = "pink", "None" = "lightgreen", "Sleep Apnea" = "lightblue"))

# Calculate the correlation matrix
cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")

# Melt the correlation matrix for ggplot
cor_melted <- as.data.frame(as.table(cor_matrix))

# Create a heatmap to visualize the correlations
ggplot(cor_melted, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), name = "Correlation") +
  labs(title = "Correlation Heatmap of Sleep and Lifestyle Variables",
       x = "Variables",
       y = "Variables") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = round(Freq, 2)), color = "black")  # Add correlation coefficients

# Scatter Plot of Stress Level and Sleep Quality
correlation <- cor(data$Stress.Level, data$Quality.of.Sleep, use = "complete.obs")
print(paste("Correlation between Stress Level and Quality of Sleep:", round(correlation, 2)))
ggplot(data, aes(x = Stress.Level, y = Quality.of.Sleep)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Scatter Plot of Stress Level vs. Quality of Sleep",
       x = "Stress Level",
       y = "Quality of Sleep") +
  theme_minimal()


# Create a box plot to visualize Quality of Sleep by BMI Category
data$BMI.Category <- factor(data$BMI.Category, levels = c("Normal", "Overweight", "Obese", "Underweight"))
ggplot(data, aes(x = BMI.Category, y = Quality.of.Sleep, fill = BMI.Category)) +
  geom_boxplot() +
  labs(title = "Quality of Sleep by BMI Category",
       x = "BMI Category",
       y = "Quality of Sleep") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +  # Optional: use a color palette for aesthetics
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

### Ji Han
data <- fread("Sleep_Efficiency.csv")

# Select only numerical columns for correlation
numeric_data <- data %>% select(Age, `Sleep duration`, `Sleep efficiency`, 
                                `REM sleep percentage`, `Deep sleep percentage`, 
                                `Light sleep percentage`, Awakenings, 
                                `Caffeine consumption`, `Alcohol consumption`, 
                                `Exercise frequency`)

cor_matrix <- cor(numeric_data, use = "complete.obs")

melted_cor_matrix <- melt(cor_matrix)

#Correlation Heatmap
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) + # Add correlation values
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  labs(title = "Correlation Heatmap", x = "", y = "")
#Alcohol Consumption
ggplot(data, aes(x = `Alcohol consumption`, y = `Sleep efficiency`)) +
  geom_point(color = "blue", alpha = 0.6) +  # Scatter plot points
  geom_smooth(method = "lm", color = "red", se = FALSE) +  # Trend line (linear model)
  labs(title = "Alcohol Consumption vs Sleep Efficiency",
       x = "Alcohol Consumption",
       y = "Sleep Efficiency") +
  theme_minimal()
#Exercise Frequncy
ggplot(data, aes(x = `Exercise frequency`, y = `Sleep efficiency`)) +
  geom_point(color = "green", alpha = 0.6) +  # Scatter plot points
  geom_smooth(method = "lm", color = "blue", se = FALSE) +  # Trend line (linear model)
  labs(title = "Exercise Frequency vs Sleep Efficiency",
       x = "Exercise Frequency",
       y = "Sleep Efficiency") +
  theme_minimal()

###4: Model analysis of sleep health and lifestyle data(linear reg and Random forest)
### Chuan Jue

setwd("C:/Users/leech/Desktop/BC proj")

sleep_data <- fread("Sleep_health_and_lifestyle_dataset.csv")
work_data <- fread("Impact_of_Remote_Work_on_Mental_Health.csv")

# Replace spaces with underscores in column names
names(sleep_data) <- gsub(" ", "_", names(sleep_data))

# Create two new columns and extract systolic and diastolic values
sleep_data$Systolic_Blood_Pressure <- as.numeric(str_extract(sleep_data$Blood_Pressure, "^\\d+"))
sleep_data$Diastolic_Blood_Pressure <- as.numeric(str_extract(sleep_data$Blood_Pressure, "\\d+$"))

sleep_data$Blood_Pressure <- NULL
sleep_data$Person_ID <- NULL
sleep_data$Sleep_Disorder <- NULL

categorical <- c("Gender", "Occupation", "BMI_Category", "Sleep_Disorder")

# Change objects into categorical numbers starting from 1
sleep_data[, (categorical) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = categorical]


# Convert specified columns to numeric
sleep_data$Quality_of_Sleep <- as.numeric(sleep_data$Quality_of_Sleep)
sleep_data$Heart_Rate <- as.numeric(sleep_data$Heart_Rate)
sleep_data$Stress_Level <- as.numeric(sleep_data$Stress_Level)
sleep_data$Sleep_Duration <- as.numeric(sleep_data$Sleep_Duration)

#multi-variate linear regression model
str(sleep_data)

# Set seed for reproducibility
set.seed(3)

# Split data into training (70%) and testing (30%) sets
sample_index <- sample(1:nrow(sleep_data), 0.7 * nrow(sleep_data))
train_data <- sleep_data[sample_index, ]
test_data <- sleep_data[-sample_index, ]

# Fit the linear regression model using the training data
sleep_model <- lm(Quality_of_Sleep ~ Heart_Rate + Stress_Level + Sleep_Duration, data = train_data)

# Print the summary of the model
summary(sleep_model)

# Display model summary
summary(sleep_model)

# Predict Quality of Sleep for the training data
train_predictions <- predict(sleep_model, newdata = train_data)

# Display the first few predictions
head(train_predictions)

# Calculate Mean Squared Error (MSE) for the training set
train_mse <- mean((train_data$`Quality_of_Sleep` - train_predictions)^2)
print(paste("Training Mean Squared Error:", train_mse))

# Predict Quality of Sleep for the test data
test_predictions <- predict(sleep_model, newdata = test_data)

# Check the predictions
head(test_predictions)

# Calculate Mean Squared Error (MSE) for the test set
test_mse <- mean((test_data$Quality_of_Sleep - test_predictions)^2, na.rm = TRUE)
print(paste("Test Mean Squared Error:", test_mse))

# Create a scatter plot of actual vs. predicted values
ggplot(test_data, aes(x = Quality_of_Sleep, y = test_predictions)) +
  geom_point(color = 'blue') +
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = 'dashed') +
  labs(title = "Actual vs. Predicted Quality of Sleep (Test Set)",
       x = "Actual Quality of Sleep",
       y = "Predicted Quality of Sleep") +
  theme_minimal()

###random forest

# Prepare data
set.seed(123)  # For reproducibility
train_index <- createDataPartition(sleep_data$Quality_of_Sleep, p = 0.7, list = FALSE)
train_data1 <- sleep_data[train_index, ]
test_data1 <- sleep_data[-train_index, ]

# Train Random Forest model with ntree set to 20
rf_model <- randomForest(Quality_of_Sleep ~ ., data = train_data1, ntree = 20)

# Make predictions
predictions <- predict(rf_model, newdata = test_data1)

# Evaluate model
r2 <- R2(predictions, test_data1$Quality_of_Sleep)
mae <- mean(abs(predictions - test_data1$Quality_of_Sleep))
mse <- mean((predictions - test_data1$Quality_of_Sleep)^2)

# Print evaluation metrics
cat("R-squared:", r2, "\n")
cat("Mean Absolute Error:", mae, "\n")
cat("Mean Squared Error:", mse, "\n")

metrics <- data.frame(
  Metric = c("R-squared", "Mean Absolute Error", "Mean Squared Error"),
  Value = c(r2, mae, mse)
)

# Plot the metrics in a bar chart
ggplot(metrics, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3)), vjust = -0.5, size = 5) + # Adding values above bars
  theme_minimal() +
  labs(title = "Model Evaluation Metrics", 
       x = "Metric", 
       y = "Value") +
  theme(legend.position = "none")

importance_values <- importance(rf_model)

# Create a data frame for feature importance
feature_importance_df <- data.frame(
  Feature = rownames(importance_values),
  Importance = importance_values[, 1] # Use the first column for Mean Decrease in Accuracy
)

# Calculate the total importance
total_importance <- sum(feature_importance_df$Importance)

# Calculate contribution in percentage
feature_importance_df$Contribution_Percentage <- (feature_importance_df$Importance / total_importance) * 100

# Sort features by contribution percentage
feature_importance_df <- feature_importance_df[order(-feature_importance_df$Contribution_Percentage), ]

# Print the contribution percentages
print(feature_importance_df)


# Assuming the feature_importance_df data frame is already created
# with a Contribution_Percentage column

# Plot feature contribution percentages
ggplot(feature_importance_df, aes(x = reorder(Feature, Contribution_Percentage), y = Contribution_Percentage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # Flip coordinates for better visibility
  geom_text(aes(label = round(Contribution_Percentage, 2)), vjust = 0.5, hjust = -0.1, size = 4) +  # Adding values next to bars
  theme_minimal() +
  labs(title = "Feature Contribution to Quality of Sleep Prediction", 
       x = "Features", 
       y = "Contribution Percentage (%)") +
  theme(axis.text.y = element_text(size = 10))  # Adjust y-axis text size for better readability


