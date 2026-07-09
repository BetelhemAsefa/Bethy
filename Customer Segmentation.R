#load libraries
library(tidyverse)

#import dataset
mall_data <- read.csv("Mall_Customers.csv", stringsAsFactors = FALSE)

#inspect Data
head(mall_data)
str(mall_data)
summary(mall_data)

#check missing values
colSums(is.na(mall_data))

#convert variables
mall_data$Gender <- as.factor(mall_data$Gender)

#visualize distributions

#age distribution
ggplot(mall_data, aes(x = Age)) +
  geom_histogram(bins = 20, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Age")

#income distribution
ggplot(mall_data, aes(x = Annual.Income..k..)) +
  geom_histogram(bins = 20, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Annual Income")

#spending score distribution
ggplot(mall_data, aes(x = Spending.Score..1.100.)) +
  geom_histogram(bins = 20, fill = "blue", color = "black") +
  labs(title = "Distribution of Spending Score")

#Scatter Plot 
ggplot(mall_data, aes(x = Annual.Income..k.., y = Spending.Score..1.100.)) +
  geom_point() +
  labs(title = "Income vs Spending Score")

#boxplots (Check Outliers)
ggplot(mall_data, aes(y = Annual.Income..k..)) +
  geom_boxplot() +
  labs(title = "Income Boxplot")

ggplot(mall_data, aes(y = Spending.Score..1.100.)) +
  geom_boxplot() +
  labs(title = "Spending Score Boxplot")

#select variables for clustering
cluster_data <- mall_data %>%
  select(Age, Annual.Income..k.., Spending.Score..1.100.)

# =========================
# Task 2: Normalize Data
cluster_scaled <- scale(cluster_data)

# =========================
# Task 3: Elbow Method
# =========================
set.seed(123)

wss <- numeric(10)

for (i in 1:10) {
  km <- kmeans(cluster_scaled, centers = i, nstart = 25)
  wss[i] <- km$tot.withinss
}

plot(1:10, wss, type = "b",
     xlab = "Number of Clusters (k)",
     ylab = "Within-Cluster Sum of Squares",
     main = "Elbow Method")

# =========================
# Task 4: Silhouette Score
# =========================
library(cluster)

sil_width <- numeric(10)

for (k in 2:10) {
  km <- kmeans(cluster_scaled, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(cluster_scaled))
  sil_width[k] <- mean(ss[, 3])
}

plot(2:10, sil_width[2:10], type = "b",
     xlab = "Number of Clusters (k)",
     ylab = "Average Silhouette Width",
     main = "Silhouette Method")

# =========================
# Task 5: Final Model (k = 5)
# =========================
set.seed(123)

kmeans_final <- kmeans(cluster_scaled, centers = 5, nstart = 25)

# Add cluster labels to dataset
mall_data$Cluster <- as.factor(kmeans_final$cluster)

# =========================
# Task 6: Cluster Visualization
# =========================
ggplot(mall_data,
       aes(x = Annual.Income..k..,
           y = Spending.Score..1.100.,
           color = Cluster)) +
  geom_point(size = 3) +
  labs(title = "Customer Segments (k-means)",
       x = "Annual Income",
       y = "Spending Score")

# =========================
# Task 7: Cluster Summary
# =========================
mall_data %>%
  group_by(Cluster) %>%
  summarise(
    Avg_Age = mean(Age),
    Avg_Income = mean(Annual.Income..k..),
    Avg_Spending = mean(Spending.Score..1.100.),
    Count = n()
  )

# =========================
# Task 8: Boxplots by Cluster
# =========================

ggplot(mall_data, aes(x = Cluster, y = Annual.Income..k.., fill = Cluster)) +
  geom_boxplot() +
  labs(title = "Income Distribution by Cluster")

ggplot(mall_data, aes(x = Cluster, y = Spending.Score..1.100., fill = Cluster)) +
  geom_boxplot() +
  labs(title = "Spending Score by Cluster")

