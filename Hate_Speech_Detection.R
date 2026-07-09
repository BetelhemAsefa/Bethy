library(pacman)

#use pacman to load packages
p_load(tidyverse, tidytext, janitor, stringr, ggplot2, caret, e1071)

#import the dataset
hate_data <- read.csv("HateSpeechDataset.csv", stringsAsFactors = FALSE)

#clean column names
hate_data <- clean_names(hate_data)

#check column names
names(hate_data)

#check the structure of the dataset
str(hate_data)

#display the first and last 5 rows
head(hate_data, 5)
tail(hate_data, 5)

#number of rows and columns
nrow(hate_data)
ncol(hate_data)

#number of numeric variables
sum(sapply(hate_data, is.numeric))

#number of character variables
sum(sapply(hate_data, is.character))

#check variable classes
unique(sapply(hate_data, class))

#select only the variables needed for the project
hate_data1 <- hate_data %>%
  select(content, label)

#remove missing values
hate_data1 <- hate_data1 %>%
  filter(!is.na(content), !is.na(label))

#keep only valid labels
hate_data1 <- hate_data1 %>%
  filter(label %in% c("0", "1"))

#convert label to factor
hate_data1$label <- as.factor(hate_data1$label)

#check for missing values again
colSums(is.na(hate_data1))

#view cleaned dataframe
#View(hate_data1)

#create new variables for basic text features
hate_data1 <- hate_data1 %>%
  mutate(
    char_count = nchar(content),
    word_count = str_count(content, "\\S+"),
    exclamation_count = str_count(content, "!"),
    question_count = str_count(content, "\\?"),
    hashtag_count = str_count(content, "#"),
    mention_count = str_count(content, "@")
  )

#view the updated dataframe
#View(hate_data1)

#check the distribution of labels
table(hate_data1$label)

#plot label distribution
ggplot(hate_data1, aes(x = label)) +
  geom_bar() +
  labs(title = "Distribution of Hate Speech Labels",
       x = "Label",
       y = "Count")

#visualize word count distribution
ggplot(hate_data1, aes(x = word_count)) +
  geom_histogram(bins = 30) +
  labs(title = "Distribution of Word Count",
       x = "Number of Words",
       y = "Frequency")

#compare word count by label
ggplot(hate_data1, aes(x = label, y = word_count)) +
  geom_boxplot() +
  labs(title = "Word Count by Label",
       x = "Label",
       y = "Word Count")

#load stop words
data("stop_words")

#create custom stop words list
custom_stop_words <- data.frame(
  word = c(stop_words$word, "rt", "lol", "amp"),
  lexicon = "custom"
)

#tokenize the text into words
hate_words <- hate_data1 %>%
  unnest_tokens(word, content)

#display first rows
head(hate_words, 10)

#remove stop words
hate_words_clean <- hate_words %>%
  anti_join(custom_stop_words, by = "word")

#sentiment analysis
library(tidytext)

hate_words_sent <- hate_data1 %>%
  unnest_tokens(word, content)

sentiment_data <- hate_words_sent %>%
  inner_join(get_sentiments("bing"), by = "word")

sentiment_summary <- sentiment_data %>%
  count(label, sentiment)

#plot
ggplot(sentiment_summary, aes(x = sentiment, y = n, fill = label)) +
  geom_col(position = "dodge") +
  labs(title = "Sentiment by Label",
       x = "Sentiment",
       y = "Count")

#topic modeling (LDA)

library(topicmodels)

#make sure row_id exists
hate_data1 <- hate_data1 %>%
  mutate(row_id = row_number())

hate_topic_words <- hate_data1 %>%
  unnest_tokens(word, content) %>%
  anti_join(stop_words, by = "word")

hate_dtm <- hate_topic_words %>%
  count(row_id, word) %>%
  cast_dtm(row_id, word, n)

#run LDA
lda_model <- LDA(hate_dtm, k = 3)

topic_terms <- tidy(lda_model, matrix = "beta")

top_terms <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10)

#plot topics
top_terms %>%
  ggplot(aes(x = reorder_within(term, beta, topic),
             y = beta,
             fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~topic, scales = "free") +
  coord_flip() +
  labs(title = "Top Words per Topic")

#display first rows
head(hate_words_clean, 10)

#count most common words overall
word_freq <- hate_words_clean %>%
  count(word, sort = TRUE)

#display top 20 words
head(word_freq, 20)

#plot top 15 words
word_freq %>%
  slice_max(order_by = n, n = 15) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 15 Most Frequent Words",
       x = "Word",
       y = "Frequency")

#count most common words by label
word_label_freq <- hate_data1 %>%
  unnest_tokens(word, content) %>%
  anti_join(custom_stop_words, by = "word") %>%
  count(label, word, sort = TRUE)

#display top words by label
word_label_freq %>%
  group_by(label) %>%
  slice_max(order_by = n, n = 10)

#plot top words by label
word_label_freq %>%
  group_by(label) %>%
  slice_max(order_by = n, n = 10) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(word, n), y = n, fill = label)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~label, scales = "free") +
  coord_flip() +
  labs(title = "Top Words by Label",
       x = "Word",
       y = "Frequency")

#create bigrams
hate_bigrams <- hate_data1 %>%
  unnest_tokens(bigram, content, token = "ngrams", n = 2)

#display first rows
head(hate_bigrams, 10)

#separate the bigrams into two words
hate_bigrams_sep <- hate_bigrams %>%
  separate(bigram, into = c("word1", "word2"), sep = " ")

#remove stop words from both positions
hate_bigrams_clean <- hate_bigrams_sep %>%
  filter(!word1 %in% custom_stop_words$word,
         !word2 %in% custom_stop_words$word)

#create bigram column again
hate_bigrams_clean <- hate_bigrams_clean %>%
  unite(bigram, word1, word2, sep = " ")

#count bigram frequency
bigram_freq <- hate_bigrams_clean %>%
  count(bigram, sort = TRUE)

#display top 15 bigrams
head(bigram_freq, 15)

#plot top 15 bigrams
bigram_freq %>%
  slice_max(order_by = n, n = 15) %>%
  ggplot(aes(x = reorder(bigram, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 15 Most Frequent Bigrams",
       x = "Bigram",
       y = "Frequency")

#average text features by label
label_summary <- hate_data1 %>%
  group_by(label) %>%
  summarise(
    avg_char_count = mean(char_count, na.rm = TRUE),
    avg_word_count = mean(word_count, na.rm = TRUE),
    avg_exclamation = mean(exclamation_count, na.rm = TRUE),
    avg_question = mean(question_count, na.rm = TRUE),
    avg_hashtag = mean(hashtag_count, na.rm = TRUE),
    avg_mention = mean(mention_count, na.rm = TRUE)
  )

#view the summary table
#View(label_summary)

#save cleaned dataset for future use
write.csv(hate_data1, "hate_data_cleaned.csv", row.names = FALSE)

#make sure row_id exists first
hate_data1 <- hate_data1 %>%
  mutate(row_id = row_number())


#use tf-idf words as predictors
hate_tfidf <- hate_data1 %>%
  select(row_id, content) %>%
  unnest_tokens(word, content) %>%
  anti_join(custom_stop_words, by = "word") %>%
  count(row_id, word, sort = TRUE) %>%
  bind_tf_idf(word, row_id, n)

head(hate_tfidf)

#select top 50 tf-idf words
top_words_model <- hate_tfidf %>%
  group_by(word) %>%
  summarise(avg_tfidf = mean(tf_idf, na.rm = TRUE)) %>%
  slice_max(avg_tfidf, n = 50)

tfidf_model_data <- hate_tfidf %>%
  filter(word %in% top_words_model$word) %>%
  select(row_id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0)

#merge with original model data
model_data2 <- hate_data1 %>%
  left_join(tfidf_model_data, by = "row_id") %>%
  select(-row_id, -content)

#remove NA
model_data2[is.na(model_data2)] <- 0

#train/test split
set.seed(123)
train_index <- createDataPartition(model_data2$label, p = 0.80, list = FALSE)
train_data2 <- model_data2[train_index, ]
test_data2  <- model_data2[-train_index, ]

#train Naive Bayes model
nb_model2 <- naiveBayes(label ~ ., data = train_data2)

#predict
nb_pred2 <- predict(nb_model2, newdata = test_data2)

#evaluation
confusionMatrix(nb_pred2, test_data2$label)