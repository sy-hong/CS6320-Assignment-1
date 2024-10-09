'''
CS6320 Assignment 1
Team 24
Zhouhang Sun, Weicheng Liu, Shi Yin Hong
'''

import re
import math
from collections import defaultdict


# 停用词列表（可以根据需要添加更多）
# Stop words list (can add more if needed)
# Given a relatively small corpus, using a standard stopword library such as nltk leads to a relatively
# greater extent of information loss. Thus, we analyze the given corpus and randomly picked
# most commonly seen 58 words.
stop_words = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "the",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "if", "in", "into", "is", "it", "its", "itself", "just"
])


# 从文件中读取语料库
# Read the corpus from a file
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


# 文本预处理（移除标点符号、转为小写等）
# Preprocess text (remove punctuation, lowercase, etc.)
def preprocess_text(text):
    text = text.lower()                  # 转为小写 / Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号 / Remove punctuation
    tokens = text.split()                # 分词 / Tokenize
    # 移除停用词 / Remove stop words
    return [word for word in tokens if word not in stop_words]


# 对语料进行分词
# Tokenize the corpus
def tokenize_corpus(corpus):
    return [preprocess_text(review) for review in corpus]


# 处理未知词，使用 <UNK> 替换低频词
# Handle unknown words: replace low frequency words with <UNK>
def handle_unknown_words_by_threshold(corpus, n=5):
    word_counts = defaultdict(int)
    processed_corpus = []

    # 统计词频 / Count word frequencies
    for review in corpus:
        for token in review:
            word_counts[token] += 1

    # 找出词频低于 n 的词 / Identify low-frequency words
    low_freq_words = set([word for word, count in word_counts.items() if count < n])

    # 将低频词替换为 <UNK> / Replace low-frequency words with <UNK>
    for review in corpus:
        processed_review = []
        for token in review:
            if token in low_freq_words:
                processed_review.append('<UNK>')
            else:
                processed_review.append(token)
        processed_corpus.append(processed_review)

    return processed_corpus, word_counts, low_freq_words


# 从 <UNK> 恢复词，如果它们在训练集中出现了超过5次
# Restore words from <UNK> if they occur more than 5 times in the training set
def replace_and_restore_rare_words_in_validation(val_corpus, unk_dict, train_word_counts, n=5):
    restored_val_corpus = []
    restored_words = set()

    for review in val_corpus:
        restored_review = []
        for token in review:
            if token == '<UNK>':
                # 如果词的频率 >= n，则从训练集中恢复 / If the word's frequency is >= n, restore from the training set
                original_word = [word for word in unk_dict if train_word_counts.get(word, 0) >= n]
                if original_word:
                    restored_review.append(original_word[0])
                    restored_words.add(original_word[0])
                else:
                    restored_review.append('<UNK>')
            else:
                restored_review.append(token)
        restored_val_corpus.append(restored_review)

    # print("从 <UNK> 恢复的词:", restored_words)
    return restored_val_corpus


# 统计unigram（单个词）的频率
# Count unigrams
def count_unigrams(corpus):
    unigram_counts = defaultdict(int)
    total_tokens = 0
    for review in corpus:
        tokens = review
        total_tokens += len(tokens)
        for token in tokens:
            unigram_counts[token] += 1
    return unigram_counts, total_tokens


# 统计bigram（词对）的频率
# Count bigrams
def count_bigrams(corpus):
    bigram_counts = defaultdict(lambda: defaultdict(int))
    unigram_counts = defaultdict(int)
    total_bigrams = 0
    for review in corpus:
        tokens = review
        for i in range(len(tokens) - 1):
            word1, word2 = tokens[i], tokens[i+1]
            unigram_counts[word1] += 1
            bigram_counts[word1][word2] += 1
            total_bigrams += 1

        # 处理最后一个unigram
        # Handle last unigram
        unigram_counts[tokens[-1]] += 1

    return bigram_counts, unigram_counts, total_bigrams


# 使用Add-k平滑计算unigram概率
# Calculate unigram probabilities with Add-k smoothing
def calculate_unigram_probabilities_add_k(unigram_counts, total_tokens, vocab_size, k):
    unigram_probs = {}
    for word, count in unigram_counts.items():
        unigram_probs[word] = (count + k) / (total_tokens + k * vocab_size)
    return unigram_probs


# 使用Add-k平滑计算bigram概率
# Calculate bigram probabilities with Add-k smoothing
def calculate_bigram_probabilities_add_k(bigram_counts, unigram_counts, vocab_size, k):
    bigram_probs = {}
    for word1 in unigram_counts:
        bigram_probs[word1] = {}
        for word2 in unigram_counts:
            bigram_probs[word1][word2] = (bigram_counts.get(word1, {}).get(word2, 0) + k) / (unigram_counts[word1] + k * vocab_size)
    return bigram_probs


# 使用现有的unigram概率计算困惑度
# Calculate unigram perplexity
def calculate_unigram_perplexity(val_corpus, unigram_probs):
    log_prob_sum = 0
    total_words = 0
    for review in val_corpus:
        tokens = review
        for word in tokens:
            # 不再需要处理未见词，假设已经处理过
            prob = unigram_probs.get(word, 0)  
            if prob > 0:
                log_prob_sum += math.log(prob, 2)
            total_words += 1
    perplexity = 2 ** (-log_prob_sum / total_words)
    return perplexity


# 使用现有的bigram概率计算困惑度
# Calculate bigram perplexity
def calculate_bigram_perplexity(val_corpus, bigram_probs):
    log_prob_sum = 0
    total_tokens = 0
    for review in val_corpus:
        tokens = review
        for i in range(1, len(tokens)):
            word1, word2 = tokens[i-1], tokens[i]
            prob = bigram_probs.get(word1, {}).get(word2, 0) 
            if prob > 0:
                log_prob_sum += math.log(prob, 2)
            total_tokens += 1
    perplexity = 2 ** (-log_prob_sum / total_tokens)
    return perplexity


# Process the dataset
train_corpus = read_corpus(r'./A1_DATASET/train.txt')
val_corpus = read_corpus(r'./A1_DATASET/val.txt')

# 对训练集和验证集进行分词
# Tokenize the train and validation corpora
tokenized_train_corpus = tokenize_corpus(train_corpus)
tokenized_val_corpus = tokenize_corpus(val_corpus)

# 处理训练集中的未知词，使用 <UNK> 替换低频词
# Handle unknown words in the train set, replacing low frequency words with <UNK>
processed_train_corpus, train_word_counts, unk_dict = handle_unknown_words_by_threshold(tokenized_train_corpus, n=5)

# 在验证集中将低频词替换为 <UNK>
# Replace rare words in the validation set with <UNK>
processed_val_corpus = [[word if word in train_word_counts else '<UNK>' for word in review] for review in tokenized_val_corpus]

# 恢复验证集中出现超过5次的词的 <UNK>
# Restore words from <UNK> in the validation set if they appear more than 5 times in the train set
restored_val_corpus = replace_and_restore_rare_words_in_validation(processed_val_corpus, unk_dict, train_word_counts, n=5)

# 打印验证语料库示例
# Print a sample of the restored validation corpus
# print("Restored Validation Corpus Sample:", restored_val_corpus[:5])

# 统计unigram（单个词）和bigram（词对）
# Count unigrams and bigrams
unigram_counts, total_tokens = count_unigrams(tokenized_train_corpus)
bigram_counts, unigram_counts, total_bigrams = count_bigrams(tokenized_train_corpus)
vocab_size = len(unigram_counts)  # 词汇表的大小 / Vocabulary size


# 初始化k值和结果列表
# Initialize k value and results list
k = 0.1
perplexity_results = []  # 用于存储困惑度结果 / List to store perplexity results

# 计算并报告不同k值的训练和验证集困惑度
# Calculate and report perplexity for different k-values (training and validation sets)
while k <= 1.0:
    k = round(k, 1)  # 确保k值保留一位小数 / Ensure k is rounded to one decimal place

    # 计算unigram和bigram困惑度 (训练集)
    # Calculate unigram and bigram perplexity for the training set
    unigram_probs_add_k = calculate_unigram_probabilities_add_k(unigram_counts, total_tokens, vocab_size, k)
    bigram_probs_add_k = calculate_bigram_probabilities_add_k(bigram_counts, unigram_counts, vocab_size, k)

    train_unigram_perplexity = calculate_unigram_perplexity(processed_train_corpus, unigram_probs_add_k)
    train_bigram_perplexity = calculate_bigram_perplexity(processed_train_corpus, bigram_probs_add_k)

    # 打印训练集困惑度
    # Print training set perplexities
    print(f"Training Set Unigram Perplexity (k={k}): {train_unigram_perplexity}")
    print(f"Training Set Bigram Perplexity (k={k}): {train_bigram_perplexity}")

    # 计算unigram和bigram困惑度 (验证集)
    # Calculate unigram and bigram perplexity for the validation set
    val_unigram_perplexity = calculate_unigram_perplexity(restored_val_corpus, unigram_probs_add_k)
    val_bigram_perplexity = calculate_bigram_perplexity(restored_val_corpus, bigram_probs_add_k)

    # 打印验证集困惑度
    # Print validation set perplexities
    print(f"Validation Set Unigram Perplexity (k={k}): {val_unigram_perplexity}")
    print(f"Validation Set Bigram Perplexity (k={k}): {val_bigram_perplexity}")

    # 如果k=1, 输出Laplace平滑的信息
    # If k=1, print the Laplace smoothing message
    if k == 1:
        print(f"At k=1, it is Laplace smoothing, and the perplexity is: Unigram Perplexity = {val_unigram_perplexity}, Bigram Perplexity = {val_bigram_perplexity}")

    # 存储结果
    # Store results
    perplexity_results.append((k, val_unigram_perplexity, val_bigram_perplexity))

    # k值增加0.1
    # Increment k by 0.1
    k += 0.1

# 排序并打印最终困惑度结果
# Sort and print the final perplexity results
perplexity_results.sort(key=lambda x: x[1])  # 按照unigram困惑度排序 / Sort by unigram perplexity
print("\nSorted Perplexities (Unigram, Bigram):")
for k_value, unigram_perplexity, bigram_perplexity in perplexity_results:
    print(f"k={round(k_value, 5)}, Unigram Perplexity={unigram_perplexity}, Bigram Perplexity={bigram_perplexity}")


