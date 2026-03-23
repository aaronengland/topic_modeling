# Topic Modeling on Newsgroup Posts

In this project, I used topic modeling to automatically discover the hidden themes within a collection of nearly 18,000 online discussion posts — without ever telling the algorithm what the posts were about. Think of it like handing someone a massive stack of unlabeled documents and asking them to sort them into piles by subject. I tested two different approaches: Latent Dirichlet Allocation (LDA), a classic probabilistic method, and BERTopic, a modern approach that uses transformer-based language understanding. Since the dataset comes with 20 known category labels, I was able to compare the topics each model discovered against the true subjects to evaluate how well each method performed.

---

## Dataset Overview

The dataset is the 20 Newsgroups collection, a well-known benchmark in natural language processing. It contains posts from 20 different online discussion groups (newsgroups) spanning topics like computer hardware, politics, religion, science, and sports.

| Property | Value |
|----------|-------|
| Total Posts (after cleaning) | ~17,000 |
| Original Categories | 20 |
| Broad Category Groups | 7 |
| Mean Post Length | ~150 words |
| Source | scikit-learn (built-in dataset) |

The 20 categories fall into seven broad groups: computers (comp.*), recreation (rec.*), science (sci.*), society/religion (soc.*, talk.religion.*), politics (talk.politics.*), miscellaneous (misc.*), and alternative (alt.*). Some categories are very similar to each other — for example, `comp.sys.ibm.pc.hardware` and `comp.sys.mac.hardware` — which makes this dataset a realistic test of whether topic models can distinguish between closely related subjects.

I removed email headers, footers, and quoted text from each post so that the models had to rely on the actual content rather than metadata like email addresses or formatting patterns.

---

## Exploratory Data Analysis

### Category Distribution

![Category Distribution](01_eda/output/01_category_distribution.png)

This chart shows how many posts belong to each of the 20 newsgroup categories. The distribution is fairly balanced — most categories have between 800 and 1,000 posts. This balance is helpful because it means no single topic dominates the dataset, giving both models a fair shot at discovering all 20 themes.

### Broad Category Distribution

![Broad Category Distribution](01_eda/output/02_broad_category_distribution.png)

Grouping the 20 categories into their seven broad families shows that computer-related and recreation topics have the most posts, while miscellaneous and alternative topics have fewer. This broader view is useful because some topic models may discover themes at this higher level rather than distinguishing between, say, `rec.sport.baseball` and `rec.sport.hockey`.

### Document Length Distribution

![Document Length Distribution](01_eda/output/03_document_length_distribution.png)

These histograms show the distribution of post lengths in both characters and words. Most posts are relatively short — the median is around 100-150 words — but the distribution has a long right tail with some posts stretching into the thousands of words. Short posts give topic models less text to work with, making it harder to confidently assign a topic.

### Post Length by Category

![Post Length by Category](01_eda/output/04_length_by_category.png)

This box plot compares post lengths across the 20 categories. Some categories tend to produce longer posts (like talk.politics.misc and talk.religion.misc, where people write detailed arguments), while others tend to be shorter (like misc.forsale, where people post brief item descriptions). These length differences reflect the nature of the conversations in each group.

### Word Frequency

![Word Frequency](01_eda/output/05_word_frequency.png)

This chart shows the 30 most common words across all posts after removing common filler words. Domain-specific terms like "file," "program," "drive," and "system" appear frequently, reflecting the technical nature of many newsgroup discussions. The presence of words like "god," "government," and "space" hints at the religious, political, and scientific topics in the dataset.

### Word Frequency by Broad Category

![Word Frequency by Category](01_eda/output/06_word_frequency_by_category.png)

This panel shows the top 10 words within each broad category group. The differences are striking — computer groups feature words like "drive," "windows," and "card," while science groups feature "space," "medical," and "launch." These distinct vocabularies are exactly what topic models look for when discovering themes. Categories with overlapping vocabulary (like politics and religion, which both use words like "people" and "believe") will be harder for models to separate.

---

## Text Preprocessing

Before feeding text into the models, I cleaned and standardized it through the following pipeline:

1. **Lowercasing** — Convert all text to lowercase for consistency
2. **Email/URL removal** — Strip email addresses and web links
3. **Number removal** — Remove numeric characters
4. **Punctuation removal** — Strip special characters and punctuation
5. **Stopword removal** — Remove common filler words (the, is, at, etc.) plus domain-specific stopwords
6. **Lemmatization** — Reduce words to their base form (e.g., "running" becomes "run")
7. **Short word removal** — Remove words with fewer than 3 characters

**Example:**
- Original: *"I think the new Windows 3.1 drivers for the ATI card are available at ftp://..."*
- Cleaned: *"window driver ati card available"*

### Before vs. After Preprocessing

![Before After Length](02_preprocessing/output/01_before_after_length.png)

These side-by-side histograms show how preprocessing reduces the word count of each post. The cleaning pipeline typically removes 40-60% of the words, stripping away noise while preserving the meaningful content words that topic models need.

### Cleaned Word Frequency

![Cleaned Word Frequency](02_preprocessing/output/02_cleaned_word_frequency.png)

After preprocessing, the most frequent words are more topically informative — technical terms, subject-specific nouns, and action verbs that help distinguish one topic from another. This cleaner vocabulary gives both LDA and BERTopic a stronger signal to work with.

---

## Model 1: Latent Dirichlet Allocation (LDA)

LDA is a probabilistic model that treats each document as a mixture of topics, and each topic as a mixture of words. Think of it like this: when someone writes a post about computer hardware, they draw words from a "hardware" topic (drive, disk, controller) and maybe a bit from a "general computing" topic (system, file, program). LDA works backward from the observed words to figure out what those hidden topics must be.

I used gensim's implementation and tuned three key decisions:
- **Number of topics**: Tested 5 to 30 in steps of 5, selecting the count with the highest coherence score
- **Hyperparameters**: Used Bayesian optimization (Optuna, 10 trials) to tune alpha and eta, which control how topics are distributed across documents and words
- **Vocabulary filtering**: Removed words appearing in fewer than 15 documents or more than 50% of documents to eliminate both rare noise and ubiquitous non-informative terms

### Coherence Scores

![Coherence Scores](03_lda/output/01_coherence_scores.png)

This chart shows the coherence score (c_v) at each tested topic count. Coherence measures how semantically similar the top words within each topic are — higher scores mean the topics are more interpretable to humans. The red dot marks the optimal number of topics. This score guided the final model configuration.

### Top Words per Topic

![Top Words per Topic](03_lda/output/02_top_words_per_topic.png)

Each panel shows the 10 most representative words for one LDA topic, ranked by their weight (probability of appearing in that topic). Well-formed topics have words that clearly relate to a single theme — for example, a topic dominated by "space," "orbit," "launch," and "nasa" clearly corresponds to the sci.space newsgroup. Topics with a mix of unrelated words suggest the model is struggling to find a clean separation.

### Word Clouds

![Word Clouds](03_lda/output/03_word_clouds.png)

These word clouds provide an intuitive visual summary of each topic — larger words have higher weights. They make it easy to quickly scan all topics and get a sense of what each one is about without reading lists of words and numbers.

### Document Distribution by Topic

![Topic Distribution](03_lda/output/04_topic_distribution.png)

This bar chart shows how many documents are assigned to each topic (based on their dominant topic). A relatively even distribution suggests the model is finding balanced, meaningful themes rather than lumping most documents into one or two catch-all topics.

### Topic-Word Heatmap

![Topic Heatmap](03_lda/output/05_topic_heatmap.png)

This heatmap shows the weight of key words across all topics simultaneously. Dark cells indicate that a word is strongly associated with that topic. Ideally, each column (word) should have one or two dark cells — meaning the word is distinctive to specific topics rather than spread evenly across all of them.

---

## Model 2: BERTopic

BERTopic takes a fundamentally different approach. Instead of treating documents as bags of words, it first converts each post into a dense numerical vector (embedding) using a pretrained transformer model that understands the meaning and context of language. It then uses UMAP to reduce the dimensionality of these embeddings and HDBSCAN to cluster similar documents together. Finally, it extracts topic descriptions using a class-based TF-IDF procedure.

The key components:
- **Sentence embeddings**: all-MiniLM-L6-v2 transformer model (384-dimensional vectors)
- **Dimensionality reduction**: UMAP (15 neighbors, 5 components, cosine distance)
- **Clustering**: HDBSCAN (minimum cluster size of 150)
- **Topic representation**: Class-based TF-IDF with bigram support

### Top Words per Topic

![BERTopic Top Words](04_bertopic/output/01_top_words_per_topic.png)

BERTopic's topic words tend to be more specific and contextually relevant than LDA's because the underlying embeddings capture semantic meaning rather than just word co-occurrence. Each panel shows the top words ranked by their c-TF-IDF score — a measure of how distinctive that word is to the topic compared to the rest of the corpus.

### Word Clouds

![BERTopic Word Clouds](04_bertopic/output/02_word_clouds.png)

BERTopic's word clouds often include bigrams (two-word phrases) alongside single words, which can make topics more interpretable. For example, instead of separate words "hard" and "drive," you might see "hard drive" as a single meaningful unit.

### Document Distribution by Topic

![BERTopic Topic Distribution](04_bertopic/output/03_topic_distribution.png)

Unlike LDA, BERTopic can label some documents as outliers — posts that do not clearly belong to any discovered topic. This chart shows the document count for each topic. The presence of outliers is actually a strength: rather than forcing every document into a topic, BERTopic honestly acknowledges when a post does not fit neatly into any category.

### Document Clusters

![Document Clusters](04_bertopic/output/04_document_clusters.png)

This scatter plot shows all documents projected onto two dimensions using UMAP, colored by their assigned topic. Documents that cluster tightly together share similar content, while documents in the gray "outlier" region are too dissimilar to be confidently assigned. Clear, well-separated clusters indicate strong topic distinctiveness.

### Topic Similarity Heatmap

![Topic Similarity](04_bertopic/output/05_topic_similarity.png)

This heatmap shows how similar the discovered topics are to each other, measured by cosine similarity of their word distributions. High similarity between two topics might indicate they could be merged, or it might reflect genuinely related newsgroup categories (like the multiple computer hardware groups). Values close to 0 mean the topics are completely distinct.

---

## Model Comparison

### Coherence Score

![Coherence Comparison](05_comparison/output/01_coherence_comparison.png)

This bar chart compares the coherence scores (c_v) of both models. Coherence measures how well the top words in each topic "make sense together" from a human perspective. Higher is better.

### Topics Discovered and Document Coverage

![Topic Count Comparison](05_comparison/output/02_topic_count_comparison.png)

The left panel compares how many topics each model discovered against the true number of categories (20, shown as a red dashed line). The right panel shows what percentage of documents were successfully assigned to a topic. LDA always assigns every document (100% coverage), while BERTopic may classify some documents as outliers.

### Side-by-Side Topic Comparison

![Side by Side](05_comparison/output/03_side_by_side_topics.png)

This visualization places LDA topics (blue, left) next to BERTopic topics (orange, right) to compare the quality and specificity of the discovered themes. Look for which model produces topics with more coherent, focused word lists versus topics that mix unrelated vocabulary.

### Summary Table

![Summary Table](05_comparison/output/04_summary_table.png)

| Metric | LDA | BERTopic |
|--------|-----|----------|
| Approach | Probabilistic (bag-of-words) | Embedding-based (transformer) |
| Topic Discovery | Fixed count (tuned via coherence) | Automatic (data-driven) |
| Document Coverage | 100% (assigns every document) | Partial (labels outliers) |
| Speed | Fast (minutes) | Slower (embedding generation) |
| Interpretability | Word probability distributions | c-TF-IDF + cluster visualization |
| Bigram Support | No | Yes |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 20 Newsgroups dataset | Provides ground truth categories to validate discovered topics — critical for demonstrating that the models are finding real structure, not noise. |
| Removed headers, footers, and quotes | Forces models to discover topics from content rather than metadata like email addresses or reply formatting. |
| Coherence score (c_v) as primary metric | Measures how semantically coherent each topic's top words are from a human perspective — directly captures what makes a topic "good." |
| LDA + BERTopic comparison | Contrasts the classic probabilistic approach with a modern embedding-based method, showcasing both breadth of knowledge and the evolution of the field. |
| Bayesian hyperparameter tuning (Optuna) | Efficiently searches the hyperparameter space in 10 trials, avoiding the computational cost of exhaustive grid search. |
| Vocabulary filtering (no_below=15, no_above=0.5) | Removes rare words that add noise and ubiquitous words that do not help distinguish topics, improving both speed and quality. |
| HDBSCAN for BERTopic clustering | Unlike K-Means, HDBSCAN does not require specifying the number of clusters upfront and can identify outlier documents that do not belong to any topic. |
| Sentence-transformers embeddings | Captures semantic meaning beyond word co-occurrence — "automobile" and "car" map to similar vectors even though they share no characters. |
