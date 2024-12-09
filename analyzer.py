import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.util import ngrams
from nltk.probability import FreqDist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import re

# Download required NLTK data
import nltk
for package in ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'wordnet', 'stopwords', 'averaged_perceptron_tagger_eng']:
    nltk.download(package, quiet=True)

app = Flask(__name__)

def plot_to_base64(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def analyze_review_text(text):
    """Analyze single review using NLTK techniques"""
    # 1. Basic tokenization
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # 2. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # 3. Stemming & Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stems = [stemmer.stem(w) for w in words]
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    
    # 4. POS Tagging
    pos_tags = pos_tag(words)
    
    # 5. Named Entity Recognition
    named_entities = ne_chunk(pos_tag(words))
    entities = [(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                for chunk in named_entities if hasattr(chunk, 'label')]
    
    # 6. N-grams
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    
    return {
        'sentence_count': len(sentences),
        'word_count': len(words),
        'unique_words': len(set(words)),
        'pos_tags': FreqDist(tag for word, tag in pos_tags),
        'entities': entities,
        'common_bigrams': FreqDist(bigrams).most_common(5),
        'common_trigrams': FreqDist(trigrams).most_common(5)
    }

def analyze_reviews(df):
    """Analyze all reviews and generate insights"""
    print(f"Total reviews analyzed: {len(df)}")
    
    # Basic statistics
    total = len(df)
    positive = len(df[df['rating'] == 2])
    negative = len(df[df['rating'] == 1])
    
    insights = []
    insights.append(f"Total Reviews: {total}")
    insights.append(f"Positive Reviews: {positive} ({positive/total*100:.1f}%)")
    insights.append(f"Negative Reviews: {negative} ({negative/total*100:.1f}%)")
    
    # Train predictive model
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['review_text'])
    y = df['rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Get model predictions and metrics
    y_pred = model.predict(X_test)
    classification_metrics = classification_report(y_test, y_pred, output_dict=True)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_matrix_plot = plot_to_base64(plt)
    
    # Get most important features for each class
    feature_names = vectorizer.get_feature_names_out()
    pos_features = sorted(zip(model.feature_log_prob_[1], feature_names))[-10:]
    neg_features = sorted(zip(model.feature_log_prob_[0], feature_names))[-10:]
    
    # Add model insights
    insights.append("\nModel Performance:")
    insights.append(f"Overall Accuracy: {classification_metrics['accuracy']*100:.1f}%")
    insights.append(f"Positive Review F1-Score: {classification_metrics['2']['f1-score']*100:.1f}%")
    insights.append(f"Negative Review F1-Score: {classification_metrics['1']['f1-score']*100:.1f}%")
    
    # Analyze sample of reviews
    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=42)
    
    pos_reviews = [analyze_review_text(text) for text in sample[sample['rating'] == 2]['review_text']]
    neg_reviews = [analyze_review_text(text) for text in sample[sample['rating'] == 1]['review_text']]
    
    # Length analysis
    pos_lengths = [r['word_count'] for r in pos_reviews]
    neg_lengths = [r['word_count'] for r in neg_reviews]
    insights.append(f"Average positive review length: {np.mean(pos_lengths):.1f} words")
    insights.append(f"Average negative review length: {np.mean(neg_lengths):.1f} words")
    
    # Plot length distribution
    plt.figure(figsize=(10, 6))
    plt.hist([pos_lengths, neg_lengths], label=['Positive', 'Negative'])
    plt.title('Review Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    length_plot = plot_to_base64(plt)
    
    # Collect all linguistic features
    pos_entities = [ent for review in pos_reviews for ent in review['entities']]
    neg_entities = [ent for review in neg_reviews for ent in review['entities']]
    
    # Process bigrams - convert tuples to strings
    pos_bigrams = []
    for review in pos_reviews:
        for bigram, count in review['common_bigrams']:
            pos_bigrams.append((' '.join(bigram), count))
            
    neg_bigrams = []
    for review in neg_reviews:
        for bigram, count in review['common_bigrams']:
            neg_bigrams.append((' '.join(bigram), count))
    
    # Hypothesis Testing Section
    insights.append("\nHypothesis Testing:")
    
    # A & B) Analyze potentially fake/generic reviews
    def get_review_specificity_score(text):
        # Count specific indicators of authenticity
        specific_details = len(re.findall(r'\d+(?:\.\d+)?', text))  # Numbers/measurements
        product_specific = len(re.findall(r'quality|feature|design|material|performance|works?|used?', text, re.I))
        personal_exp = len(re.findall(r'I|my|me|we|our|myself', text, re.I))
        
        # Indicators of generic content
        generic_praise = len(re.findall(r'great|good|nice|awesome|amazing|excellent|best|perfect', text, re.I))
        
        return (specific_details + product_specific + personal_exp) / (generic_praise + 1)
    
    df['specificity_score'] = df['review_text'].apply(get_review_specificity_score)
    df['word_count'] = df['review_text'].str.split().str.len()
    df['is_short'] = df['word_count'] < 20  # Flag very short reviews
    
    # Plot specificity distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='rating', y='specificity_score', data=df)
    plt.title('Review Specificity by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Specificity Score')
    specificity_plot = plot_to_base64(plt)
    
    # Analyze potentially fake reviews
    suspicious_reviews = df[
        (df['specificity_score'] < df['specificity_score'].quantile(0.25)) & 
        (df['is_short'] | (df['word_count'] > df['word_count'].quantile(0.95)))
    ]
    
    insights.append("\nA & B) Review Authenticity Analysis:")
    insights.append(f"- Potentially generic/fake reviews detected: {len(suspicious_reviews)} ({len(suspicious_reviews)/len(df)*100:.1f}%)")
    insights.append(f"- Average specificity score: {df['specificity_score'].mean():.2f}")
    insights.append(f"- Very short reviews (<20 words): {df['is_short'].sum()} ({df['is_short'].mean()*100:.1f}%)")
    
    # C) Analyze customer service related reviews
    service_keywords = r'refund|return|customer service|support|warranty|shipping|delivery'
    df['mentions_service'] = df['review_text'].str.contains(service_keywords, case=False)
    
    service_stats = pd.crosstab(df['mentions_service'], df['rating'])
    service_chi2 = stats.chi2_contingency(service_stats)
    
    service_positive_rate = df[df['mentions_service']]['rating'].mean()
    overall_positive_rate = df['rating'].mean()
    
    insights.append("\nC) Customer Service Mention Analysis:")
    insights.append(f"- Reviews mentioning service terms: {df['mentions_service'].sum()} ({df['mentions_service'].mean()*100:.1f}%)")
    insights.append(f"- Positive rating rate with service mentions: {service_positive_rate:.2f}")
    insights.append(f"- Overall positive rating rate: {overall_positive_rate:.2f}")
    insights.append(f"- Statistical significance: p-value = {service_chi2[1]:.4f}")
    if service_chi2[1] < 0.05:
        insights.append("- Conclusion: Service-related reviews show significantly different rating patterns")
        
    # Add example suspicious reviews
    if len(suspicious_reviews) > 0:
        sample_suspicious = suspicious_reviews.sample(min(3, len(suspicious_reviews)))
        insights.append("\nExample potentially generic/fake reviews:")
        for _, review in sample_suspicious.iterrows():
            insights.append(f"- '{review['review_text'][:100]}...' (Score: {review['specificity_score']:.2f})")
    
    return {
        'summary': {
            'total': total,
            'positive': positive,
            'negative': negative,
            'model_accuracy': classification_metrics['accuracy']
        },
        'insights': insights,
        'plots': {
            'length_distribution': length_plot,
            'confusion_matrix': confusion_matrix_plot,
            'specificity_distribution': specificity_plot
        },
        'linguistic_analysis': {
            'positive': {
                'entities': FreqDist(pos_entities).most_common(10),
                'bigrams': FreqDist(dict(pos_bigrams)).most_common(10),
                'predictive_features': pos_features
            },
            'negative': {
                'entities': FreqDist(neg_entities).most_common(10),
                'bigrams': FreqDist(dict(neg_bigrams)).most_common(10),
                'predictive_features': neg_features
            }
        },
        'model_metrics': classification_metrics,
        'hypothesis_testing': {
            'suspicious_review_count': len(suspicious_reviews),
            'service_mention_stats': {
                'mention_count': int(df['mentions_service'].sum()),
                'positive_rate': float(service_positive_rate),
                'p_value': float(service_chi2[1])
            }
        }
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    try:
        content = file.read().decode('utf-8')
        reviews = []
        total_lines = len(content.split('\n'))
        print(f"Processing {total_lines} lines from file: {file.filename}")
        
        # Parse reviews
        for line in content.split('\n'):
            if line.strip():
                try:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        label = parts[0]
                        rating = int(label.split('__')[-1])
                        review_text = parts[1]
                        reviews.append({
                            'rating': rating,
                            'review_text': review_text
                        })
                except Exception as e:
                    continue
        
        if not reviews:
            return 'No valid reviews found in the file', 400
        
        print(f"Successfully parsed {len(reviews)} valid reviews")
        
        # Analyze reviews
        df = pd.DataFrame(reviews)
        results = analyze_reviews(df)
        
        return render_template('results.html', **results)
        
    except Exception as e:
        return f'Error processing file: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)