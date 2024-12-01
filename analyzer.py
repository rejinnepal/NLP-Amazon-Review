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
    
    return {
        'summary': {
            'total': total,
            'positive': positive,
            'negative': negative
        },
        'insights': insights,
        'plots': {
            'length_distribution': length_plot
        },
        'linguistic_analysis': {
            'positive': {
                'entities': FreqDist(pos_entities).most_common(10),
                'bigrams': FreqDist(dict(pos_bigrams)).most_common(10)
            },
            'negative': {
                'entities': FreqDist(neg_entities).most_common(10),
                'bigrams': FreqDist(dict(neg_bigrams)).most_common(10)
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