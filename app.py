import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="InboxShield — NLP Email Spam Detector",
    page_icon="📧",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "emails.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
    return df


@st.cache_resource
def train_models(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    bow = Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])
    tfidf = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
        ),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    bow.fit(X_train, y_train)
    tfidf.fit(X_train, y_train)

    results = {}
    for name, model in [("Bag of Words", bow), ("TF-IDF", tfidf)]:
        pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "report": classification_report(
                y_test, pred, target_names=["not_spam", "spam"], output_dict=True
            ),
            "matrix": confusion_matrix(y_test, pred, labels=["not_spam", "spam"]),
        }

    return bow, tfidf, X_train, X_test, y_train, y_test, results


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "on",
        "is", "are", "be", "this", "that", "your", "you", "we", "our",
        "with", "from", "it", "as", "at", "by", "will", "has", "have",
    }
    return " ".join(t for t in tokens if t not in stop_words)


def ngram_table(texts, n, top_n=12):
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(n, n),
    )
    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    result = pd.DataFrame({"ngram": terms, "count": counts})
    return result.sort_values("count", ascending=False).head(top_n)


def laplace_probability(sentence, corpus_texts, n=2):
    """Simple Laplace-smoothed n-gram sentence probability."""
    tokens = re.findall(r"[a-z]+", sentence.lower())
    corpus = " ".join(corpus_texts).lower()
    corpus_tokens = re.findall(r"[a-z]+", corpus)
    if len(tokens) < n:
        return 0.0

    from collections import Counter
    vocab = set(corpus_tokens)
    vocab_size = max(len(vocab), 1)
    ngrams = Counter(tuple(corpus_tokens[i:i+n]) for i in range(len(corpus_tokens)-n+1))
    prefixes = Counter(tuple(corpus_tokens[i:i+n-1]) for i in range(len(corpus_tokens)-n+1))
    log_prob = 0.0
    count = 0
    for i in range(len(tokens)-n+1):
        gram = tuple(tokens[i:i+n])
        prefix = gram[:-1]
        p = (ngrams[gram] + 1) / (prefixes[prefix] + vocab_size)
        log_prob += __import__("math").log(p)
        count += 1
    return __import__("math").exp(log_prob / max(count, 1))


def show_spacy_analysis(text):
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy model is not installed. Run: python -m spacy download en_core_web_sm")
            return
        doc = nlp(text)
        pos_df = pd.DataFrame(
            [{"token": t.text, "POS": t.pos_, "lemma": t.lemma_} for t in doc if not t.is_space]
        )
        ent_df = pd.DataFrame(
            [{"entity": e.text, "type": e.label_} for e in doc.ents]
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**POS tagging**")
            st.dataframe(pos_df, use_container_width=True)
        with c2:
            st.markdown("**Named Entity Recognition (NER)**")
            if len(ent_df):
                st.dataframe(ent_df, use_container_width=True)
            else:
                st.info("No named entities detected.")
    except Exception as exc:
        st.error(f"NLP linguistic analysis error: {exc}")


def show_sentiment(text):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)
        compound = score["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment", label)
        c2.metric("Compound", f"{compound:.3f}")
        c3.metric("Positive", f"{score['pos']:.2f}")
        c4.metric("Negative", f"{score['neg']:.2f}")
    except Exception as exc:
        st.error(f"Sentiment analysis error: {exc}")


def main():
    df = load_data()
    bow, tfidf, X_train, X_test, y_train, y_test, results = train_models(
        df["text"].tolist(), df["label"].tolist()
    )

    st.title("📧 InboxShield")
    st.caption("A compact NLP project for email spam detection and text analysis")

    with st.sidebar:
        st.header("📌 Project Overview")
        st.markdown(
            """
            **🎯 Goal**  
            Detect whether an email is **spam or not spam** using NLP and machine learning.

            **🔄 Processing Pipeline**  
            Email → Preprocessing → Text Features → ML Model → Prediction

            **🤖 Classification Models**  
            • Bag of Words + Logistic Regression  
            • TF-IDF + Logistic Regression

            **🧠 NLP Techniques**  
            • N-gram analysis  
            • POS tagging & NER  
            • Sentiment analysis  
            • Word2Vec embeddings
            """
        )
        st.divider()
        st.subheader("📊 Dataset")
        st.write(f"**Total emails:** {len(df)}")
        st.write(f"**Spam:** {(df.label == 'spam').sum()}")
        st.write(f"**Not spam:** {(df.label == 'not_spam').sum()}")

    tabs = st.tabs([
        "📬 Email Dataset",
        "🔍 Classify Email",
        "📊 Model Comparison",
        "🔤 N-gram Analysis",
        # "🧠 NLP Analysis",
        # "🧩 Word2Vec",
    ])

    with tabs[0]:
        st.subheader("Complete email dataset")
        st.write("Every sample is already labeled as **spam** or **not spam**.")
        col1, col2 = st.columns(2)
        with col1:
            label_filter = st.multiselect(
                "Filter by class",
                ["spam", "not_spam"],
                default=["spam", "not_spam"],
            )
        with col2:
            search = st.text_input("Search subject/body", placeholder="e.g. prize, meeting, payment")
        view = df[df["label"].isin(label_filter)].copy()
        if search.strip():
            mask = view["text"].str.contains(search.strip(), case=False, na=False)
            view = view[mask]
        display = view[["id", "label", "subject", "body"]].copy()
        display["label"] = display["label"].map({"spam": "🚨 SPAM", "not_spam": "✅ NOT SPAM"})
        st.dataframe(display, use_container_width=True, height=520)

    with tabs[1]:
        st.subheader("Classify a new email")
        subject = st.text_input("Subject", placeholder="Congratulations, you won!")
        body = st.text_area(
            "Email body",
            height=180,
            placeholder="Paste the email content here...",
        )
        if st.button("Classify email", type="primary"):
            if not (subject.strip() or body.strip()):
                st.warning("Enter a subject or email body first.")
            else:
                text = f"{subject} {body}".strip()
                prediction = tfidf.predict([text])[0]
                probabilities = tfidf.predict_proba([text])[0]
                classes = list(tfidf.classes_)
                spam_prob = probabilities[classes.index("spam")]
                if prediction == "spam":
                    st.error(f"🚨 SPAM detected — probability: {spam_prob:.1%}")
                else:
                    st.success(f"✅ NOT SPAM — spam probability: {spam_prob:.1%}")
                st.progress(float(spam_prob))
                st.caption("The model uses TF-IDF features with unigram + bigram information and Logistic Regression.")

    with tabs[2]:
        st.subheader("Bag of Words vs TF-IDF")
        comparison = []
        for name, r in results.items():
            comparison.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']:.2%}",
                "Spam Precision": f"{r['report']['spam']['precision']:.2%}",
                "Spam Recall": f"{r['report']['spam']['recall']:.2%}",
                "F1": f"{r['report']['spam']['f1-score']:.2%}",
            })
        st.table(pd.DataFrame(comparison))

        selected = st.selectbox("Confusion matrix", ["Bag of Words", "TF-IDF"])
        matrix = results[selected]["matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix)
        ax.set_xticks([0, 1], ["Not spam", "Spam"])
        ax.set_yticks([0, 1], ["Not spam", "Spam"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(selected)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, matrix[i, j], ha="center", va="center")
        st.pyplot(fig, clear_figure=True)

        # st.info(
        #     "For the viva: Bag of Words records word frequency, while TF-IDF also "
        #     "down-weights words that occur in many documents. The project uses TF-IDF "
        #     "for the final classifier."
        # )

    with tabs[3]:
        st.subheader("N-gram language modeling")
        st.write("N-grams show how word sequences change as the order increases.")
        n = st.radio("Choose n-gram order", [1, 2, 3], horizontal=True)
        table = ngram_table(df["text"], n)
        st.dataframe(table, use_container_width=True)
        st.bar_chart(table.set_index("ngram")["count"])

        sentence = st.text_input(
            "Estimate sentence probability",
            value="claim your free reward",
        )
        prob = laplace_probability(sentence, df["text"].tolist(), n=max(2, n))
        st.metric(f"Smoothed {max(2, n)}-gram probability", f"{prob:.6f}")
        st.caption(
            "This is a demonstration of n-gram probability with Laplace smoothing; "
            "it is separate from the spam classifier."
        )

    # with tabs[4]:
    #     st.subheader("Preprocessing + POS + NER + sentiment")
    #     sample_options = ["Use a dataset email", "Enter my own email"]
    #     mode = st.radio("Input", sample_options, horizontal=True)
    #     if mode == "Use a dataset email":
    #         idx = st.selectbox(
    #             "Select email",
    #             range(len(df)),
    #             format_func=lambda i: f"#{df.iloc[i]['id']} — {df.iloc[i]['subject']}",
    #         )
    #         analysis_text = df.iloc[idx]["text"]
    #     else:
    #         analysis_text = st.text_area("Text", height=160, value="Your payment is due tomorrow.")

    #     st.markdown("**Raw text**")
    #     st.code(analysis_text)
    #     st.markdown("**After basic preprocessing**")
    #     st.code(clean_text(analysis_text))
    #     show_sentiment(analysis_text)
    #     show_spacy_analysis(analysis_text)

    # with tabs[5]:
    #     st.subheader("Word2Vec semantic representation")
    #     st.write(
    #         "A small Word2Vec model is trained on the project corpus. "
    #         "This demonstrates dense word embeddings and semantic similarity."
    #     )
    #     word = st.text_input("Enter a word from the dataset", value="account")
    #     if st.button("Find similar words"):
    #         try:
    #             from gensim.models import Word2Vec
    #             tokenized = [re.findall(r"[a-z]+", clean_text(x).lower()) for x in df["text"]]
    #             tokenized = [x for x in tokenized if x]
    #             model = Word2Vec(
    #                 sentences=tokenized,
    #                 vector_size=50,
    #                 window=4,
    #                 min_count=1,
    #                 workers=1,
    #                 sg=1,
    #                 epochs=150,
    #                 seed=42,
    #             )
    #             if word.lower() not in model.wv:
    #                 st.warning("That word is not in the training vocabulary. Try: account, payment, reward, project, meeting.")
    #             else:
    #                 sims = model.wv.most_similar(word.lower(), topn=8)
    #                 st.dataframe(
    #                     pd.DataFrame(sims, columns=["word", "similarity"]),
    #                     use_container_width=True
    #                 )
    #                 st.caption(
    #                     "Because this is a tiny classroom corpus, similarities are illustrative rather than production-quality."
    #                 )
    #         except Exception as exc:
    #             st.error(f"Word2Vec error: {exc}")

    st.divider()
    st.caption("Built for an NLP mini-project • Dataset is synthetic classroom data")


if __name__ == "__main__":
    main()
