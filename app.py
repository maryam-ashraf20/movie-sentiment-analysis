import streamlit as st
import joblib
from nltk import word_tokenize
from preprocessing import clean_text, lemmatize_tokens

# Load the trained model and TF-IDF vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# Page configuration
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# App interface
st.title("🎬 Movie Sentiment Analyzer")
st.write(
    "Analyze a movie review using a Machine Learning model "
    "trained with TF-IDF and LinearSVC."
)

st.info(
    "Enter an English movie review to predict whether "
    "its sentiment is positive or negative."
)

review = st.text_area(
    "Movie review",
    placeholder="Example: This movie was absolutely amazing!",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        try:
            # Preprocess the input using the same pipeline
            # used during model training
            cleaned = clean_text(review)
            tokens = word_tokenize(cleaned)
            lemmatized = lemmatize_tokens(tokens)
            processed_review = " ".join(lemmatized)

            # Convert text to TF-IDF features
            features = vectorizer.transform([processed_review])

            # Predict sentiment
            prediction = model.predict(features)[0]

            # Display result
            if prediction == 0:
                st.error("🔴 Predicted sentiment: Negative")
            elif prediction == 1:
                st.success("🟢 Predicted sentiment: Positive")
            else:
                st.warning(f"Unexpected prediction: {prediction}")

            # Show preprocessing details for learning/debugging
            with st.expander("View preprocessing details"):
                st.write("**Cleaned text:**")
                st.write(cleaned)

                st.write("**Tokens:**")
                st.write(tokens)

                st.write("**Lemmatized review:**")
                st.write(processed_review)

        except LookupError as error:
            st.error(
                "An NLTK resource is missing. Check the error "
                "details and download the required resource."
            )
            st.code(str(error))

        except Exception as error:
            st.error("Something went wrong while analyzing the review.")
            st.code(str(error))