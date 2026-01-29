import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from google import genai
from google.genai import types
from groq import Groq

import os
from dotenv import load_dotenv

# ---------------- ENV ----------------
load_dotenv()

# ---------------- SCHEDULER ----------------
import schedule
import threading
import time
from external_api import reddit_api, news

def run_scheduler():
    schedule.every().day.at("21:25").do(reddit_api.reddit_api)
    while True:
        schedule.run_pending()
        time.sleep(5)

threading.Thread(target=run_scheduler, daemon=True).start()

# ---------------- STREAMLIT APP ----------------
if __name__ == "__main__":

    st.set_page_config(
        page_title="AI Market Trend & Consumer Sentiment Forecaster",
        layout="wide"
    )

    st.title("AI-Powered Market Trend & Consumer Sentiment Dashboard")
    st.markdown("Consumer sentiment, topic trend, and social insights from reviews, news, and Reddit data")

    # ---------------- DATA LOADER ----------------
    @st.cache_data
    def load_data():
        reviews = pd.read_csv("final data/category_wise_lda_output_with_topic_labels.csv")
        reddit = pd.read_excel("final data/reddit_category_trend_data.xlsx")
        news_df = pd.read_csv("final data/news_data_with_sentiment.csv")

        if "review_date" in reviews.columns:
            reviews["review_date"] = pd.to_datetime(reviews["review_date"], errors="coerce")

        if "published_at" in news_df.columns:
            news_df["published_at"] = pd.to_datetime(news_df["published_at"], errors="coerce")

        if "created_date" in reddit.columns:
            reddit["created_date"] = pd.to_datetime(reddit["created_date"], errors="coerce")

        return reviews, reddit, news_df

    reviews_df, reddit_df, news_df = load_data()

    # ---------------- VECTOR DB ----------------
    @st.cache_resource
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.load_local(
            "consumer_sentiment_faiss1",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_db

    vector_db = load_vector_db()

    # ---------------- LLM CLIENTS ----------------
    @st.cache_resource
    def load_gemini_client():
        return genai.Client(api_key=os.getenv("Gemini_Api_key"))

    @st.cache_resource
    def load_groq_client():
        return Groq(api_key=os.getenv("GROQ_API_KEY"))

    gemini_client = load_gemini_client()
    groq_client = load_groq_client()

    # ---------------- LLM FALLBACK FUNCTION ----------------
    def generate_insight(prompt):
        # Primary: Gemini
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.2
                ),
            )
            return response.text

        # Fallback: Groq (silent)
        except Exception:
            try:
                completion = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a market intelligence analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return completion.choices[0].message.content

            except Exception:
                return "Insight service is temporarily unavailable."

    # ---------------- DASHBOARD LAYOUT ----------------
    main_col, right_sidebar = st.columns([3, 1])

    with main_col:

        st.sidebar.header("Filters")

        source_filter = st.sidebar.multiselect(
            "Select Source",
            options=reviews_df["source"].unique(),
            default=reviews_df["source"].unique()
        )

        category_filter = st.sidebar.multiselect(
            "Select Category",
            options=reviews_df["category"].unique(),
            default=reviews_df["category"].unique()
        )

        filtered_reviews = reviews_df[
            (reviews_df["source"].isin(source_filter)) &
            (reviews_df["category"].isin(category_filter))
        ]

        # ---------------- KPIs ----------------
        st.subheader("Key Metrics")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Reviews", len(filtered_reviews))
        c2.metric("Positive %", round((filtered_reviews["sentiment_label"] == "Positive").mean() * 100, 1))
        c3.metric("Negative %", round((filtered_reviews["sentiment_label"] == "Negative").mean() * 100, 1))
        c4.metric("Neutral %", round((filtered_reviews["sentiment_label"] == "Neutral").mean() * 100, 1))

        # ---------------- CHARTS ----------------
        sentiment_dist = filtered_reviews["sentiment_label"].value_counts().reset_index()
        sentiment_dist.columns = ["Sentiment", "Count"]

        st.plotly_chart(
            px.pie(sentiment_dist, names="Sentiment", values="Count", hole=0.4),
            use_container_width=True
        )

    # ---------------- AI PANEL ----------------
    with right_sidebar:
        st.markdown("## 🤖 AI Insight Panel")
        st.caption("Ask questions using reviews, news, and Reddit data")

        user_query = st.text_area("Your Question", height=140)
        ask_btn = st.button("Get Insight", use_container_width=True)

        if ask_btn and user_query:
            with st.spinner("Analyzing Market Intelligence..."):

                results = vector_db.similarity_search(user_query, k=10)
                retrieved_docs = [r.page_content for r in results]

                prompt = f"""
You are a market intelligence analyst.

Use ONLY the information provided in the context.
Do not use bullet points or external knowledge.

Context:
{retrieved_docs}

Question:
{user_query}

Answer:
"""

                answer = generate_insight(prompt)

            st.success("Insight Generated")
            st.write(answer)
