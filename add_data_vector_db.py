import pandas as pd 


# load files 
reviews_df = pd.read_csv("final data/category_wise_lda_output_with_topic_labels.csv")
news_df = pd.read_csv("final data/news_data_with_sentiment.csv")
reddit_df = pd.read_excel("final data/reddit_category_trend_data.xlsx")

def clean_category(cat):
    if pd.isna(cat):
        return cat
    return str(cat).replace("_", " ").strip()

documents=[]
metadatas=[]

# -----------Reviews------------

for _, row in reviews_df.iterrows():
    text=f"""
    Product: {row["product"] }
    Review: {row["review_text"]}
    Sentiment: {row["sentiment_label"]}
    Category: {clean_category(row["category"])}
    Topic: {row["topic_label"]}    
    """
    
    documents.append(text)
    
    metadatas.append({
        "source":row["source"],
        "product":row["product"],
        "category": clean_category(row["category"]),
        "sentiment": row["sentiment_label"]
    })
    
    
# ---------------News--------------

for _, row in news_df.iterrows():
    text=f"""
    
    News Title: {row["title"]}
    Discription: {row["description"]}
    Content: {row["content"]}
    Category: {clean_category(row["category"])}
    Sentiment: {row["sentiment_label"]}
    
    """
    
    documents.append(text)
    
    metadatas.append({
        "source":"news",
        "category":clean_category(row["category"]),
        "sentiment": row["sentiment_label"]
    })
    
    
    
# ------------Reddit------------
for _, row in reddit_df.iterrows():
    text=f"""
    Reddit Post: {row["title"]}
    Discussion: {row["selftext"]}
    Category: {clean_category(row["category_label"])}
    """
    
    documents.append(text)
    
    metadatas.append({
        "source":"reddit",
        "subreddit":row["subreddit"],
        "category":clean_category(row["category_label"])
    })
    
    

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import FAISS

vector_db = FAISS.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas
)

vector_db.save_local("consumer_sentiment_faiss1")