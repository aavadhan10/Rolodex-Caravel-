import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic
import re
import unicodedata
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import datetime
import os
import csv
from collections import Counter
import base64

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

def init_anthropic_client():
    claude_api_key = st.secrets["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

def load_and_clean_data(file_path, encoding='utf-8'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # If UTF-8 fails, try latin-1
        data = pd.read_csv(file_path, encoding='latin-1')

    def clean_text(text):
        if isinstance(text, str):
            # Remove non-printable characters
            text = ''.join(char for char in text if char.isprintable())
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            # Replace specific problematic sequences
            text = text.replace('Ã¢ÂÂ', "'").replace('Ã¢ÂÂ¨', ", ")
            # Remove any remaining unicode escape sequences
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Clean column names
    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()

    # Clean text in all columns
    for col in data.columns:
        data[col] = data[col].apply(clean_text)

    # Remove unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    return data

@st.cache_resource
def create_weighted_vector_db(data):
    def weighted_text(row):
        return ' '.join([
            str(row['First Name']),
            str(row['Last Name']),
            str(row['Level/Title']),
            str(row['Call']),
            str(row['Jurisdiction']),
            str(row['Location']),
            str(row['Area of Practise + Add Info']),
            str(row['Industry Experience']),
            str(row['Languages']),
            str(row['Previous In-House Companies']),
            str(row['Previous Companies/Firms']),
            str(row['Education']),
            str(row['Awards/Recognition']),
            str(row['City of Residence']),
            str(row['Notable Items/Personal Details']),
            str(row['Expert'])
        ])

    combined_text = data.apply(weighted_text, axis=1)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(combined_text)
    X_normalized = normalize(X, norm='l2', axis=1, copy=False)
    
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X_normalized.toarray()))
    return index, vectorizer

def call_claude(messages):
    try:
        system_message = messages[0]['content'] if messages[0]['role'] == 'system' else ""
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')
        prompt = f"{system_message}\n\nHuman: {user_message}\n\nAssistant:"

        response = client.completions.create(
            model="claude-2.1",
            prompt=prompt,
            max_tokens_to_sample=500,
            temperature=0.7
        )
        return response.completion
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

def expand_query(query):
    """
    Expand the query with synonyms and related words.
    """
    expanded_query = []
    for word, tag in nltk.pos_tag(nltk.word_tokenize(query)):
        synsets = wordnet.synsets(word)
        if synsets:
            synonyms = set()
            for synset in synsets:
                synonyms.update(lemma.name().replace('_', ' ') for lemma in synset.lemmas())
            expanded_query.extend(list(synonyms)[:3])  # Limit to 3 synonyms per word
        expanded_query.append(word)
    return ' '.join(expanded_query)

def normalize_query(query):
    """
    Normalize the query by removing punctuation and converting to lowercase.
    """
    query = re.sub(r'[^\w\s]', '', query)
    return query.lower()

def log_query_and_result(query, result):
    log_file = "query_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "Result"])
        writer.writerow([timestamp, query, result])

def get_most_asked_queries(n=10):
    if not os.path.exists("query_log.csv"):
        return pd.DataFrame(columns=["Query", "Count", "Last Result"])
    
    df = pd.read_csv("query_log.csv")
    query_counts = Counter(df["Query"])
    most_common = query_counts.most_common(n)
    
    results = []
    for query, count in most_common:
        last_result = df[df["Query"] == query].iloc[-1]["Result"]
        results.append({"Query": query, "Count": count, "Last Result": last_result})
    
    return pd.DataFrame(results)

def get_csv_download_link(df, filename="most_asked_queries.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

def preprocess_query(query):
    # Tokenize and perform POS tagging
    tokens = word_tokenize(query)
    tagged = pos_tag(tokens)
    
    # Extract nouns, proper nouns, and adjectives
    keywords = [word.lower() for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
    
    # Remove common words that might interfere with the search
    stop_words = ['lawyer', 'best', 'top', 'find', 'me', 'give', 'who']
    keywords = [word for word in keywords if word not in stop_words]
    
    return keywords

def keyword_search(data, query_keywords):
    # Define the columns to search in
    search_columns = ['Area of Practise + Add Info', 'Industry Experience', 'Expert']
    
    # Function to check if any query term is in a text
    def contains_any_term(text):
        if not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(term in text_lower for term in query_keywords)
    
    # Create a mask for each search column
    masks = [data[col].apply(contains_any_term) for col in search_columns]
    
    # Combine all masks with OR operation
    final_mask = masks[0]
    for mask in masks[1:]:
        final_mask |= mask
    
    # Return the filtered dataframe
    return data[final_mask]

def calculate_relevance_score(lawyer_text, query_keywords):
    if not isinstance(lawyer_text, str):
        return 0
    lawyer_text_lower = lawyer_text.lower()
    keyword_count = sum(lawyer_text_lower.count(keyword) for keyword in query_keywords)
    return keyword_count
    
def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    # Preprocess the question
    query_keywords = preprocess_query(question)
    
    # Perform keyword search
    keyword_results = keyword_search(matters_data, query_keywords)
    
    # If keyword search yields results, use them. Otherwise, use all data.
    if not keyword_results.empty:
        relevant_data = keyword_results
    else:
        relevant_data = matters_data

    # Calculate keyword-based relevance scores
    relevant_data['keyword_score'] = relevant_data.apply(
        lambda row: calculate_relevance_score(' '.join(row.astype(str)), query_keywords), axis=1
    )

    # Perform semantic search
    question_vec = matters_vectorizer.transform([' '.join(query_keywords)])
    D, I = matters_index.search(normalize(question_vec).toarray(), k=len(relevant_data))
    
    # Add semantic relevance scores
    semantic_scores = 1 / (1 + D[0])
    relevant_data['semantic_score'] = 0
    relevant_data.iloc[I[0], relevant_data.columns.get_loc('semantic_score')] = semantic_scores

    # Calculate final relevance score (you can adjust the weights)
    relevant_data['relevance_score'] = (relevant_data['keyword_score'] * 0.7) + (relevant_data['semantic_score'] * 0.3)

    # Sort by relevance score
    relevant_data = relevant_data.sort_values('relevance_score', ascending=False)

    # Get top 10 unique lawyers
    top_lawyers = relevant_data[['First Name', 'Last Name']].drop_duplicates().head(10)

    # Get all matters for top lawyers, sorted by relevance
    top_relevant_data = relevant_data[relevant_data[['First Name', 'Last Name']].apply(tuple, axis=1).isin(top_lawyers.apply(tuple, axis=1))]
    top_relevant_data = top_relevant_data.sort_values('relevance_score', ascending=False)

    primary_info = top_relevant_data[['First Name', 'Last Name', 'Level/Title', 'Call', 'Jurisdiction', 'Location', 'Area of Practise + Add Info', 'Industry Experience', 'Education']].drop_duplicates(subset=['First Name', 'Last Name'])
    secondary_info = top_relevant_data[['First Name', 'Last Name', 'Area of Practise + Add Info', 'Industry Experience', 'relevance_score']]

    primary_context = primary_info.to_string(index=False)
    secondary_context = secondary_info.to_string(index=False)

    messages = [
        {"role": "system", "content": "You are an expert legal consultant tasked with recommending the most suitable lawyers based on the given information. Analyze the primary information about the lawyers and consider the secondary information about their areas of practice to refine your recommendation. Focus on the core legal expertise required, regardless of how the query is phrased."},
        {"role": "user", "content": f"Core query keywords: {', '.join(query_keywords)}\nOriginal question: {question}\n\nTop Lawyers Information:\n{primary_context}\n\nRelevant Areas of Practice (including relevance scores):\n{secondary_context}\n\nBased on all this information, provide your final recommendation for the most suitable lawyer(s) and explain your reasoning in detail. Recommend up to 3 lawyers, discussing their relevant experience and areas of expertise that specifically relate to the core query. If fewer than 3 lawyers are relevant, only recommend those who are truly suitable. Ensure you consider all provided lawyers, especially those with high relevance scores or whose expertise directly matches the query keywords. If no lawyers have relevant experience, state that no suitable lawyers were found for this specific query."}
    ]

    claude_response = call_claude(messages)
    if not claude_response:
        return

    # Log the query and result
    log_query_and_result(question, claude_response)

    st.write("### Claude's Recommendation:")
    st.write(claude_response)

    if not primary_info.empty:
        st.write("### Top Recommended Lawyer(s) Information:")
        st.write(primary_info.to_html(index=False), unsafe_allow_html=True)

        st.write("### Relevant Areas of Practice of Recommended Lawyer(s):")
        st.write(secondary_info.to_html(index=False), unsafe_allow_html=True)
    else:
        st.write("No lawyers with relevant experience were found for this query.")

# Streamlit app layout
st.title("Rolodex AI Caravel Law: Find Your Legal Match 👨‍⚖️ Utilizing Claude 3.5")
st.write("Ask questions about the skill-matched lawyers for your specific legal needs and their availability:")

default_questions = {
    "Which lawyers have the most experience with intellectual property?": "intellectual property",
    "Can you recommend a lawyer specializing in employment law?": "employment law",
    "Who are the best lawyers for financial cases?": "financial law",
    "Which lawyer should I contact for real estate matters?": "real estate"
}

user_input = st.text_input("Type your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

for question, _ in default_questions.items():
    if st.button(question):
        user_input = question
        break

if user_input:
    progress_bar = st.progress(0)
    progress_bar.progress(10)
    matters_data = load_and_clean_data('BD_Caravel.csv')
    if not matters_data.empty:
        progress_bar.progress(50)
        matters_index, matters_vectorizer = create_weighted_vector_db(matters_data)
        progress_bar.progress(90)
        query_claude_with_data(user_input, matters_data, matters_index, matters_vectorizer)
        progress_bar.progress(100)
    else:
        st.error("Failed to load data.")
    progress_bar.empty()

# Add a hidden section for downloading query data (you can access this by adding ?admin=true to the URL)
if st.experimental_get_query_params().get("admin", [""])[0].lower() == "true":
    st.write("---")
    st.write("## Admin Section")
    if st.button("Download Most Asked Queries and Results"):
        df_most_asked = get_most_asked_queries()
        st.write(df_most_asked)
        st.markdown(get_csv_download_link(df_most_asked), unsafe_allow_html=True)
