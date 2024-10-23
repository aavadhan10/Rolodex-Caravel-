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

# ADD THE NEW LOGGING FUNCTIONS HERE, RIGHT AFTER IMPORTS:
def log_query_and_result(query, result):
    """Log queries and results to a CSV file"""
    log_file = "query_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Create the file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Query", "Result"])
        
        # Append the new query and result
        with open(log_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, query, result])
    except Exception as e:
        st.warning(f"Unable to log query: {str(e)}")

def get_most_asked_queries(n=10):
    """Get the most frequently asked queries from the log"""
    if not os.path.exists("query_log.csv"):
        return pd.DataFrame(columns=["Query", "Count", "Last Result"])
    
    try:
        df = pd.read_csv("query_log.csv")
        query_counts = Counter(df["Query"])
        most_common = query_counts.most_common(n)
        
        results = []
        for query, count in most_common:
            last_result = df[df["Query"] == query].iloc[-1]["Result"]
            results.append({
                "Query": query,
                "Count": count,
                "Last Result": last_result
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        st.warning(f"Error getting query statistics: {str(e)}")
        return pd.DataFrame(columns=["Query", "Count", "Last Result"])

def get_csv_download_link(df, filename="most_asked_queries.csv"):
    """Create a download link for CSV data"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        return href
    except Exception as e:
        st.warning(f"Error creating download link: {str(e)}")
        return ""



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
            text = ''.join(char for char in text if char.isprintable())
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            text = text.replace('Ã¢ÂÂ', "'").replace('Ã¢ÂÂ¨', ", ")
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()
    
    for col in data.columns:
        data[col] = data[col].apply(clean_text)
    
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    return data

def load_availability_data(file_path):
    availability_data = pd.read_csv(file_path)
    # Clean up column names and handle any whitespace in names
    availability_data.columns = [col.strip() for col in availability_data.columns]
    availability_data['What is your name?'] = availability_data['What is your name?'].str.strip()
    
    # Split name into first and last name more safely
    # First create empty columns
    availability_data['First Name'] = ''
    availability_data['Last Name'] = ''
    
    # Then process each name individually
    for idx, row in availability_data.iterrows():
        name_parts = str(row['What is your name?']).split()
        if len(name_parts) >= 2:
            availability_data.at[idx, 'First Name'] = name_parts[0]
            availability_data.at[idx, 'Last Name'] = ' '.join(name_parts[1:])
        elif len(name_parts) == 1:
            availability_data.at[idx, 'First Name'] = name_parts[0]
            availability_data.at[idx, 'Last Name'] = ''
    
    # Clean up any remaining whitespace
    availability_data['First Name'] = availability_data['First Name'].str.strip()
    availability_data['Last Name'] = availability_data['Last Name'].str.strip()
    
    return availability_data

def get_availability_status(row, availability_data):
    """Get availability status for a lawyer"""
    if availability_data is None:
        return "Unknown"
        
    lawyer = availability_data[
        (availability_data['First Name'].str.strip() == row['First Name'].strip()) &
        (availability_data['Last Name'].str.strip() == row['Last Name'].strip())
    ]
    
    if lawyer.empty:
        return "Unknown"
        
    can_take_work = lawyer['Do you have capacity to take on new work?'].iloc[0]
    
    if can_take_work == 'No':
        return "Not Available"
    elif can_take_work == 'Maybe':
        return "Limited Availability"
    
    days_per_week = lawyer['What is your capacity to take on new work for the forseeable future? Days per week'].iloc[0]
    hours_per_month = lawyer['What is your capacity to take on new work for the foreseeable future? Hours per month'].iloc[0]
    
    days_per_week = str(days_per_week)
    hours_per_month = str(hours_per_month)
    
    max_days = max([int(d.strip()) for d in days_per_week.split(';')] if ';' in days_per_week else [int(days_per_week.split()[0])])
    
    if max_days >= 4 and 'More than 80 hours' in hours_per_month:
        return "High Availability"
    elif max_days >= 2:
        return "Moderate Availability"
    else:
        return "Low Availability"

def display_available_lawyers():
    """Display all available lawyers and their capacity"""
    availability_data = load_availability_data('Caravel Law Availability - October 18th, 2024.csv')
    matters_data = load_and_clean_data('BD_Caravel.csv')
    
    # Merge availability data with lawyer information
    available_lawyers = availability_data[availability_data['Do you have capacity to take on new work?'].isin(['Yes', 'Maybe'])]
    
    st.write("### Currently Available Lawyers")
    
    # Create a more detailed view of available lawyers
    for _, lawyer in available_lawyers.iterrows():
        name = f"{lawyer['First Name']} {lawyer['Last Name']}"
        
        # Get lawyer's practice areas from matters_data
        lawyer_info = matters_data[
            (matters_data['First Name'] == lawyer['First Name']) & 
            (matters_data['Last Name'] == lawyer['Last Name'])
        ]
        
        practice_areas = lawyer_info['Area of Practise + Add Info'].iloc[0] if not lawyer_info.empty else "Information not available"
        
        # Create an expander for each lawyer
        with st.expander(f"🧑‍⚖️ {name} - {'Ready for New Work' if lawyer['Do you have capacity to take on new work?'] == 'Yes' else 'Limited Availability'}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Availability Details:**")
                st.write(f"• Days per week: {lawyer['What is your capacity to take on new work for the forseeable future? Days per week']}")
                st.write(f"• Hours per month: {lawyer['What is your capacity to take on new work for the foreseeable future? Hours per month']}")
                st.write(f"• Preferred engagement types: {lawyer['What type of engagement would you like to consider?']}")
            
            with col2:
                st.write("**Practice Areas:**")
                st.write(practice_areas)
            
            # Show any availability notes or upcoming time off
            notes = lawyer['Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?']
            if pd.notna(notes) and notes.lower() not in ['no', 'n/a', 'none', 'nil']:
                st.write("**Availability Notes:**")
                st.write(notes)

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
    """Expand the query with synonyms and related words."""
    expanded_query = []
    for word, tag in nltk.pos_tag(nltk.word_tokenize(query)):
        synsets = wordnet.synsets(word)
        if synsets:
            synonyms = set()
            for synset in synsets:
                synonyms.update(lemma.name().replace('_', ' ') for lemma in synset.lemmas())
            expanded_query.extend(list(synonyms)[:3])
        expanded_query.append(word)
    return ' '.join(expanded_query)

def normalize_query(query):
    """Normalize the query by removing punctuation and converting to lowercase."""
    query = re.sub(r'[^\w\s]', '', query)
    return query.lower()

def preprocess_query(query):
    """Process and extract key terms from the query."""
    tokens = word_tokenize(query)
    tagged = pos_tag(tokens)
    keywords = [word.lower() for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
    stop_words = ['lawyer', 'best', 'top', 'find', 'me', 'give', 'who']
    keywords = [word for word in keywords if word not in stop_words]
    return keywords

def keyword_search(data, query_keywords):
    search_columns = ['Area of Practise + Add Info', 'Industry Experience', 'Expert']
    
    def contains_any_term(text):
        if not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(term in text_lower for term in query_keywords)
    
    masks = [data[col].apply(contains_any_term) for col in search_columns]
    final_mask = masks[0]
    for mask in masks[1:]:
        final_mask |= mask
    
    return data[final_mask]

def calculate_relevance_score(text, query_keywords):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(text_lower.count(keyword) for keyword in query_keywords)

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

def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    # Load availability data
    availability_data = load_availability_data('Caravel Law Availability - October 18th, 2024.csv')
    
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

    # Calculate final relevance score
    relevant_data['relevance_score'] = (relevant_data['keyword_score'] * 0.7) + (relevant_data['semantic_score'] * 0.3)

    # Add availability information
    relevant_data['availability_status'] = relevant_data.apply(
        lambda row: get_availability_status(row, availability_data), axis=1
    )
    
    # Adjust relevance score based on availability
    availability_weights = {
        "High Availability": 1.0,
        "Moderate Availability": 0.8,
        "Low Availability": 0.6,
        "Limited Availability": 0.4,
        "Not Available": 0.1,
        "Unknown": 0.5
    }
    
    relevant_data['availability_weight'] = relevant_data['availability_status'].map(availability_weights)
    relevant_data['final_score'] = relevant_data['relevance_score'] * relevant_data['availability_weight']

    # Sort by final score
    relevant_data = relevant_data.sort_values('final_score', ascending=False)

    # Get top 10 unique lawyers
    top_lawyers = relevant_data[['First Name', 'Last Name']].drop_duplicates().head(10)

    # Get all matters for top lawyers, sorted by relevance
    top_relevant_data = relevant_data[relevant_data[['First Name', 'Last Name']].apply(tuple, axis=1).isin(top_lawyers.apply(tuple, axis=1))]
    top_relevant_data = top_relevant_data.sort_values('final_score', ascending=False)

    # Include availability status in the output
    primary_info = top_relevant_data[['First Name', 'Last Name', 'Level/Title', 'Call', 'Jurisdiction', 'Location', 
                                    'Area of Practise + Add Info', 'Industry Experience', 'Education', 'availability_status']].drop_duplicates(subset=['First Name', 'Last Name'])
    secondary_info = top_relevant_data[['First Name', 'Last Name', 'Area of Practise + Add Info', 'Industry Experience', 
                                      'final_score', 'availability_status']].drop_duplicates(subset=['First Name', 'Last Name'])
     # Get detailed availability information for recommended lawyers
    availability_details = {}
    for _, lawyer in primary_info.iterrows():
        lawyer_availability = availability_data[
            (availability_data['First Name'] == lawyer['First Name']) & 
            (availability_data['Last Name'] == lawyer['Last Name'])
        ]
        if not lawyer_availability.empty:
            availability_details[f"{lawyer['First Name']} {lawyer['Last Name']}"] = {
                'engagement_types': lawyer_availability['What type of engagement would you like to consider?'].iloc[0],
                'days_per_week': lawyer_availability['What is your capacity to take on new work for the forseeable future? Days per week'].iloc[0],
                'hours_per_month': lawyer_availability['What is your capacity to take on new work for the foreseeable future? Hours per month'].iloc[0],
                'comments': lawyer_availability['Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?'].iloc[0]
            }

    primary_context = primary_info.to_string(index=False)
    secondary_context = secondary_info.to_string(index=False)
    availability_context = "\n\nDetailed Availability Information:\n" + "\n".join(
        f"{name}:\n- Engagement Types: {details['engagement_types']}\n- Days per week: {details['days_per_week']}\n- Hours per month: {details['hours_per_month']}\n- Availability Notes: {details['comments']}"
        for name, details in availability_details.items()
    )

    messages = [
        {"role": "system", "content": "You are an expert legal consultant tasked with recommending the most suitable lawyers based on their expertise AND their current availability. Consider both their relevant experience and their capacity to take on new work. Prioritize lawyers who have both the right expertise and good availability.You are an expert legal consultant. When recommending lawyers, only discuss the positive qualities and relevant experience of the lawyers you are specifically recommending. Do not mention or explain anything about other lawyers or why they weren't chosen"},
        {"role": "user", "content": f"Core query keywords: {', '.join(query_keywords)}\nOriginal question: {question}\n\nTop Lawyers Information:\n{primary_context}\n\nRelevant Areas of Practice (including relevance scores):\n{secondary_context}\n{availability_context}\n\nBased on all this information, provide your final recommendation for the most suitable lawyer(s) and explain your reasoning in detail. Consider both expertise and current availability. Recommend up to 3 lawyers, discussing their relevant experience and current availability status. Mention any important availability notes (like upcoming vacations or specific engagement preferences). If no lawyers have both relevant experience and availability, explain this clearly.You are an expert legal consultant. When recommending lawyers, only discuss the positive qualities and relevant experience of the lawyers you are specifically recommending. Do not mention or explain anything about other lawyers or why they weren't chosen"}
    ]

    claude_response = call_claude(messages)
    if not claude_response:
        return

    # Log the query and result
    log_query_and_result(question, claude_response)

    # Display section - Only showing recommendations
    st.write("### Claude's Recommendation:")
    st.write(claude_response)

    if not primary_info.empty:
        # Extract names of recommended lawyers from Claude's response in order
        response_text = claude_response.lower()
        recommended_lawyers = []
        
        # Get all lawyers mentioned in Claude's response in the order they appear
        for _, lawyer in primary_info.iterrows():
            full_name = f"{lawyer['First Name']} {lawyer['Last Name']}"
            if full_name.lower() in response_text:
                # Find the position where this lawyer is first mentioned
                position = response_text.find(full_name.lower())
                recommended_lawyers.append((position, lawyer))
        
        # Sort lawyers by their position in Claude's response
        recommended_lawyers.sort(key=lambda x: x[0])
        # Extract just the lawyer info, now in correct order
        recommended_lawyers = [lawyer for _, lawyer in recommended_lawyers]

        if recommended_lawyers:
            st.write("### Availability Details for Recommended Lawyer(s):")
            availability_data = load_availability_data('Caravel Law Availability - October 18th, 2024.csv')
            
            for lawyer in recommended_lawyers:
                lawyer_availability = availability_data[
                    (availability_data['First Name'] == lawyer['First Name']) & 
                    (availability_data['Last Name'] == lawyer['Last Name'])
                ]
                
                if not lawyer_availability.empty:
                    name = f"{lawyer['First Name']} {lawyer['Last Name']}"
                    with st.expander(f"🧑‍⚖️ {name} - {'Ready for New Work' if lawyer_availability['Do you have capacity to take on new work?'].iloc[0] == 'Yes' else 'Limited Availability'}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Availability Details:**")
                            st.write(f"• Days per week: {lawyer_availability['What is your capacity to take on new work for the forseeable future? Days per week'].iloc[0]}")
                            st.write(f"• Hours per month: {lawyer_availability['What is your capacity to take on new work for the foreseeable future? Hours per month'].iloc[0]}")
                            st.write(f"• Preferred engagement types: {lawyer_availability['What type of engagement would you like to consider?'].iloc[0]}")
                        
                        with col2:
                            st.write("**Practice Areas:**")
                            st.write(lawyer['Area of Practise + Add Info'])
                        
                        notes = lawyer_availability['Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?'].iloc[0]
                        if pd.notna(notes) and notes.lower() not in ['no', 'n/a', 'none', 'nil']:
                            st.write("**Availability Notes:**")
                            st.write(notes)

            # Create ordered DataFrame based on Claude's recommendation order
            recommended_df = pd.DataFrame([{
                col: lawyer[col] for col in primary_info.columns
            } for lawyer in recommended_lawyers])
            
            st.write("### Recommended Lawyers Info:")
            st.write(recommended_df.to_html(index=False), unsafe_allow_html=True)

    else:
        st.write("No lawyers with relevant experience were found for this query.")
# Streamlit app layout
st.title("Rolodex AI Caravel Law: Find Your Legal Match 👨‍⚖️ Utilizing Claude 3.5")
st.write("Ask questions about the skill-matched lawyers for your specific legal needs and their availability:")

# Add tabs for different views
tab1, tab2 = st.tabs(["Search Lawyers by Skillset", "View Available Lawyers Week (Updated 10/23/24)"])

with tab1:
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

with tab2:
    if st.button("Show Available Lawyers"):
        display_available_lawyers()

# Admin section (hidden by default)
if st.experimental_get_query_params().get("admin", [""])[0].lower() == "true":
    st.write("---")
    st.write("## Admin Section")
    if st.button("Download Most Asked Queries and Results"):
        df_most_asked = get_most_asked_queries()
        st.write(df_most_asked)
        st.markdown(get_csv_download_link(df_most_asked), unsafe_allow_html=True)
