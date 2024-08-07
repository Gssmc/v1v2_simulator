import streamlit as st
import random
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import spacy
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease, text_standard
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline

# Set page config for better appearance
st.set_page_config(page_title="BS-CUSTOMER", page_icon="💬", layout="wide")

# Configure logging


# Custom CSS for styling
st.markdown("""
    <style>
    /* Set the main background color */
    .css-18e3th9 {
        background-color: #ffeb3b !important;
        color: black;
    }
    /* Set the sidebar background color */
    .css-1d391kg {
        background-color: #ff9800 !important;
        color: white;
    }
    /* Set the button colors */
    .stButton>button {
        background-color: #f44336;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #d32f2f;
        color: white;
    }
    /* Set the selectbox colors */
    .stSelectbox>div>div>button {
        background-color: #ff6f00;
        color: white;
    }
    .stSelectbox>div>div>button:hover {
        background-color: #e65100;
        color: white;
    }
    /* Set the checkbox colors */
    .stCheckbox>div {
        color: white;
    }
    /* Set the chat message colors */
    .stChatMessage {
        background-color: #333333;
        color: white;
    }
    .stChatMessage>div>div>div {
        color: white;
    }
    /* Set the text input colors */
    .stTextInput>div>div>input {
        background-color: #444444;
        color: white;
    }
    .stTextInput>div>div>button {
        background-color: #f44336;
        color: white;
        border: none;
    }
    .stTextInput>div>div>button:hover {
        background-color: #d32f2f;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'chat' not in st.session_state:
        st.session_state['chat'] = [{"content": "Hi, I need your help", "role": "ai"}]
    if 'chat-history' not in st.session_state:
        st.session_state['chat-history'] = [{"content": "Hi, I need your help", "role": "ai"}]
    if 'selected_option' not in st.session_state:
        st.session_state['selected_option'] = None
    if 'last_played_index' not in st.session_state:
        st.session_state['last_played_index'] = -1
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    if 'current_caller_id' not in st.session_state:
        st.session_state['current_caller_id'] = None
    if 'current_question_index' not in st.session_state:
        st.session_state['current_question_index'] = 0
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []
    if 'question_start_time' not in st.session_state:
        st.session_state['question_start_time'] = None
    

initialize_session_state()

# Sidebar navigation
with st.sidebar:

    if st.button("Home"):
        st.session_state['page'] = 'home'
    if st.button("QnAPlayer"):
        st.session_state['page'] = 'QnAPlayer'
    if st.button("QnAPlayer+"):
        st.session_state['page'] = 'QnAPlayer+'
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load questions
def load_questions():
    try:
        return pd.read_csv('C:\\Users\\GokulSivakumar\\OneDrive - revature.com\\Desktop\\AAM_Customer_Simulator\\Data\\Care_questions_no_repeats.csv')
    except FileNotFoundError:
        st.error("Questions dataset not found!")
        return pd.DataFrame(columns=['caller_id', 'question', 'parentintent', 'childintent', 'answer'])

# Save user responses
def save_user_response(domain, sub_domain, question, user_answer, actual_answer, caller_id, response_time):
    response_data = {
        'domain': domain,
        'sub_domain': sub_domain,
        'question': question,
        'user_answer': user_answer,
        'actual_answer': actual_answer,
        'caller_id': caller_id,
        'response_time': response_time
    }
    df = pd.DataFrame([response_data])
    df.to_csv('user_responses.csv', mode='a', header=not os.path.exists('user_responses.csv'), index=False)

 

nlp = spacy.load("en_core_web_md")

def calculate_similarity(user_answer, actual_answer):
    user_doc = nlp(user_answer)
    actual_doc = nlp(actual_answer)
    return user_doc.similarity(actual_doc)

def calculate_relevancy(user_answer, actual_answer):
    vectorizer = TfidfVectorizer().fit_transform([user_answer, actual_answer])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1] * 5  # Scale to 5

def calculate_positivity(user_answer):
    blob = TextBlob(user_answer)
    return (blob.sentiment.polarity + 1) * 2.5  # Scale to 5

def calculate_communication(user_answer):
    # Readability score
    readability_score = flesch_reading_ease(user_answer)
    
    # Grammatical correctness
    doc = nlp(user_answer)
    grammatical_errors = sum([1 for token in doc if token.dep_ == 'amod' and token.head.dep_ == 'nsubj'])
    
    # Clarity (using readability score)
    if readability_score >= 60:
        clarity_score = 5
    elif readability_score >= 50:
        clarity_score = 4
    elif readability_score >= 40:
        clarity_score = 3
    elif readability_score >= 30:
        clarity_score = 2
    else:
        clarity_score = 1

    # Grammatical correctness score
    if grammatical_errors == 0:
        grammar_score = 5
    elif grammatical_errors <= 2:
        grammar_score = 4
    elif grammatical_errors <= 4:
        grammar_score = 3
    elif grammatical_errors <= 6:
        grammar_score = 2
    else:
        grammar_score = 1

    # Average the clarity and grammar scores
    communication_score = (clarity_score + grammar_score) / 2
    return communication_score


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create a pipeline for text classification
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def calculate_problem_solving(user_answer):
    # Classify the input text
    results = pipeline(user_answer)

    # Extract the problem-solving score
    problem_solving_score = results[0][1]['score']
    
    # Scale the score to be out of 5
    max_score = 5
    scaled_score = problem_solving_score * max_score
    
    return scaled_score

def analyze_performance(responses_df):
    responses_df['response_time'] = responses_df['response_time'].astype(float)

    responses_df['relevancy'] = responses_df.apply(lambda row: calculate_relevancy(row['user_answer'], row['actual_answer']), axis=1)
    responses_df['positivity'] = responses_df['user_answer'].apply(calculate_positivity)
    responses_df['communication'] = responses_df['user_answer'].apply(calculate_communication)
    responses_df['problem_solving'] = responses_df['user_answer'].apply(calculate_problem_solving)

    responses_df['overall_score'] = responses_df[['relevancy', 'positivity', 'communication', 'problem_solving']].mean(axis=1)

    mean_response_time = responses_df['response_time'].mean()
    overall_score = responses_df['overall_score'].mean()

    return mean_response_time, overall_score, responses_df




def display_about():
    st.header("Welcome to BS-CUSTOMER")
    
    st.markdown("""
        BS-CUSTOMER is a simulator designed to train BS Customer Support agents using three main services: QnAPlayer, QnAPlayer+, and Simulator.
        - **QnAPlayer**: Tests the response of the BS agent.
        - **QnAPlayer+**: Evaluates the relevancy of the BS agent's answers using similarity scores.
        
        Our goal is to provide robust training to BS agents, ensuring they can handle various scenarios effectively.
    """)

    st.subheader("Features")
    st.markdown("""
        - **Service Management**: Get help with managing your services efficiently.
        - **Billing**: Clear your doubts regarding billing and payments.
        - **Account Management**: Manage your account settings and preferences with ease.
    """)

    st.subheader("How to Use")
    st.markdown("""
        To start training with BS-CUSTOMER, select one of the services (QnAPlayer, QnAPlayer+) from the dropdown menu.
    """)

    service_option = st.selectbox("Select Service", ["QnAPlayer", "QnAPlayer+"])

    if service_option == "QnAPlayer":
        st.header("About QnAPlayer Service - Question & Answer")
    
        st.markdown("""
            The QnAPlayer service of BS-CUSTOMER simulates interactions to train BS Customer Support agents using a structured Q&A approach:
            
            **How It Works:**
            1. **Select Domain and Sub-Domain**: Choose from various domains and scenario related to customer queries.
            2. **Start Interaction**: Begin the interaction to receive questions from the selected domain.
            3. **Submit Answer**: Provide answers and receive feedback on correctness.
            4. **Next Interaction**: Move to the next set of questions for further training.
            5. **Get Analysis**: View detailed analysis of your interactions.
            
            This service helps in testing the response capabilities of BS agents across different domains effectively.
        """)
    elif service_option == "QnAPlayer+":
        st.header("About QnAPlayer+ Service - Question & Answer with Similarity Score")
    
        st.markdown("""
            The QnAPlayer+ service of BS-CUSTOMER enhances the training process by evaluating answers using similarity scores:
            
            **Key Features:**
            1. **Domain and Sub-Domain Selection**: Choose specific domains and sub-domains for focused training.
            2. **Interactive Q&A**: Engage in Q&A sessions where answers are evaluated based on similarity to correct responses.
            3. **Real-time Feedback**: Receive instant feedback on the accuracy of your responses.
            4. **Detailed Analysis**: Get insights into response times and similarity scores for each interaction.
            
            This service ensures that BS agents provide relevant and accurate information aligned with expected responses.
        """)

    st.subheader("Contact Us")
    st.markdown("""
        Have questions or feedback? Contact us at support@bs-customer.com or visit our [website](https://www.brightspeed.com/).
    """)

def display_v1():
    st.title("QnAPlayer")

    questions_df = load_questions()

    with st.sidebar:
        st.header("Select a Domain and Scenario")
        selected_domain = st.selectbox("Select a Domain", questions_df['parentintent'].unique(), key='domain_v1')
        selected_sub_domain = st.selectbox("Select a Scenario", questions_df[questions_df['parentintent'] == selected_domain]['childintent'].unique(), key='sub_domain_v1')

        if st.session_state.get('selected_option_v1') != (selected_domain, selected_sub_domain):
            st.session_state['selected_option_v1'] = (selected_domain, selected_sub_domain)
            st.session_state.pop('current_caller_id_v1', None)
            st.session_state.pop('current_question_index_v1', None)
            st.session_state.pop('responses_v1', None)
            st.session_state.pop('chat_v1', None)
            st.experimental_rerun()

    if st.session_state.get('current_caller_id_v1') is None:
        st.write("Please select a domain and scenario, then click 'Start Interaction'.")

        if st.button("Start Interaction"):
            filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
            if not filtered_df.empty:
                random_caller_id = random.choice(filtered_df['caller_id'].unique())
                st.session_state['current_caller_id_v1'] = random_caller_id
                st.session_state['current_question_index_v1'] = 0
                st.session_state['responses_v1'] = []
                st.session_state['chat_v1'] = []
                st.session_state['question_start_time_v1'] = None
                st.session_state['user_answer_v1'] = ''
                st.experimental_rerun()
            else:
                st.write("No questions available for this domain and scenario.")
                st.session_state.pop('current_caller_id_v1', None)
    else:
        filtered_df = questions_df[questions_df['caller_id'] == st.session_state['current_caller_id_v1']]
        if st.session_state['current_question_index_v1'] < len(filtered_df):
            current_question = filtered_df.iloc[st.session_state['current_question_index_v1']]
            if st.session_state['question_start_time_v1'] is None:
                st.session_state['question_start_time_v1'] = time.time()
                st.session_state['chat_v1'].append({"content": f"Question {st.session_state['current_question_index_v1'] + 1}: {current_question['question']}", "role": "ai"})

            if len(st.session_state['chat_v1']) > 0:
                for message in st.session_state['chat_v1']:
                    st.chat_message(message['role']).write(message['content'])

            user_answer_key = f"user_answer_v1_{st.session_state['current_question_index_v1']}"

            user_answer = st.text_area("Your Answer", key=user_answer_key)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Submit Answer"):
                    response_time = time.time() - st.session_state['question_start_time_v1']
                    st.session_state['question_start_time_v1'] = None

                    save_user_response(
                        domain=current_question['parentintent'],
                        sub_domain=current_question['childintent'],
                        question=current_question['question'],
                        user_answer=user_answer,
                        actual_answer=current_question['answer'],
                        caller_id=st.session_state['current_caller_id_v1'],
                        response_time=response_time
                    )

                    st.session_state['responses_v1'].append({
                        'question': current_question['question'],
                        'user_answer': user_answer,
                        'actual_answer': current_question['answer'],
                        'response_time': response_time,
                    })
                    st.session_state['chat_v1'].append({"content": user_answer, "role": "user"})
                    st.session_state['current_question_index_v1'] += 1
                    st.experimental_rerun()

            with col2:
                if st.button("Stop Interaction"):
                    st.session_state.pop('current_caller_id_v1', None)
                    st.session_state.pop('current_question_index_v1', None)
                    st.session_state.pop('responses_v1', None)
                    st.session_state.pop('chat_v1', None)
                    st.session_state.pop(user_answer_key, None)
                    st.experimental_rerun()

            with col3:
                if st.button("Restart Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v1'] = random_caller_id
                        st.session_state['current_question_index_v1'] = 0
                        st.session_state['responses_v1'] = []
                        st.session_state['chat_v1'] = []
                        st.session_state['question_start_time_v1'] = None
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v1', None)

        else:
            st.write("You have answered all the questions for this interaction. Please click 'Get Analysis' for analysis or 'Next Interaction' for a new caller interaction.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Next Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v1'] = random_caller_id
                        st.session_state['current_question_index_v1'] = 0
                        st.session_state['responses_v1'] = []
                        st.session_state['chat_v1'] = []
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v1', None)
            
            with col2:
                    if st.button("Get Analysis"):
                        
                        try:
                            st.balloons()
                            responses_df = pd.DataFrame(st.session_state['responses_v1'])
                            st.write("Interaction Details:", responses_df)
                            
                            fig1 = px.bar(
                                responses_df,
                                x='question',
                                y='response_time',
                                title='Response Times for Each Question',
                                labels={'response_time': 'Response Time (s)', 'question': 'Question'},
                            )
                            st.plotly_chart(fig1)


                        except ValueError as e:
                            st.error(f"An error occurred while generating the analysis: {str(e)}")
                            st.write("Please restart the interaction and try again.")
def display_v2():
    st.title("QnAPlayer+")

    questions_df = load_questions()

    with st.sidebar:
        st.header("Select a Domain and Scenario")
        selected_domain = st.selectbox("Select a Domain", questions_df['parentintent'].unique(), key='domain_v2')
        selected_sub_domain = st.selectbox("Select a Scenario", questions_df[questions_df['parentintent'] == selected_domain]['childintent'].unique(), key='sub_domain_v2')

        if st.session_state.get('selected_option_v2') != (selected_domain, selected_sub_domain):
            st.session_state['selected_option_v2'] = (selected_domain, selected_sub_domain)
            st.session_state.pop('current_caller_id_v2', None)
            st.session_state.pop('current_question_index_v2', None)
            st.session_state.pop('responses_v2', None)
            st.session_state.pop('chat_v2', None)
            st.experimental_rerun()

    if st.session_state.get('current_caller_id_v2') is None:
        st.write("Please select a domain and scenario, then click 'Start Interaction'.")

        if st.button("Start Interaction"):
            filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
            if not filtered_df.empty:
                random_caller_id = random.choice(filtered_df['caller_id'].unique())
                st.session_state['current_caller_id_v2'] = random_caller_id
                st.session_state['current_question_index_v2'] = 0
                st.session_state['responses_v2'] = []
                st.session_state['chat_v2'] = []
                st.session_state['question_start_time_v2'] = None
                st.experimental_rerun()
            else:
                st.write("No questions available for this domain and scenario.")
                st.session_state.pop('current_caller_id_v2', None)
    else:
        filtered_df = questions_df[questions_df['caller_id'] == st.session_state['current_caller_id_v2']]
        if st.session_state['current_question_index_v2'] < len(filtered_df):
            current_question = filtered_df.iloc[st.session_state['current_question_index_v2']]
            if st.session_state['question_start_time_v2'] is None:
                st.session_state['question_start_time_v2'] = time.time()
                st.session_state['chat_v2'].append({"content": f"Question {st.session_state['current_question_index_v2'] + 1}: {current_question['question']}", "role": "ai"})

            if len(st.session_state['chat_v2']) > 0:
                for message in st.session_state['chat_v2']:
                    st.chat_message(message['role']).write(message['content'])

            user_answer_key = f"user_answer_v2_{st.session_state['current_question_index_v2']}"

            user_answer = st.text_area("Your Answer", key=user_answer_key)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Submit Answer"):
                    response_time = time.time() - st.session_state['question_start_time_v2']
                    st.session_state['question_start_time_v2'] = None

                    similarity_score = calculate_similarity(user_answer, current_question['answer'])

                    save_user_response(
                        domain=current_question['parentintent'],
                        sub_domain=current_question['childintent'],
                        question=current_question['question'],
                        user_answer=user_answer,
                        actual_answer=current_question['answer'],
                        caller_id=st.session_state['current_caller_id_v2'],
                        response_time=response_time
                    )

                    st.session_state['responses_v2'].append({
                        'question': current_question['question'],
                        'user_answer': user_answer,
                        'actual_answer': current_question['answer'],
                        'response_time': response_time,
                        'similarity_score': similarity_score
                    })
                    st.session_state['chat_v2'].append({"content": user_answer, "role": "user"})
                    st.session_state['current_question_index_v2'] += 1
                    st.experimental_rerun()

            with col2:
                if st.button("Stop Interaction"):
                    st.session_state.pop('current_caller_id_v2', None)
                    st.session_state.pop('current_question_index_v2', None)
                    st.session_state.pop('responses_v2', None)
                    st.session_state.pop('chat_v2', None)
                    st.session_state.pop(user_answer_key, None)
                    st.experimental_rerun()

            with col3:
                if st.button("Restart Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v2'] = random_caller_id
                        st.session_state['current_question_index_v2'] = 0
                        st.session_state['responses_v2'] = []
                        st.session_state['chat_v2'] = []
                        st.session_state['question_start_time_v2'] = None
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v2', None)

        else:
            st.write("You have answered all the questions for this interaction. Please click 'Get Analysis' for analysis or 'Next Interaction' for a new caller interaction.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Next Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v2'] = random_caller_id
                        st.session_state['current_question_index_v2'] = 0
                        st.session_state['responses_v2'] = []
                        st.session_state['chat_v2'] = []
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v2', None)

            with col2:
                if st.button("Get Analysis"):
                    try:
                        st.balloons()
                        responses_df = pd.DataFrame(st.session_state['responses_v2'])

                        mean_response_time, overall_score, detailed_df = analyze_performance(responses_df)

                        st.write("Interaction Details:")
                        st.dataframe(detailed_df)

                        # Save interaction details to CSV
                        detailed_df.to_csv('interaction_details.csv', mode='a', header=not os.path.exists('interaction_details.csv'), index=False)

                        # Store the interaction analysis
                        if 'interaction_scores' not in st.session_state:
                            st.session_state['interaction_scores'] = []

                        st.session_state['interaction_scores'].append(overall_score)

                        # Display the mean of all interaction scores
                        mean_interaction_score = np.mean(st.session_state['interaction_scores'])
                        st.markdown(f"<h1 style='text-align: center; color: green;'>Your Score: {mean_interaction_score:.2f}/5</h1>", unsafe_allow_html=True)
                    except ValueError as e:
                        st.error(f"An error occurred while generating the analysis: {str(e)}")
                        st.write("Please restart the interaction and try again.")



# Handle navigation
if st.session_state['page'] == 'home':
    display_about()
elif st.session_state['page'] == 'QnAPlayer':
    display_v1()
elif st.session_state['page'] == 'QnAPlayer+':
    display_v2()
else:
    display_about()






