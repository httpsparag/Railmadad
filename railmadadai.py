import streamlit as st
import sqlite3
import random
import string
from datetime import datetime
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set the here path for the database
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indian_railway_complaints.db')

# Load dataset for automatic categorization
df = pd.read_csv('new_complaints.csv')  # Ensure you have a valid CSV dataset for training
texts = df['Tweet']
labels = df['Departments']

# Combine similar labels (if needed)
df['Departments'] = df['Departments'].replace({
    'Coach Cleanliness': 'Coach - Cleanliness',
    'Food Department': 'Food - Security'
})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Function to generate a unique reference number
def generate_reference_number():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

# Function to submit a complaint
def submit_complaint(pnr, description, station, seat_number):
    # Predict complaint category
    category = predict_department(description)

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    reference_number = generate_reference_number()
    cursor.execute('''
    INSERT INTO complaints (pnr, complaint_date, complaint_category, complaint_description, complaint_resolved, station, seat_number, reference_number)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (pnr, datetime.now(), category, description, False, station, seat_number, reference_number))
    conn.commit()
    conn.close()
    return reference_number

# Prediction function
def predict_department(text):
    text_tfidf = vectorizer.transform([text])
    return model.predict(text_tfidf)[0]

# Function to check complaint status
def check_complaint_status(ref_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM complaints WHERE reference_number = ?", (ref_number,))
    complaint = cursor.fetchone()
    conn.close()
    return complaint

# Ensure the complaints table exists
def create_complaints_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS complaints (
        complaint_id INTEGER PRIMARY KEY,
        pnr TEXT,
        complaint_date DATETIME,
        complaint_category TEXT,
        complaint_description TEXT,
        complaint_resolved BOOLEAN,
        station TEXT,
        seat_number TEXT,
        reference_number TEXT
    )
    ''')
    conn.commit()
    conn.close()

create_complaints_table()

# Streamlit app
st.set_page_config(page_title="Rail Madad Complaint Portal", page_icon="🚂", layout="wide")

# Custom CSS for Dark Mode and Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e2f;
        color: #ffffff;
    }
    .main .block-container {
        background-color: #2c2c3e;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
    }
    .sidebar .sidebar-content {
        background-color: #2c2c3e;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #357ab8;
        color: #e2e2e2;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
        background-color: #3a3a4d;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #565672;
    }
    .stAlert {
        background-color: #44445c;
        color: white;
        border: 1px solid #565672;
        border-radius: 5px;
        padding: 10px;
    }
    .quick-link {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Layout
col1, col2, col3 = st.columns([2, 5, 2])

# Left Aside
with col1:
    st.sidebar.header("Past Complaints")
    if st.sidebar.button("View Past Complaints", key="view_past_complaints"):
        st.sidebar.write("Feature coming soon!")

# Main Content
with col2:
    st.title("🚂 Rail Madad Complaint Portal")

    # Complaint submission form
    st.header("Submit a New Complaint")

    pnr = st.text_input("PNR Number")
    station = st.text_input("Station")
    seat_number = st.text_input("Seat Number")
    complaint_text = st.text_area("Write your complaint here...")

    if st.button("Submit Complaint", key="submit_complaint"):
        if complaint_text.strip() != "" and pnr and station and seat_number:
            reference_number = submit_complaint(pnr, complaint_text, station, seat_number)
            st.markdown(f"**Check your status by this Reference Number:** {reference_number}")
            st.success("Your complaint has been submitted. We will ensure that your problem gets resolved.", icon="✅")
        else:
            st.error("Please fill in all fields before submitting.")

# Right Aside
with col3:
    st.sidebar.header("Quick Links")

    # Check Status
    st.sidebar.subheader("Check Complaint Status")
    ref_number = st.sidebar.text_input("Enter complaint reference number:")
    if st.sidebar.button("Check Status", key="check_status_sidebar"):
        complaint = check_complaint_status(ref_number)
        if complaint:
            st.sidebar.write(f"Complaint #{ref_number}:")
            st.sidebar.write(f"Category: {complaint[3]}")
            st.sidebar.write(f"Details: {complaint[4][:50]}...")
            st.sidebar.write(f"Status: {'Resolved' if complaint[5] else 'Processing'}")
        else:
            st.sidebar.write("No complaint found with this reference number.")

    # Upload Feature
    st.sidebar.subheader("Upload Files")
    uploaded_file = st.sidebar.file_uploader("Upload supporting documents:", type=["jpg", "png", "jpeg", "pdf"])
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully.")

        # Other Quick Links
    st.sidebar.markdown("<div class='quick-link'>", unsafe_allow_html=True)
    if st.sidebar.button("Get Help", key="get_help"):
        st.sidebar.write("Contact Railmadad Helpline 139")
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

