import streamlit as st
import pieces_os_client as pos_client
import requests
import sentiment_analysis
import spam_detector

# Initialize variables
current_api = ""
query = ""

# Streamlit UI
st.title("Streamlit Chatbot with simple NLP")
url = "https://images.g2crowd.com/uploads/product/image/social_landscape/social_landscape_43395aae44695b07e11c5cb6aa5bcc60/pieces-for-developers.png"
st.image(url, caption="Streamlit Chatbot with simple NLP", use_column_width=True, width=10)

# API options
api_options = {
    "Sentiment analysis",
    "Spam detector"
}
selected_api = st.selectbox("Choose an API", options=api_options)

# Initialize chat history for Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me to predict sentiment analysis or detect spam!"}]

# Display chat messages from history on Pieces Bot app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Function to handle different APIs
def get_response_from_api(api, query):
    if api == "Sentiment analysis":
        pre = sentiment_analysis.predict(query)
        response = ""
        for s in pre:
            response += s
        if (response == "Prediction: Positive"
                or response == "Prediction: Negative"
                or response == "Prediction: Neutral"):
            return response
        else:
            return "Error generating response: " + response
    elif api == "Spam detector":
        pre = spam_detector.predict(query)
        response = ""
        for s in pre:
            response += s
        if response == "Prediction: Spam" or response == "Prediction: Ham":
            return response
        else:
            return "Error generating response: " + response
    else:
        return "Invalid selection."


# Process and store Query and Response
def chatbot_function(query, selected_api):
    with st.chat_message("user"):  # Displaying the User Message
        st.markdown(query)
    try:
        # Storing the User Message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            response = get_response_from_api(selected_api, query)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error occurred: {e}")


# Accept the user input
query = st.chat_input("Ask a question")

# Calling the Function when Input is Provided
if query:
    chatbot_function(query, selected_api)
