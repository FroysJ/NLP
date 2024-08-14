import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Spam messages (each message is a separate entry)
spam_messages = [
    "100% free", "Amazing deal", "Act now! Don’t hesitate!", "As seen on TV",
    "Big bucks", "Bonus offer", "Cash bonus", "Click here", "Credit card offers",
    "Earn money", "Exclusive deal", "Free gift", "Guaranteed satisfaction",
    "Limited time only", "Lose weight now", "Lowest price ever", "Make money fast",
    "No risk", "Offer expires soon", "One-time offer", "Order now", "Risk-free",
    "Special promotion", "Unsecured debt", "Urgent response required", "Win big",
    "You are a winner", "Buy now, limited time offer!"
]

# Non-spam (ham) messages
ham_messages = [
    "Hi, how are you?", "Let's catch up soon.", "Are you available for a meeting tomorrow?",
    "Please find attached the report.", "Lunch at 1 PM?", "Can you review this document?",
    "Looking forward to our call.", "Your appointment is confirmed.", "Happy birthday!",
    "Let's plan for the weekend.", "Good morning!", "Thank you for your help yesterday.",
    "RSVP by Monday.", "Thank you for your patience, I will send the details shortly.",
    "I will be out of town", "Please contact me", "I hope you are doing well.",
    "Have a great day!", "Sincerely", "Thank you for your email."
]

# Create labels: 1 for spam, 0 for ham
spam_labels = [1] * len(spam_messages)
ham_labels = [0] * len(ham_messages)

# Combine spam and ham messages into a single dataset
messages = spam_messages + ham_messages
labels = spam_labels + ham_labels

# Create a DataFrame
df = pd.DataFrame({'message': messages, 'label': labels})

# Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Testing Accuracy: {accuracy:.2f}')


# Predicting spam
def predict(message):
    transform = vectorizer.transform([message])
    prediction = model.predict(transform)
    ret = ("Prediction: ", "Spam" if prediction[0] == 1 else "Ham")
    return ret


for s in spam_messages + ham_messages:
    new_message = s
    transform = vectorizer.transform([new_message])
    prediction = model.predict(transform)
    print("Prediction: ", "Spam" if prediction[0] == 1 else "Ham")
