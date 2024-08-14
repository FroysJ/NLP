import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Positive
pos_messages = [
    "happy", "love", "excited", "amazing", "wonderful", "incredible", "good", "inspiring",
    "awesome", "positive", "terrific", "wow", "delighted", "like", "joyful", "looking forward",
    "great", "nice", "best", "better", "thanks", "pleasure", "grateful"
]

# Neutral
neut_messages = [
    "indifferent", "don't care", "neutral", "meh", "not too good", "not too bad", "alright",
    "fine", "okay", "ok", "tolerable", "typical", "expected", "normal", "unsurprising", "bland",
    "simple", "ordinary", "generic"
]

# Negative
neg_messages = [
    "terrible", "bad", "dislike", "hate", "worst", "worse", "horrible", "yuck", "disgusting",
    "negative", "awful", "sucks", "gross", "unappealing", "unfortunate", "regretfully",
    "unideal", "despise", "poor", "revolting", "reviling", "repugnant", "sad"
]

# Create labels: 2 for positive, 1 for neutral, 0 for negative
pos_labels = [3] * len(pos_messages)
neut_labels = [2] * len(neut_messages)
neg_labels = [1] * len(neg_messages)

# Combine spam and ham messages into a single dataset
messages = pos_messages + neut_messages + neg_messages
labels = pos_labels + neut_labels + neg_labels

# Create a DataFrame
df = pd.DataFrame({'message': messages, 'label': labels})

# Preprocessing
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Testing Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred, zero_division=0))


def return_label(num):
    if num == 3:
        return "Positive"
    elif num == 1:
        return "Negative"
    elif num == 2:
        return "Neutral"


# Predicting sentiment
def predict(message):
    transform = vectorizer.transform([message])
    prediction = best_model.predict(transform)
    ret = ("Prediction: ", return_label(prediction[0]))
    return ret


for s in messages:
    new_message = s
    transform = vectorizer.transform([new_message])
    prediction = best_model.predict(transform)
    print("Prediction: ", return_label(prediction[0]))
