# Import necessary libraries
import joblib
import json
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load and preprocess the data
data_path = "data.json"
with open(data_path, 'r') as f:
    data = json.load(f)

# Step 2: Flatten the JSON data into a structured DataFrame
df = pd.DataFrame(data['intents'])
dic = {"tag": [], "patterns": [], "responses": []}

for i in range(len(df)):
    ptrns = df.iloc[i]['patterns']
    rspns = df.iloc[i]['responses']
    tag = df.iloc[i]['tag']
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

# Step 3: Create final DataFrame
final_df = pd.DataFrame(dic)

# Step 4: Visualize the distribution of intents
intent_counts = final_df['tag'].value_counts()
fig = go.Figure(data=[go.Bar(x=intent_counts.index, y=intent_counts.values)])
fig.update_layout(title='Distribution of Intents', xaxis_title='Intents', yaxis_title='Count')
fig.show()

# Step 5: Analyze average pattern and response lengths
final_df['pattern_count'] = final_df['patterns'].apply(lambda x: len(x))
final_df['response_count'] = final_df['responses'].apply(lambda x: len(x[0]) if isinstance(x, list) else len(x))
avg_pattern_count = final_df.groupby('tag')['pattern_count'].mean()
avg_response_count = final_df.groupby('tag')['response_count'].mean()

fig = go.Figure()
fig.add_trace(go.Bar(x=avg_pattern_count.index, y=avg_pattern_count.values, name='Average Pattern Count'))
fig.add_trace(go.Bar(x=avg_response_count.index, y=avg_response_count.values, name='Average Response Count'))
fig.update_layout(title='Pattern and Response Analysis', xaxis_title='Intents', yaxis_title='Average Count')
fig.show()

# Step 6: Split data for training and testing
X = final_df['patterns']
y = final_df['tag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 8: Train the SVM model
model = SVC()
model.fit(X_train_vec, y_train)

# Step 9: Predict and evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}\n")

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report = {label: {metric: report[label][metric] for metric in report[label]} for label in report if isinstance(report[label], dict)}

# Step 10: Visualize performance metrics
labels = list(report.keys())
evaluation_metrics = ['precision', 'recall', 'f1-score']
metric_scores = {metric: [report[label][metric] for label in labels if label in report] for metric in evaluation_metrics}

fig = go.Figure()
for metric in evaluation_metrics:
    fig.add_trace(go.Bar(name=metric, x=labels, y=metric_scores[metric]))

fig.update_layout(title='Intent Prediction Model Performance',
                  xaxis_title='Intent',
                  yaxis_title='Score',
                  barmode='group')
fig.show()



# saving the model and vectorizer
import pickle
joblib.dump(model, 'mental_health_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
import joblib

# Step 11: Chatbot Functions
def predict_intent(user_input):
    user_input_vec = vectorizer.transform([user_input])
    intent = model.predict(user_input_vec)[0]
    return intent

def generate_response(intent):
    responses = final_df[final_df['tag'] == intent]['responses'].values
    if len(responses) > 0:
        response_list = responses[0]  # List of responses
        return response_list[0] if isinstance(response_list, list) else response_list
    else:
        return "I'm here to help. Please let me know how I can assist you."

# Step 12: Chatbot Interaction
print("\n--- Welcome to Pandora: Mental Health Chatbot ---\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Pandora: Take care. Goodbye!")
        break
    intent = predict_intent(user_input)
    response = generate_response(intent)
    print("Pandora:", response)




