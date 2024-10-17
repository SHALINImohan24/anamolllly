# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt
# df = pd.read_csv(r"C:\Users\Shalini M\Downloads\eventlog (1).csv")
# print(df.info())
# print(df.head())
# df['MessageLength'] = df['Message'].apply(len)
# features = ['MessageLength']
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df[features])
# print(df_scaled)
# iso_forest = IsolationForest(contamination=0.05, random_state=42)
# df['anomaly_if'] = iso_forest.fit_predict(df_scaled)
# df['anomaly_if'] = df['anomaly_if'] == -1  
# print(df[['MessageLength', 'anomaly_if']].head())
# df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
# normal_data = df[~df['anomaly_if']]
# anomalies_if = df[df['anomaly_if']]
# plt.figure(figsize=(10, 6))
# plt.scatter(normal_data['TimeGenerated'], normal_data['MessageLength'], color='blue', label='Normal Data', alpha=0.6)
# plt.scatter(anomalies_if['TimeGenerated'], anomalies_if['MessageLength'], color='red', label='Anomalies', marker='x')
# plt.title('Anomalies Detected by Isolation Forest')
# plt.xlabel('Timestamp')
# plt.ylabel('Message Length')
# plt.legend()
# plt.grid(True)
# plt.show()
# def create_word_form_message(row):
#     machine = row['MachineName']
#     entry_type = row['Message']   
#     message = f"The machine '{machine}' generated an event of message '{entry_type}'."
#     return message
# df['WordFormMessage'] = df.apply(create_word_form_message, axis=1)
# print(df[['WordFormMessage']].head())
# anomalies_if = df[df['anomaly_if']]
# print("Anomaly Messages:")
# print(anomalies_if[['Message']].head(20))  
# total_anomalies = anomalies_if.shape[0]
# print(f"Total number of anomalies: {total_anomalies}")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"C:\Users\Shalini M\Downloads\eventlog (1).csv")

# Check data structure
print(df.info())
print(df.head())

# 1. Text Feature Extraction using TF-IDF (from 'Message' column)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # Adjust max_features as needed
tfidf_matrix = vectorizer.fit_transform(df['Message'])

# 2. Isolation Forest for Anomaly Detection
iso_forest = IsolationForest(contamination='auto', random_state=42)
df['anomaly_if'] = iso_forest.fit_predict(tfidf_matrix.toarray())

# Convert -1 to True for anomalies and 1 to False for normal data
df['anomaly_if'] = df['anomaly_if'] == -1

# Print results of anomaly detection
print(df[['Message', 'anomaly_if']].head())

# 3. Count of Anomalies
total_logs = df.shape[0]
anomalies = df['anomaly_if'].sum()  # Count where 'anomaly_if' is True
normal_logs = total_logs - anomalies

print(f"Total number of logs: {total_logs}")
print(f"Total number of anomalies detected: {anomalies}")
print(f"Percentage of anomalies: {(anomalies / total_logs) * 100:.2f}%")

# Split the data into normal and anomalous logs
normal_data = df[~df['anomaly_if']]
anomalies_if = df[df['anomaly_if']]

# 4. Visualization (Optional, depending on time features)
df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])  # Convert 'TimeGenerated' to datetime if present

plt.figure(figsize=(10, 6))
plt.scatter(normal_data['TimeGenerated'], [1] * len(normal_data), color='blue', label='Normal Data', alpha=0.6)
plt.scatter(anomalies_if['TimeGenerated'], [1] * len(anomalies_if), color='red', label='Anomalies', marker='x')
plt.title('Anomalies Detected by Isolation Forest (Based on Message Content)')
plt.xlabel('Timestamp')
plt.ylabel('Log Event')
plt.legend()
plt.grid(True)
plt.show()

# 5. Word-Based Anomaly Messages (Optional: display human-readable anomaly messages)
print("Anomaly Messages:")
print(anomalies_if[['Message']].head(20))  # Show the first 20 anomalous messages
