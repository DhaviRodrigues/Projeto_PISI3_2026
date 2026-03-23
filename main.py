import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
    
df = pd.read_csv('IMDB_top_5000_movies.csv')
df['target_class'] = df['rank'].apply(lambda x: 1 if x <= 1000 else 0)

features = ['numVotes', 'runtimeMinutes', 'startYear']
x = df[features].fillna(0)
y = df['target_class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))