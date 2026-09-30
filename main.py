from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
emails = [
    "win a free prize now",
    "claim your free reward",
    "you won a cash prize",
    "congratulations you won money",
    "get free money today",
    "exclusive offer claim now",
    "you have won a lottery",
    "win cash and prizes",
    "free gift waiting for you",
    "claim your bonus now",

    "meeting is scheduled for tomorrow",
    "please submit the project report",
    "can we discuss the project",
    "the team meeting starts at ten",
    "please send me the report",
    "your project presentation is tomorrow",
    "let us discuss the assignment",
    "please attend the meeting",
    "the report is ready for review",
    "can you share the project files"
]

label = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]

X_train,X_test,Y_train,Y_test=train_test_split(emails,label,test_size=0.2,random_state=42,stratify=label)
print(X_test)
print(Y_test)
vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)
model=LogisticRegression()
model.fit(X_train_tfidf,Y_train)

Y_pred=model.predict(X_test_tfidf)
print(Y_pred)

accuracy=accuracy_score(Y_test,Y_pred)
print("Accuracy: ",accuracy*100)

email=input("Enter new email")
new_email=[email]
new_email_tfidf=vectorizer.transform(new_email)
print(new_email_tfidf.toarray())

new_pred=model.predict(new_email_tfidf)
if(new_pred[0]==1):
    print("Spam")
else:
    print("Not spam")






  

    
