from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

data=load_iris()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

base_model=DecisionTreeClassifier()
bagging_model=BaggingClassifier(
    estimator=base_model,
    n_estimators=50,
    random_state=42
)

bagging_model.fit(x_train, y_train)
y_pred=bagging_model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm=confusion_matrix(y_test,y_pred)
print("confusion Matrix:\n",cm)
print("\nclassification_report:\n",classification_report(y_test,y_pred))
