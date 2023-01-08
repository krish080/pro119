from cgi import test
import  pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image
import pydotplus

col_names = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Survived']

df = pd.read_csv("data.csv", names=col_names).iloc[1:]

features = ['PassengerId','Pclass','Sex','Age','SibSp','Parch']

X = df[features]

Y = df.Survived

# print(df.head(3))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()

export_graphviz(clf, out_file = dot_data , filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('image.png')
Image(graph.create_png())
