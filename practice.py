from sklearn.tree import DecisionTreeClassifier

X=[
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
]

y= [

    0,0,0,1,1,1
]

model = DecisionTreeClassifier()

model.fit(X,y)

prediction = model.predict([[7]])
print(prediction)
print("Threshold: ", model.tree_.threshold)