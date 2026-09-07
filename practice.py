from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

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

print("\nTree Rules:")
print(
    export_text(
        model,
        feature_names=["study_hours"]
    )
)


new_students = [
    [1.5],
    [2.5],
    [3.2],
    [3.8],
    [5.5],
    [8]
]

predictions = model.predict(new_students)
print(prediction)