from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

X = [
    [1, 40],
    [2, 50],
    [3, 55],
    [4, 70],
    [5, 80],
    [6, 90]
]

y= [0,0,0,1,1,1]

model = DecisionTreeClassifier()

model.fit(X,y)

new_student = [[5,85]]

prediction = model.predict(new_student)
threshold = model.tree_.threshold
print("Prediction: ", prediction)
print("Threshold: ", threshold)

"""print("\nTree rules: ")
print(
    export_text(
        model,
        feature_names=["Study hours"]
    )
)"""