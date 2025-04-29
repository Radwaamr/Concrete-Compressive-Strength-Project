import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("Concrete_Data_Yeh.csv")

# ده للتأكد بس، ممكن تشيليه بعدين
print(df.columns)

# استخدمي الاسم الصحيح هنا
X = df.drop("csMPa", axis=1)
y = df["csMPa"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "concrete_model.pkl")
