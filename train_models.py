
from pathlib import Path
import json, joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.inspection import permutation_importance

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "customer_churn.csv")

X = df.drop(columns=["customer_id","churn"])
y = df["churn"]

num_cols = ["tenure_months","monthly_charges","total_charges"]
cat_cols = [c for c in X.columns if c not in num_cols]

def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

estimators = {
    "Logistic Regression": LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=320,max_depth=12,min_samples_leaf=4,class_weight="balanced",random_state=42,n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=180,learning_rate=0.05,max_depth=3,random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=21,weights="distance"),
}

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.22,random_state=42,stratify=y)

metrics = {}
probabilities = {}
for name,estimator in estimators.items():
    pipe = Pipeline([("preprocessor",preprocessor()),("model",estimator)])
    pipe.fit(X_train,y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]
    metrics[name] = {
        "accuracy":float(accuracy_score(y_test,pred)),
        "precision":float(precision_score(y_test,pred,zero_division=0)),
        "recall":float(recall_score(y_test,pred,zero_division=0)),
        "f1":float(f1_score(y_test,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y_test,proba)),
        "log_loss":float(log_loss(y_test,proba)),
        "confusion_matrix":confusion_matrix(y_test,pred).tolist(),
    }
    probabilities[name]=proba
    slug=name.lower().replace("-","").replace(" ","_")
    joblib.dump(pipe, BASE/"models"/f"{slug}.joblib")

best=max(metrics,key=lambda k:metrics[k]["roc_auc"])
slug=best.lower().replace("-","").replace(" ","_")
best_pipe=joblib.load(BASE/"models"/f"{slug}.joblib")

sx=X_test.sample(min(1200,len(X_test)),random_state=11)
sy=y_test.loc[sx.index]
perm=permutation_importance(best_pipe,sx,sy,scoring="roc_auc",n_repeats=4,random_state=42,n_jobs=-1)
pd.DataFrame({"feature":X.columns,"importance":perm.importances_mean}).sort_values("importance",ascending=False).to_csv(BASE/"models"/"feature_importance.csv",index=False)

meta={
    "best_model":best,
    "selection_metric":"roc_auc",
    "train_rows":len(X_train),
    "test_rows":len(X_test),
    "dataset_rows":len(df),
    "churn_rate":float(df["churn"].mean()),
    "numeric_features":num_cols,
    "categorical_features":cat_cols,
    "metrics":metrics,
}
(BASE/"models"/"model_metrics.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

rows=[]
p=probabilities[best]
for t in np.round(np.arange(.10,.91,.05),2):
    pred=(p>=t).astype(int)
    rows.append({
        "threshold":float(t),
        "accuracy":float(accuracy_score(y_test,pred)),
        "precision":float(precision_score(y_test,pred,zero_division=0)),
        "recall":float(recall_score(y_test,pred,zero_division=0)),
        "f1":float(f1_score(y_test,pred,zero_division=0)),
    })
pd.DataFrame(rows).to_csv(BASE/"models"/"threshold_metrics.csv",index=False)

print("Training complete.")
print("Best model:",best)
print(json.dumps(metrics[best],indent=2))
