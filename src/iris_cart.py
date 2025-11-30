import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

# ========== PATHS ==========
DATA_PATH = "data/Iris.csv"
RESULTS_PATH = "results/figures/"

os.makedirs(RESULTS_PATH, exist_ok=True)


# ========== A. EXPLORATION ==========
def exploration(db):
    print("=== PROPORTION DE CHAQUE CLASSE ===")
    print(db["Species"].value_counts(normalize=True))

    print("\n=== STATISTIQUES DESCRIPTIVES ===")
    print(db.describe())

    db_corr = db.drop(columns=["Id", "Species"])
    corr = db_corr.corr()

    print("\n=== MATRICE DE CORRÉLATION ===")
    print(corr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation")
    plt.savefig(RESULTS_PATH + "correlation_matrix.png")
    plt.close()

    db_corr.hist(figsize=(10, 8))
    plt.suptitle("Histogrammes")
    plt.savefig(RESULTS_PATH + "histograms.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    db_corr.boxplot()
    plt.title("Boxplots")
    plt.savefig(RESULTS_PATH + "boxplots.png")
    plt.close()


# ========== B. PREPROCESSING ==========
def preprocessing(db):
    print("\n=== VALEURS MANQUANTES ===")
    print(db.isnull().sum())

    le = LabelEncoder()
    db["Species_encoded"] = le.fit_transform(db["Species"])

    X = db.drop(columns=["Id", "Species", "Species_encoded"])
    y = db["Species_encoded"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, X.columns


# ========== C. SPLIT ==========
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ========== D. TRAINING ==========
def train_models(X_train, y_train):
    cart = DecisionTreeClassifier(random_state=42)
    cart.fit(X_train, y_train)

    cart_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
    cart_pruned.fit(X_train, y_train)

    return cart, cart_pruned


# ========== E. VISUALIZATION ==========
def plot_trees(cart, cart_pruned, feature_names, class_names):
    plt.figure(figsize=(14, 9))
    tree.plot_tree(cart, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("CART sans élagage")
    plt.savefig(RESULTS_PATH + "tree_full.png")
    plt.close()

    plt.figure(figsize=(14, 9))
    tree.plot_tree(cart_pruned, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("CART élagué")
    plt.savefig(RESULTS_PATH + "tree_pruned.png")
    plt.close()


# ========== F. EVALUATION ==========
def evaluate(model, X_test, y_test, title="CART"):
    print(f"\n=== ACCURACY {title} ===")
    print(model.score(X_test, y_test))

    cm = confusion_matrix(y_test, model.predict(X_test))
    print(cm)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Matrice de confusion")
    plt.savefig(RESULTS_PATH + "confusion_matrix.png")
    plt.close()

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, model.predict(X_test)))


# ========== G. ROC MULTICLASS ==========
def plot_roc(model, X_test, y_test):
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)

    plt.figure()
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Classe {i} (AUC={auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Multi-classe")
    plt.legend()
    plt.savefig(RESULTS_PATH + "roc_curve.png")
    plt.close()


# ========== MAIN ==========
def main():
    db = pd.read_csv(DATA_PATH)

    exploration(db)

    X, y, le, feature_names = preprocessing(db)

    X_train, X_test, y_train, y_test = split_data(X, y)

    cart, cart_pruned = train_models(X_train, y_train)

    plot_trees(cart, cart_pruned, feature_names, le.classes_)

    evaluate(cart, X_test, y_test, "CART")
    evaluate(cart_pruned, X_test, y_test, "CART PRUNED")

    plot_roc(cart, X_test, y_test)

    print("\n✅ TP IRIS CART TERMINÉ")


if __name__ == "__main__":
    main()
