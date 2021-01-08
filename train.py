import argparse
import h5py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to HDF5 file")
    parser.add_argument("--output", required=True, help="Path to output model")
    args = parser.parse_args()

    print("[INFO] Loading dataset ...")
    db = h5py.File(args.db, "r")

    # Pointer for train/test split
    i = int(db["data"].shape[0] * 0.75)

    print("[INFO] Training model ...")
    params = { "C": [0.01] }
    model = LogisticRegression()
    classifier = GridSearchCV(model, params, cv=3, n_jobs=-1)
    classifier.fit(db["data"][:i], db["labels"][:i])

    print("[INFO] Evaluating model ...")
    predictions = classifier.predict(db["data"][i:])
    report = classification_report(db["labels"][i:], predictions, target_names=db["class_names"])
    print(report)

    file = open(args.output, "wb")
    file.write(pickle.dumps(classifier.best_estimator_))
    file.close()

    db.close()

if __name__ == '__main__':
    main()
