from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files

def load_data(file_containing_data: str) -> tuple:
    dataset = load_files(file_containing_data, encoding='utf-8', shuffle=False)
    texts = dataset.get('data')
    labels = dataset.get('target_names')
    return (texts, labels)

def main():

    texts, labels = load_data("categories")

    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(texts)

    # Train classifier
    clf = LogisticRegression()
    clf.fit(text_vectors, labels)

    # Predict new string
    new_text = [""]
    new_text_vector = vectorizer.transform(new_text)
    print(clf.predict(new_text_vector))
    for label in labels:
        print(f"{label.capitalize()}: {clf.score(new_text_vector, [label])}")



if __name__ == "__main__":
    main()
