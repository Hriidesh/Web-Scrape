import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

 #Prepare a labeled dataset


labeled_dataset_file = 'labeled_dataset.txt'

# Step 2: Feature extraction
vectorizer = CountVectorizer()


movie_title = "The Matrix"
movie_reviews_url = "https://www.imdb.com/title/tt0133093/reviews/?ref_=tt_ql_2"  # Example URL for the movie "The Matrix"

response = requests.get(movie_reviews_url)

if response.status_code == 200:
    html_content = response.text
    print("Request successful. HTML content retrieved.")

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract user comments
    comments = []
    comments_elements = soup.find_all("div", {"class": "text"})
    for comment_element in comments_elements:
        comment = comment_element.text.strip()
        comments.append(comment)

    # Extract sentiment labels
    sentiment_labels = []
    labels_elements = soup.find_all("span", {"class": "rating-other-user-rating"})
    for label_element in labels_elements:
        label = label_element.text.strip().split('/')[0]
        sentiment_labels.append("positive" if int(label) >= 6 else "negative")

    # Combine comments and labels into the labeled dataset
    labeled_dataset = zip(comments, sentiment_labels)

    # Write the labeled dataset to the file
    with open(labeled_dataset_file, 'w', encoding='utf-8') as file:
        for comment, label in labeled_dataset:
            file.write(comment + '\t' + label + '\n')

    # Load the labeled dataset from the file
    with open(labeled_dataset_file, 'r', encoding='utf-8') as file:
        dataset_lines = file.readlines()

    X_train = []
    y_train = []

    for line in dataset_lines:
        # Assume each line of the dataset file has the format: <comment>\t<label>
        line = line.strip()
        if line:
            split_line = line.split('\t')
            if len(split_line) == 2:
                comment, label = split_line
                X_train.append(comment)
                y_train.append(label)

    # Convert the comments into numerical feature vectors
    X_train_vectors = vectorizer.fit_transform(X_train)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vectors, y_train)

    # Use the trained model for sentiment analysis
    comments_vectors = vectorizer.transform(comments)
    sentiment_labels = model.predict(comments_vectors)

    # Print sentiment analysis results
    print("Sentiment Analysis Results:")
    for i, sentiment_label in enumerate(sentiment_labels):
        print("Comment", i + 1, "- Sentiment:", sentiment_label)

    # Perform recommendation system based on user preferences
    top_comments = comments[:5]


else:
    print("Error occurred while retrieving the HTML content.")
