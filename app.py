
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies = pd.read_csv("movies.csv")

movies = movies[
    ["title", "genres", "overview"]
].fillna("")

movies["content"] = (
    movies["genres"]
    + " "
    + movies["overview"]
)

tfidf = TfidfVectorizer(
    stop_words="english"
)

matrix = tfidf.fit_transform(
    movies["content"]
)

similarity = cosine_similarity(
    matrix
)


def recommend(movie):

    try:

        index = movies[
            movies["title"]
            .str.lower()
            ==
            movie.lower()
        ].index[0]

        scores = list(
            enumerate(
                similarity[index]
            )
        )

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        result = []

        for i in scores[1:6]:

            result.append(
                movies.iloc[
                    i[0]
                ]["title"]
            )

        return result

    except:

        return []


@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)

def home():

    selected = ""

    results = []

    theme = "light"

    if request.method == "POST":

        selected = request.form["movie"]

        results = recommend(
            selected
        )

        if selected:

            theme = "dark"

    return render_template(
        "index.html",
        movies=movies[
            "title"
        ].head(
            300
        ).tolist(),
        selected=selected,
        recommendations=results,
        theme=theme
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )
