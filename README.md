# Predicting Scientific Authour Using NLP
This project was part of a codalab challenge during the Machine Learning course. In this challenge we competed with our peers to try and rank as high as possible on 
the leaderboard. Our team finished in 14th place out of 52. 

## Context
In this project, we predicted the author of scientific papers based on the following features: title, abstract, year, and venue. Natural Language Processing (NLP) techniques were utilized to transform the textual data into usable features. The NLP techniques included converting text data to lowercase and splitting it into words, removing stopwords, and performing stemming/lemmatization to reduce words to their base forms. Then, new features were created from the text data using TfidVectorizer to vectorize the title and abstract features.  Several algorithms
were tested, SGDClassifier and LinearSVC provided the best results. 


