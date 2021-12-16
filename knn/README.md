## Description
This is a simple movie recommender algorithm using KNN. It uses fuzzy matching to find the movie name from the list and recommends top 10 movies or so

## Usage
`./knn.py --movie-name "<insert_your_movie_name>"`

## Example

<pre>
`./knn.py --movie-name "Inglourious Basterds"`
INFO:root:Matched movie name: Inglourious Basterds (2009)       Similarity score: 78
INFO:root:

Here are the top recommended movies
INFO:root:Movie name: Django Unchained (2012)   distance: 31.072495876578696
INFO:root:Movie name: Gran Torino (2008)        distance: 33.8710791088799
INFO:root:Movie name: Shutter Island (2010)     distance: 34.3693177121688
INFO:root:Movie name: Social Network, The (2010)        distance: 34.61574786134195
INFO:root:Movie name: Zombieland (2009) distance: 34.90343822605446
INFO:root:Movie name: Drive (2011)      distance: 35.1390096616282
INFO:root:Movie name: Fighter, The (2010)       distance: 35.316426772820606
INFO:root:Movie name: District 9 (2009) distance: 35.471819801075895
INFO:root:Movie name: True Grit (2010)  distance: 35.63004350263974
</pre>
