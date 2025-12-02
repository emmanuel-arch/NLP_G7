# MovieLens Dataset (ml-latest-small)

## Summary
This dataset (`ml-latest-small`) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service.  

- **Size:** 100,836 ratings and 3,683 tag applications  
- **Movies:** 9,742  
- **Users:** 610 (each rated at least 20 movies)  
- **Period:** March 29, 1996 – September 24, 2018  
- **Generated:** September 26, 2018  

Users were selected at random for inclusion. No demographic information is included. Each user is represented by an anonymized ID.  

The dataset contains four files:  
- `ratings.csv`  
- `tags.csv`  
- `movies.csv`  
- `links.csv`  

⚠️ This is a *development* dataset. It may change over time and is not intended for shared research results. For benchmark datasets, see [GroupLens Datasets](http://grouplens.org/datasets/).

---

## Usage License
The dataset may be used for research purposes under the following conditions:

- Do not state or imply endorsement from the University of Minnesota or GroupLens Research.  
- Acknowledge use of the dataset in publications (see citation below).  
- Redistribution is allowed, including transformations, under the same license conditions.  
- Commercial use requires permission from GroupLens faculty.  
- Software scripts are provided *as is*, without warranty.  

The University of Minnesota and GroupLens are not liable for damages arising from use of this dataset.  

Questions: <grouplens-info@umn.edu>

---

## Citation
If you use this dataset in publications, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.  
> [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)

---

## About GroupLens
GroupLens is a research group in the Department of Computer Science and Engineering at the University of Minnesota. Since 1992, their projects have explored:

- Recommender systems  
- Online communities  
- Mobile and ubiquitous technologies  
- Digital libraries  
- Local geographic information systems  

GroupLens operates [MovieLens](http://movielens.org), a collaborative filtering movie recommender.  

Collaborators welcome: <grouplens-info@cs.umn.edu>

---

## Content and File Details

### Formatting
- Files are CSV with UTF-8 encoding.  
- Columns with commas are escaped using double quotes (`"`).  
- Ensure your tools support UTF-8 for accented characters.  

### User IDs
- Randomly selected, anonymized.  
- Consistent across `ratings.csv` and `tags.csv`.  

### Movie IDs
- Only movies with at least one rating or tag are included.  
- IDs are consistent across all four files.  
- Example: Movie ID `1` → [MovieLens URL](https://movielens.org/movies/1).  

---

### `ratings.csv`
Structure: userId,movieId,rating,timestamp
- Ratings: 0.5 to 5.0 stars (half-star increments).  
- Ordered by `userId`, then `movieId`.  
- Timestamps: seconds since Jan 1, 1970 (UTC).  

---

### `tags.csv`
Structure: userId,movieId,tag,timestamp
- User-generated metadata (single word or short phrase).  
- Ordered by `userId`, then `movieId`.  
- Timestamps: seconds since Jan 1, 1970 (UTC).  

---

### `movies.csv`
Structure: movieId,title,genres
- Titles include release year in parentheses.  
- Genres are pipe-separated (`|`).  
- Possible genres: Action, Adventure, Animation, Children’s, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, (no genres listed).  

---

### `links.csv`
Structure: movieId,imdbId,tmdbId
- Links to external movie databases:  
  - MovieLens: `https://movielens.org/movies/<movieId>`  
  - IMDb: `http://www.imdb.com/title/tt<imdbId>`  
  - TMDb: `https://www.themoviedb.org/movie/<tmdbId>`  

---

## Cross-Validation
Earlier versions included pre-computed folds or scripts. These are no longer bundled.  
Use modern toolkits (e.g., [LensKit](http://lenskit.org)) for cross-validation in recommender system evaluation.

---

