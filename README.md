# Exploratory Data Analysis on the Film industry

# Table of Contents
1. [Business-understanding](#business-understanding)
2. [Data-understanding](#data-understanding)
3. [Data-Preparation](#data-preparation)
    - [Box-office-Mojo-Data](#box-office-mojo)
        - [foreign-gross](#a-cleaning-foreign_gross-column)
        - [domestic-gross](#b-handling-missing-data-in-domestic_gross-column)
    - [IMDB](#imdb)
        - [Movie-basics-table](#1-movie_basics)
        - [Movie-ratings-table](#2-movie_ratings)
        - [Genre-ratings](#3-genre-ratings)
        - [Directors](#4-directors)
        - [Writers](#5-writers)
4. [Data-Analysis](#data-analysis)
    - [Best Income Generating Studios](#best-income-generating-studios)
        - [Summary Statistics](#summary-statistics)
        - [Distribution of Earnings](#distribution-of-earnings)
        - [Top Earning Studios](#top-earning-studios)
        - [Earnings Over the Years](#earnings-over-the-years)
    - [Best Ratings](#best-ratings)
        - [Genres](#best-rated-genres)
        - [Directors](#best-rated-directors)
        - [Writers](#best-rated-writers)


# Business Understanding

Your company now sees all the big companies creating original video content and they want to get in on the fun. They have decided to create a new movie studio, but they don’t know anything about creating movies. You are charged with exploring what types of films are currently doing the best at the box office. You must then translate those findings into actionable insights that the head of your company's new movie studio can use to help decide what type of films to create.

# Data Understanding

The data was collected from various locations, the different files have different formats.
* [Box Office Mojo](https://www.boxofficemojo.com/)
* [IMDB](https://www.imdb.com/)
* [Rotten Tomatoes](https://www.rottentomatoes.com/)
* [TheMovieDB](https://www.themoviedb.org/)
* [The Numbers](https://www.the-numbers.com/)

Some are compressed CSV (comma-separated values) or TSV (tab-separated values), while the data from IMDB is located in a SQLite database.

<img alt='IMDB data erd' src='movie_data_erd.jpeg' width='700px'/>


# Data Preparation

## Loading the data


```python
# Importing modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import sqlite3

%matplotlib inline
```


```python
# create a connection to sqlite3
conn = sqlite3.connect('zippedData/im.db')
```


```python
# loading the data
bom_df = pd.read_csv('zippedData/bom.movie_gross.csv.gz')
rt_movie_info_df = pd.read_csv('zippedData/rt.movie_info.tsv.gz', delimiter='\t')
rt_reviews_df = pd.read_csv('zippedData/rt.reviews.tsv.gz', delimiter='\t', encoding='latin-1')
tmdb_df = pd.read_csv('zippedData/tmdb.movies.csv.gz')
tn_budgets_df = pd.read_csv('zippedData/tn.movie_budgets.csv.gz')
```

## Box Office Mojo


```python
bom_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3387 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3387 non-null   object 
     1   studio          3382 non-null   object 
     2   domestic_gross  3359 non-null   float64
     3   foreign_gross   2037 non-null   object 
     4   year            3387 non-null   int64  
    dtypes: float64(1), int64(1), object(3)
    memory usage: 132.4+ KB
    

The data has 3397 entries with some data missing in some columns( studio, domestic_gross and foreign_gross).

There are 5 columns

Also the foreign gross is in string format

### (a). cleaning 'foreign_gross' column


```python
bom_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>studio</th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story 3</td>
      <td>BV</td>
      <td>415,000,000</td>
      <td>652000000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice in Wonderland (2010)</td>
      <td>BV</td>
      <td>334,200,000</td>
      <td>691300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Deathly Hallows Part 1</td>
      <td>WB</td>
      <td>296,000,000</td>
      <td>664300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Inception</td>
      <td>WB</td>
      <td>292,600,000</td>
      <td>535700000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shrek Forever After</td>
      <td>P/DW</td>
      <td>238,700,000</td>
      <td>513900000</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
</div>



We start off by converting the foreign_gross column to numeric data type(float64)


```python
# first eliminate commas
bom_df['foreign_gross'] = bom_df.foreign_gross.map(
    lambda x: "".join(x.split(',')) if type(x) == str else x
)
bom_df['foreign_gross'] = bom_df.foreign_gross.astype(float)

bom_df.foreign_gross.dtype
```




    dtype('float64')



Now that the column is in the correct format we can handle the missing values. 

'**The Numbers**' dataset contains budgets for some movies. We can first check some of the missing entries in the dataset.


```python
# The numbers dataset
tn_budgets_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>May 20, 2011</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>$410,600,000</td>
      <td>$241,063,875</td>
      <td>$1,045,663,875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>May 1, 2015</td>
      <td>Avengers: Age of Ultron</td>
      <td>$330,600,000</td>
      <td>$459,005,868</td>
      <td>$1,403,013,963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Dec 15, 2017</td>
      <td>Star Wars Ep. VIII: The Last Jedi</td>
      <td>$317,000,000</td>
      <td>$620,181,382</td>
      <td>$1,316,721,747</td>
    </tr>
  </tbody>
</table>
</div>




```python
tn_budgets_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5782 entries, 0 to 5781
    Data columns (total 6 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   id                 5782 non-null   int64 
     1   release_date       5782 non-null   object
     2   movie              5782 non-null   object
     3   production_budget  5782 non-null   object
     4   domestic_gross     5782 non-null   object
     5   worldwide_gross    5782 non-null   object
    dtypes: int64(1), object(5)
    memory usage: 271.2+ KB
    

no missing entries present

We start by cleaning the numeric columns to eliminate dollar sings and commas


```python
def clean_money_cols(row):
    """Function to clean money columns in the tn_budgets df"""
    i = 3
    cols = ['production_budget', 'domestic_gross',	'worldwide_gross']
    while i < len(row):
        value = row[cols[i - 3]]
        if isinstance(value, str) and value.startswith('$'):
            # remove dollar sign
            value = value[1:]
            # eliminate the commas
            value = float(value.replace(',', ''))
        row[cols[i - 3]] = value
        # increment count
        i += 1
    return row

tn_budgets_df = tn_budgets_df.apply(
    lambda row: clean_money_cols(row), axis=1
)
```

Now that the amounts columns are in the right data type, we can add another column for foreign gross

`foreign_gross = worldwide_gross - domestic_gross`


```python
# creating new column for foreign gross
tn_budgets_df['foreign_gross'] = (
    tn_budgets_df.worldwide_gross - tn_budgets_df.domestic_gross
)
tn_budgets_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>foreign_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>425,000,000</td>
      <td>760,507,625</td>
      <td>2,776,345,279</td>
      <td>2,015,837,654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>May 20, 2011</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>410,600,000</td>
      <td>241,063,875</td>
      <td>1,045,663,875</td>
      <td>804,600,000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>350,000,000</td>
      <td>42,762,350</td>
      <td>149,762,350</td>
      <td>107,000,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>May 1, 2015</td>
      <td>Avengers: Age of Ultron</td>
      <td>330,600,000</td>
      <td>459,005,868</td>
      <td>1,403,013,963</td>
      <td>944,008,095</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Dec 15, 2017</td>
      <td>Star Wars Ep. VIII: The Last Jedi</td>
      <td>317,000,000</td>
      <td>620,181,382</td>
      <td>1,316,721,747</td>
      <td>696,540,365</td>
    </tr>
  </tbody>
</table>
</div>



Fetching all the movies with missing foreign_gross


```python
missing_foreign_gross = bom_df.loc[
    bom_df.foreign_gross.isna(),
    'title'
]

print('Movies missing foreign gross:', len(missing_foreign_gross))
```

    Movies missing foreign gross: 1350
    

Now we try to get the foreign gross in the tn_budgets_df data


```python
new_foreign_gross = tn_budgets_df.loc[
    tn_budgets_df.movie.isin(missing_foreign_gross),
    ['movie', 'foreign_gross']
]

# change columns to change movie to title
new_foreign_gross.columns = ['title', 'foreign_gross']

print('foreign gross entries found in "The Numbers" data:',
      len(new_foreign_gross))

new_foreign_gross.head()
```

    foreign gross entries found in "The Numbers" data: 161
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>foreign_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>588</th>
      <td>Evolution</td>
      <td>60,030,798</td>
    </tr>
    <tr>
      <th>946</th>
      <td>Rock Dog</td>
      <td>14,727,942</td>
    </tr>
    <tr>
      <th>1041</th>
      <td>Bullet to the Head</td>
      <td>13,108,140</td>
    </tr>
    <tr>
      <th>1231</th>
      <td>The Infiltrator</td>
      <td>5,281,296</td>
    </tr>
    <tr>
      <th>1290</th>
      <td>All Eyez on Me</td>
      <td>9,954,553</td>
    </tr>
  </tbody>
</table>
</div>



Found 161 of the missing values in the other dataframe. We now fill in the values in the bom dataframe

First we check if there are 0's in the new values which also indicate missing values and remove them


```python
new_foreign_gross = new_foreign_gross.loc[
    new_foreign_gross.foreign_gross != 0
]

len(new_foreign_gross)
```




    151



Down to 151. we now fill them in the dataframe

we start by defining a helper function


```python
def fill_foreign_gross(row):
    """function to fill the foreign gross column"""
    if row.title in list(new_foreign_gross.title):
        row.foreign_gross = float(
            new_foreign_gross.loc[
                new_foreign_gross.title == row.title,
                'foreign_gross'
            ].values[0]
        )
    return row
```

filling the values


```python
bom_df = bom_df.apply(
    lambda row: fill_foreign_gross(row),
    axis=1
)
```


```python
bom_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3387 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3387 non-null   object 
     1   studio          3382 non-null   object 
     2   domestic_gross  3359 non-null   float64
     3   foreign_gross   2187 non-null   float64
     4   year            3387 non-null   int64  
    dtypes: float64(2), int64(1), object(2)
    memory usage: 132.4+ KB
    

for the remaining missing values we can fill with the mean foreign gross


```python
# Handling the remaining missing values in the foreign_gross
foreign_mean = bom_df.foreign_gross.mean()

bom_df.foreign_gross.fillna(foreign_mean, inplace=True)

# check for missing values
bom_df.foreign_gross.isna().sum()
```




    0



### (b). Handling missing data in domestic_gross column

We do the same for the domestic column, first try to get the missing values in the other dataframe, then fill with the mean


```python
# Get the missing entries in the domestic gross columns
missing_domestic_gross = bom_df.loc[
    bom_df.domestic_gross.isna(),
    'title'
]

print('Movies missing domestic gross:', len(missing_domestic_gross))
```

    Movies missing domestic gross: 28
    


```python
new_domestic_gross = tn_budgets_df.loc[
    tn_budgets_df.movie.isin(missing_domestic_gross),
    ['movie', 'domestic_gross']
]

# change columns to change movie to title
new_domestic_gross.columns = ['title', 'domestic_gross']

print('domestic gross entries found in "The Numbers" data:',
      len(new_domestic_gross))

new_domestic_gross.head()
```

    domestic gross entries found in "The Numbers" data: 2
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>domestic_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3735</th>
      <td>It's a Wonderful Afterlife</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5382</th>
      <td>All the Boys Love Mandy Lane</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We found two of the movies missing the domestic gross but its still 0 meaning that its also missing in the other data source.

Therefore we replace the missing values with mean


```python
# get the mean
domestic_mean = bom_df.domestic_gross.mean()

# fill the null values with mean
bom_df['domestic_gross'] = bom_df.domestic_gross.fillna(domestic_mean)

bom_df.domestic_gross.isna().sum()
```




    0



no more missing values in the domestic gross column


```python
bom_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3387 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3387 non-null   object 
     1   studio          3382 non-null   object 
     2   domestic_gross  3387 non-null   float64
     3   foreign_gross   3387 non-null   float64
     4   year            3387 non-null   int64  
    dtypes: float64(2), int64(1), object(2)
    memory usage: 132.4+ KB
    

The studio column has 5 missing entries


```python
bom_df.studio.fillna('missing', inplace=True)

bom_df.studio.isna().sum()
```




    0



**Creating new column for worldwide gross in the bom_df**


```python
# create new column for worldwide gross
bom_df['worldwide_gross'] = (
    bom_df.foreign_gross + bom_df.domestic_gross
)

bom_df.columns
```




    Index(['title', 'studio', 'domestic_gross', 'foreign_gross', 'year',
           'worldwide_gross'],
          dtype='object')



The Box office mojo data is now clean

## IMDB

The data is in form of a sqlite database

we first check the available tables


```python
table_q = """
SELECT name 
FROM sqlite_master 
WHERE type='table';
"""

pd.read_sql(table_q, conn)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>movie_basics</td>
    </tr>
    <tr>
      <th>1</th>
      <td>directors</td>
    </tr>
    <tr>
      <th>2</th>
      <td>known_for</td>
    </tr>
    <tr>
      <th>3</th>
      <td>movie_akas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>movie_ratings</td>
    </tr>
    <tr>
      <th>5</th>
      <td>persons</td>
    </tr>
    <tr>
      <th>6</th>
      <td>principals</td>
    </tr>
    <tr>
      <th>7</th>
      <td>writers</td>
    </tr>
  </tbody>
</table>
</div>



### 1. movie_basics


```python
mb_query = """
SELECT *
FROM movie_basics
"""

movie_basics = pd.read_sql(mb_query, conn)

movie_basics.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175</td>
      <td>Action,Crime,Drama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114</td>
      <td>Biography,Drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>122</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0069204</td>
      <td>Sabse Bada Sukh</td>
      <td>Sabse Bada Sukh</td>
      <td>2018</td>
      <td>NaN</td>
      <td>Comedy,Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0100275</td>
      <td>The Wandering Soap Opera</td>
      <td>La Telenovela Errante</td>
      <td>2017</td>
      <td>80</td>
      <td>Comedy,Drama,Fantasy</td>
    </tr>
  </tbody>
</table>
</div>



This table contains the basic information about the movies eg title, and genre


```python
movie_basics.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 146144 entries, 0 to 146143
    Data columns (total 6 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   movie_id         146144 non-null  object 
     1   primary_title    146144 non-null  object 
     2   original_title   146123 non-null  object 
     3   start_year       146144 non-null  int64  
     4   runtime_minutes  114405 non-null  float64
     5   genres           140736 non-null  object 
    dtypes: float64(1), int64(1), object(4)
    memory usage: 6.7+ MB
    

We notice missing data in some columns. 

Lets start with the original_title column. Since there are few entries missing, we fill with the tag 'missing'

#### (a). original_title column


```python
movie_basics.original_title.fillna('missing', inplace=True)

movie_basics.original_title.isna().sum()
```




    0



#### (b). runtime_minutes column

For runtime_minutes, we can fill with the average


```python
# get the mean runtime minutes
mean_runtime = movie_basics.runtime_minutes.mean()
movie_basics.runtime_minutes.fillna(mean_runtime, inplace=True)

movie_basics.runtime_minutes.isna().sum()
```




    0



#### (c). genres column

For genres column we can try to get the missing genres in different datasets. But first we obtain the movies with missing genres


```python
# movies with missing genres
missing_genres_df = movie_basics.loc[
    movie_basics['genres'].isna()
]

missing_genres_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>tt0187902</td>
      <td>How Huang Fei-hong Rescued the Orphan from the...</td>
      <td>How Huang Fei-hong Rescued the Orphan from the...</td>
      <td>2011</td>
      <td>86</td>
      <td>None</td>
    </tr>
    <tr>
      <th>22</th>
      <td>tt0253093</td>
      <td>Gangavataran</td>
      <td>Gangavataran</td>
      <td>2018</td>
      <td>134</td>
      <td>None</td>
    </tr>
    <tr>
      <th>35</th>
      <td>tt0306058</td>
      <td>Second Coming</td>
      <td>Second Coming</td>
      <td>2012</td>
      <td>95</td>
      <td>None</td>
    </tr>
    <tr>
      <th>40</th>
      <td>tt0326592</td>
      <td>The Overnight</td>
      <td>The Overnight</td>
      <td>2010</td>
      <td>88</td>
      <td>None</td>
    </tr>
    <tr>
      <th>44</th>
      <td>tt0330811</td>
      <td>Regret Not Speaking</td>
      <td>Regret Not Speaking</td>
      <td>2011</td>
      <td>86</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



We only need the title and the id


```python
missing_genres_df = missing_genres_df.loc[
    :, ['movie_id', 'primary_title', 'original_title']
]

missing_genres_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>tt0187902</td>
      <td>How Huang Fei-hong Rescued the Orphan from the...</td>
      <td>How Huang Fei-hong Rescued the Orphan from the...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>tt0253093</td>
      <td>Gangavataran</td>
      <td>Gangavataran</td>
    </tr>
    <tr>
      <th>35</th>
      <td>tt0306058</td>
      <td>Second Coming</td>
      <td>Second Coming</td>
    </tr>
    <tr>
      <th>40</th>
      <td>tt0326592</td>
      <td>The Overnight</td>
      <td>The Overnight</td>
    </tr>
    <tr>
      <th>44</th>
      <td>tt0330811</td>
      <td>Regret Not Speaking</td>
      <td>Regret Not Speaking</td>
    </tr>
  </tbody>
</table>
</div>



Lets start with the **Rotten tomatoes** data and check if it contains a genres column


```python
rt_movie_info_df.columns
```




    Index(['id', 'synopsis', 'rating', 'genre', 'director', 'writer',
           'theater_date', 'dvd_date', 'currency', 'box_office', 'runtime',
           'studio'],
          dtype='object')




```python
rt_movie_info_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>synopsis</th>
      <th>rating</th>
      <th>genre</th>
      <th>director</th>
      <th>writer</th>
      <th>theater_date</th>
      <th>dvd_date</th>
      <th>currency</th>
      <th>box_office</th>
      <th>runtime</th>
      <th>studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action and Adventure|Classics|Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Drama|Science Fiction and Fantasy</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Illeana Douglas delivers a superb performance ...</td>
      <td>R</td>
      <td>Drama|Musical and Performing Arts</td>
      <td>Allison Anders</td>
      <td>Allison Anders</td>
      <td>Sep 13, 1996</td>
      <td>Apr 18, 2000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>116 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>Michael Douglas runs afoul of a treacherous su...</td>
      <td>R</td>
      <td>Drama|Mystery and Suspense</td>
      <td>Barry Levinson</td>
      <td>Paul Attanasio|Michael Crichton</td>
      <td>Dec 9, 1994</td>
      <td>Aug 27, 1997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>128 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NR</td>
      <td>Drama|Romance</td>
      <td>Rodney Bennett</td>
      <td>Giles Cooper</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200 minutes</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We can use the  'genre' column but there is no title column to compare to the data in missing_genres_df

Lets check **The Movie Db data** 


```python
tmdb_df.columns
```




    Index(['Unnamed: 0', 'genre_ids', 'id', 'original_language', 'original_title',
           'popularity', 'release_date', 'title', 'vote_average', 'vote_count'],
          dtype='object')




```python
tmdb_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>genre_ids</th>
      <th>id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>popularity</th>
      <th>release_date</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[12, 14, 10751]</td>
      <td>12444</td>
      <td>en</td>
      <td>Harry Potter and the Deathly Hallows: Part 1</td>
      <td>34</td>
      <td>2010-11-19</td>
      <td>Harry Potter and the Deathly Hallows: Part 1</td>
      <td>8</td>
      <td>10788</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[14, 12, 16, 10751]</td>
      <td>10191</td>
      <td>en</td>
      <td>How to Train Your Dragon</td>
      <td>29</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>8</td>
      <td>7610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[12, 28, 878]</td>
      <td>10138</td>
      <td>en</td>
      <td>Iron Man 2</td>
      <td>29</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>7</td>
      <td>12368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[16, 35, 10751]</td>
      <td>862</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>28</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>8</td>
      <td>10174</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[28, 878, 12]</td>
      <td>27205</td>
      <td>en</td>
      <td>Inception</td>
      <td>28</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>8</td>
      <td>22186</td>
    </tr>
  </tbody>
</table>
</div>



For this data, there is a title column to compare to, but the genres are id refferencing to a genres table which we dont have access to.



Finally **The Numbers** data


```python
tn_budgets_df.columns
```




    Index(['id', 'release_date', 'movie', 'production_budget', 'domestic_gross',
           'worldwide_gross', 'foreign_gross'],
          dtype='object')



There is no column we can use to get the genres

Since we cant get the genres from the other data sources, we can fill the entries with 'missing' tag


```python
movie_basics.fillna('missing', inplace=True)

movie_basics.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 146144 entries, 0 to 146143
    Data columns (total 6 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   movie_id         146144 non-null  object 
     1   primary_title    146144 non-null  object 
     2   original_title   146144 non-null  object 
     3   start_year       146144 non-null  int64  
     4   runtime_minutes  146144 non-null  float64
     5   genres           146144 non-null  object 
    dtypes: float64(1), int64(1), object(4)
    memory usage: 6.7+ MB
    

There are no more missing values in the data.

Checking for duplicates in movie basics table


```python
movie_basics.duplicated().sum()
```




    0



No duplicates.

We completed cleaning the movie basics table.

### 2. movie_ratings


```python
mr_query = """
SELECT *
FROM movie_ratings
"""

movie_ratings = pd.read_sql(mr_query, conn)
movie_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt10356526</td>
      <td>8</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt10384606</td>
      <td>9</td>
      <td>559</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt1042974</td>
      <td>6</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt1043726</td>
      <td>4</td>
      <td>50352</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt1060240</td>
      <td>6</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_ratings.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 73856 entries, 0 to 73855
    Data columns (total 3 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   movie_id       73856 non-null  object 
     1   averagerating  73856 non-null  float64
     2   numvotes       73856 non-null  int64  
    dtypes: float64(1), int64(1), object(1)
    memory usage: 1.7+ MB
    

no missing values in the ratings table

Checking for duplicates


```python
movie_ratings.duplicated().sum()
```




    0



no duplicates in the movie ratings table

### 3. Genre ratings


```python
movie_basics.genres.value_counts()
```




    genres
    Documentary                   32185
    Drama                         21486
    Comedy                         9177
    missing                        5408
    Horror                         4372
                                  ...  
    Adventure,Music,Mystery           1
    Documentary,Horror,Romance        1
    Sport,Thriller                    1
    Comedy,Sport,Western              1
    Adventure,History,War             1
    Name: count, Length: 1086, dtype: int64



Some genres are combined in one entry separated by a comma. We create a new df and separate each genre and ensure each has its own row.

we start by joining movie ratings and movie basics.


```python
merged_ratings = movie_ratings.merge(movie_basics, on='movie_id', how='inner')
```


```python
# create df as copy of ratings df
genre_df = merged_ratings.copy()

# split the genres
genre_df['genres'] = genre_df.genres.str.split(',')

# one genre in each row
genre_df = genre_df.explode('genres')

genre_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt10356526</td>
      <td>8</td>
      <td>31</td>
      <td>Laiye Je Yaarian</td>
      <td>Laiye Je Yaarian</td>
      <td>2019</td>
      <td>117</td>
      <td>Romance</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt10384606</td>
      <td>9</td>
      <td>559</td>
      <td>Borderless</td>
      <td>Borderless</td>
      <td>2019</td>
      <td>87</td>
      <td>Documentary</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt1042974</td>
      <td>6</td>
      <td>20</td>
      <td>Just Inès</td>
      <td>Just Inès</td>
      <td>2010</td>
      <td>90</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt1043726</td>
      <td>4</td>
      <td>50352</td>
      <td>The Legend of Hercules</td>
      <td>The Legend of Hercules</td>
      <td>2014</td>
      <td>99</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt1043726</td>
      <td>4</td>
      <td>50352</td>
      <td>The Legend of Hercules</td>
      <td>The Legend of Hercules</td>
      <td>2014</td>
      <td>99</td>
      <td>Adventure</td>
    </tr>
  </tbody>
</table>
</div>




```python
genre_df.genres.value_counts().index
```




    Index(['Drama', 'Documentary', 'Comedy', 'Thriller', 'Horror', 'Action',
           'Romance', 'Crime', 'Adventure', 'Biography', 'Family', 'Mystery',
           'History', 'Sci-Fi', 'Fantasy', 'Music', 'Animation', 'Sport', 'War',
           'missing', 'Musical', 'News', 'Western', 'Reality-TV', 'Adult',
           'Game-Show', 'Short'],
          dtype='object', name='genres')



### 4. Directors


```python
dir_query = """
SELECT  *
FROM 
    directors d
JOIN
    persons p
USING(person_id)
"""

directors = pd.read_sql(dir_query, conn)
directors.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 291171 entries, 0 to 291170
    Data columns (total 6 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   movie_id            291171 non-null  object 
     1   person_id           291171 non-null  object 
     2   primary_name        291171 non-null  object 
     3   birth_year          68608 non-null   float64
     4   death_year          1738 non-null    float64
     5   primary_profession  290187 non-null  object 
    dtypes: float64(2), object(4)
    memory usage: 13.3+ MB
    

missing values in the birth_year, death_year and primary proffession columns.

For birth_year and death_year we fill missing value with 0 to represent missing.


```python
# handle missing birth year
directors['birth_year'].fillna(0, inplace=True)

# handle missing death year
directors['death_year'].fillna(0, inplace=True)

directors.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 291171 entries, 0 to 291170
    Data columns (total 6 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   movie_id            291171 non-null  object 
     1   person_id           291171 non-null  object 
     2   primary_name        291171 non-null  object 
     3   birth_year          291171 non-null  float64
     4   death_year          291171 non-null  float64
     5   primary_profession  290187 non-null  object 
    dtypes: float64(2), object(4)
    memory usage: 13.3+ MB
    

Only primary profession has missing values. For this we fill with the tag **'director'**, since it is contained in the directors table.


```python
directors['primary_profession'].fillna('director', inplace=True)

directors.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 291171 entries, 0 to 291170
    Data columns (total 6 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   movie_id            291171 non-null  object 
     1   person_id           291171 non-null  object 
     2   primary_name        291171 non-null  object 
     3   birth_year          291171 non-null  float64
     4   death_year          291171 non-null  float64
     5   primary_profession  291171 non-null  object 
    dtypes: float64(2), object(4)
    memory usage: 13.3+ MB
    

Now we check for duplicates


```python
directors.duplicated().sum()
```




    127638



The  table contains many duplicates


```python
directors.drop_duplicates(inplace=True)

directors.duplicated().sum()
```




    0



### 5. Writers


```python
wr_query = """
SELECT *
FROM
    writers w
JOIN
    persons p
USING(person_id)
"""

writers = pd.read_sql(wr_query, conn)
writers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>birth_year</th>
      <th>death_year</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0285252</td>
      <td>nm0899854</td>
      <td>Tony Vitale</td>
      <td>1,964</td>
      <td>NaN</td>
      <td>producer,director,writer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0438973</td>
      <td>nm0175726</td>
      <td>Steve Conrad</td>
      <td>1,968</td>
      <td>NaN</td>
      <td>writer,producer,director</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0438973</td>
      <td>nm1802864</td>
      <td>Sean Sorensen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>producer,writer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0462036</td>
      <td>nm1940585</td>
      <td>Bill Haley</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0835418</td>
      <td>nm0310087</td>
      <td>Peter Gaulke</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>writer,actor,director</td>
    </tr>
  </tbody>
</table>
</div>




```python
writers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 255871 entries, 0 to 255870
    Data columns (total 6 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   movie_id            255871 non-null  object 
     1   person_id           255871 non-null  object 
     2   primary_name        255871 non-null  object 
     3   birth_year          52917 non-null   float64
     4   death_year          4078 non-null    float64
     5   primary_profession  255029 non-null  object 
    dtypes: float64(2), object(4)
    memory usage: 11.7+ MB
    

Same columns are missing in the writers table just as in directors table. We use the same method to handle the missing values but, for this table, we use the tag 'writer' for primary proffession.


```python
# Filling the birth column
writers['birth_year'].fillna(0, inplace=True)

# filling the death column
writers['death_year'].fillna(0, inplace=True)

# filling the primary proffession column
writers['primary_profession'].fillna('writer', inplace=True)

writers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 255871 entries, 0 to 255870
    Data columns (total 6 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   movie_id            255871 non-null  object 
     1   person_id           255871 non-null  object 
     2   primary_name        255871 non-null  object 
     3   birth_year          255871 non-null  float64
     4   death_year          255871 non-null  float64
     5   primary_profession  255871 non-null  object 
    dtypes: float64(2), object(4)
    memory usage: 11.7+ MB
    

Lets check for duplicates


```python
# check duplicates
writers.duplicated().sum()
```




    77521



We drop the duplicates from the writers table


```python
writers.drop_duplicates(inplace=True)

# check duplicates
writers.duplicated().sum()
```




    0



# Data Analysis


```python
# suppressing scientific notation and adding commas for thousands separators
pd.options.display.float_format = '{:,.0f}'.format
```

This is to improve the readability of numerical data by suppressing scientific notation and add commas as thousands separators.


## 1. Best income generating studios

We use the Box Office Mojo data


```python
bom_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>studio</th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>year</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story 3</td>
      <td>BV</td>
      <td>415,000,000</td>
      <td>652,000,000</td>
      <td>2010</td>
      <td>1,067,000,000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice in Wonderland (2010)</td>
      <td>BV</td>
      <td>334,200,000</td>
      <td>691,300,000</td>
      <td>2010</td>
      <td>1,025,500,000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Deathly Hallows Part 1</td>
      <td>WB</td>
      <td>296,000,000</td>
      <td>664,300,000</td>
      <td>2010</td>
      <td>960,300,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Inception</td>
      <td>WB</td>
      <td>292,600,000</td>
      <td>535,700,000</td>
      <td>2010</td>
      <td>828,300,000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shrek Forever After</td>
      <td>P/DW</td>
      <td>238,700,000</td>
      <td>513,900,000</td>
      <td>2010</td>
      <td>752,600,000</td>
    </tr>
  </tbody>
</table>
</div>



### (a). summary statistics


```python
bom_summary = bom_df.loc[
    :,
    ['domestic_gross', 'foreign_gross', 'worldwide_gross']
].describe()
bom_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3,387</td>
      <td>3,387</td>
      <td>3,387</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28,745,845</td>
      <td>70,151,173</td>
      <td>98,897,018</td>
    </tr>
    <tr>
      <th>std</th>
      <td>66,704,973</td>
      <td>107,498,910</td>
      <td>162,851,053</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100</td>
      <td>600</td>
      <td>4,900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>122,500</td>
      <td>8,000,000</td>
      <td>18,700,000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1,400,000</td>
      <td>70,151,173</td>
      <td>70,185,873</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>28,745,845</td>
      <td>70,151,173</td>
      <td>73,251,173</td>
    </tr>
    <tr>
      <th>max</th>
      <td>936,700,000</td>
      <td>960,500,000</td>
      <td>1,518,900,000</td>
    </tr>
  </tbody>
</table>
</div>



The mean earnings for the movies are around:
- $\$$ 28M for Domestic gross
- $\$$ 70M for Foreign gross
- $\$$ 99M for worldwide gross

The earnings for the movies range from:
- $\$$ 100 to $\$$ 936M for Domestic gross 
- $\$$ 600 to $\$$ 960M for Foreign gross 
- $\$$ 4k  to $\$$ 1B for worldwide gross 

### (b). Distribution of earnings


```python
# creating the figure and axes
fig, axes = plt.subplots(ncols=3, figsize=(12, 5))

# set the style
sns.set_style('darkgrid')

# plot distribution for domestic gross
sns.boxplot(
    data=bom_df,
    x='domestic_gross',
    ax=axes[0]
)



# plot distribution for foreign gross
sns.boxplot(
    data=bom_df,
    x='foreign_gross',
    ax=axes[1]
)



# plot distribution for worldwide gross
sns.boxplot(
    data=bom_df,
    x='worldwide_gross',
    ax=axes[2]
)

# setting scale to log
axes[0].set_xscale('log')
axes[1].set_xscale('log')
axes[2].set_xscale('log')


# Labelling
axes[0].set_title('Distribution of Domestic Gross')
axes[1].set_title('Distribution of Foreign Gross')
axes[2].set_title('Distribution of Worldwide Gross');
```


    
![png](index_files/index_132_0.png)
    


we notice some outliers in the three categories. Mostly above 100M for all the groups.

Using cbook from matplotlib, we can get the exact values.


```python
from matplotlib import cbook
```


```python
# Domestic gross stats
domestic_stats = cbook.boxplot_stats(bom_df.worldwide_gross)

# Foreign gross stats
foreign_stats = cbook.boxplot_stats(bom_df.foreign_gross)

# Worldwide gross stats
worldwide_stats = cbook.boxplot_stats(bom_df.worldwide_gross)
```

We focus on the worldwide gross since its the overal earnings.


```python
# worldwide gross stats without fliers
print('\nWorldwide Gross Stats:')
for key in worldwide_stats[0]:
    if key != 'fliers':
        print(key, worldwide_stats[0][key])
```

    
    Worldwide Gross Stats:
    mean 98897018.2935503
    iqr 54551173.22656608
    cilo 68714251.64081404
    cihi 71657494.81231812
    whishi 155000000.0
    whislo 4900.0
    q1 18700000.0
    med 70185873.22656608
    q3 73251173.22656608
    

From this stats we get more information and a precise range compared to the summary statistics.

- The range of worldwide earnings is between 4900 and 155M. Values outside this range are considered outliers.
- we also get the quartiles and confidence intervals

### (c). Studios with highest Earnings


```python
len(bom_df.studio.value_counts())
```




    258



There are 257 Studios asociated with the movies. We get the studios with the highest earnings (worldwide gross)


```python
# Get total earnings for each studio
total_studio_earnings = bom_df.groupby(
    'studio'
)['worldwide_gross'].sum()

# sorting the studios according to earnings
total_studio_earnings.sort_values(
    ascending=False,
    inplace=True
)

# get top 10 studios according to earnings
top_10_studios = total_studio_earnings[:10].reset_index()
top_10_studios
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>studio</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BV</td>
      <td>44,213,702,543</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WB</td>
      <td>31,412,801,305</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fox</td>
      <td>31,020,580,426</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Uni.</td>
      <td>29,967,617,711</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sony</td>
      <td>22,714,388,634</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Par.</td>
      <td>19,924,826,363</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WB (NL)</td>
      <td>10,334,796,287</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LGF</td>
      <td>9,251,733,541</td>
    </tr>
    <tr>
      <th>8</th>
      <td>IFC</td>
      <td>6,693,197,233</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Magn.</td>
      <td>5,972,688,551</td>
    </tr>
  </tbody>
</table>
</div>



Some of the top 10 earning studios include:
- BV studios
- Warner Bros studios
- Fox Studios
- universal studios
- Sony
- Paramount 
- Warner Bros. (New Line)
- Lionsgate Films (LGF)
- Independent Film Channel (IFC)
- Magnolia Pictures

**Plot of the Top 10 Studio Earnings**

All the top 10 studios have earnings of more than a billion. We first fix the scale to display in billions.


```python
# bar plot of studio earnings
barplot = sns.barplot(
    data=top_10_studios,
    x='studio',
    y='worldwide_gross'
)
# labelling
barplot.set_title('Top 10 Studio Earnings')
barplot.set_xlabel('Studio')
barplot.set_ylabel('Worldwide Gross Earnings')

# format y-axis to show in billions
def billions(x, pos):
    return '%1.1fB' % (x * 1e-9)

formatter = FuncFormatter(billions)
barplot.yaxis.set_major_formatter(formatter);
```


    
![png](index_files/index_147_0.png)
    


### (d). Distribution of earnings over the years


```python
# Creating grouped df of earnings by the years
yearly_earnings = bom_df.groupby(
    'year'
)['worldwide_gross'].sum().reset_index()
```


```python
# Plotting the disribution over the years
ax = sns.lineplot(data=yearly_earnings, x='year', y='worldwide_gross')

# labelling
ax.set_title('Distribution of earnings over the years')
ax.set_xlabel('Year')
ax.set_ylabel('Worldwide Gross Earnings')

# convert axes to billions
ax.yaxis.set_major_formatter(formatter)
```

    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](index_files/index_150_1.png)
    


Over time, there has been a general upward trend in movie earnings. The biggest income was reported in 2016, and since then, it has decreased. The overall trend is still positive despite this recent decrease, suggesting that movie revenues are expected to rise in the long run.

## 2. Best Ratings


```python
# enabling decimals
pd.options.display.float_format = None
```

We use the IMDB data to get some insights based on ratings.
- Best rated genres
- Best rated Writers
- Best rated Directors

### (a). Best rated genres

We use the genres df

We start by grouping the data by the genre and getting the average rating


```python
# grouping data by genre
genre_ratings = genre_df.groupby('genres')[['averagerating', 'numvotes']].mean()

# round rating to 1 decimal point
genre_ratings['averagerating'] = genre_ratings['averagerating'].round(1)

# converting numvotes to integer
genre_ratings['numvotes'] = genre_ratings['numvotes'].astype(int)

# sorting the values
genre_ratings.sort_values(
    by=['averagerating', 'numvotes'],
    ascending=False,
    inplace=True
)

# reset index
genre_ratings = genre_ratings.reset_index()

# get least rated genre
least_rated = genre_ratings.loc[
    genre_ratings.averagerating == genre_ratings.averagerating.min(),
    'genres'
]

# get best rated genre
best_rated = genre_ratings.loc[
    genre_ratings.averagerating == genre_ratings.averagerating.max(),
    'genres'
]

print('Best Rated:', best_rated.values[0])
print('Least Rated:', least_rated.values[0])

genre_ratings
```

    Best Rated: Short
    Least Rated: Adult
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Short</td>
      <td>8.8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Game-Show</td>
      <td>7.3</td>
      <td>1734</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Documentary</td>
      <td>7.3</td>
      <td>266</td>
    </tr>
    <tr>
      <th>3</th>
      <td>News</td>
      <td>7.3</td>
      <td>212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Biography</td>
      <td>7.2</td>
      <td>5673</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Music</td>
      <td>7.1</td>
      <td>2771</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sport</td>
      <td>7.0</td>
      <td>3185</td>
    </tr>
    <tr>
      <th>7</th>
      <td>History</td>
      <td>7.0</td>
      <td>2776</td>
    </tr>
    <tr>
      <th>8</th>
      <td>War</td>
      <td>6.6</td>
      <td>3147</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Musical</td>
      <td>6.5</td>
      <td>1925</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Reality-TV</td>
      <td>6.5</td>
      <td>27</td>
    </tr>
    <tr>
      <th>11</th>
      <td>missing</td>
      <td>6.5</td>
      <td>24</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Drama</td>
      <td>6.4</td>
      <td>3883</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Family</td>
      <td>6.4</td>
      <td>2531</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Adventure</td>
      <td>6.2</td>
      <td>22067</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Animation</td>
      <td>6.2</td>
      <td>8808</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Crime</td>
      <td>6.1</td>
      <td>8594</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Romance</td>
      <td>6.1</td>
      <td>4084</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Comedy</td>
      <td>6.0</td>
      <td>4297</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Fantasy</td>
      <td>5.9</td>
      <td>12387</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Western</td>
      <td>5.9</td>
      <td>8758</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Mystery</td>
      <td>5.9</td>
      <td>8113</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Action</td>
      <td>5.8</td>
      <td>14476</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Thriller</td>
      <td>5.6</td>
      <td>5860</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Sci-Fi</td>
      <td>5.5</td>
      <td>19474</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Horror</td>
      <td>5.0</td>
      <td>3112</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Adult</td>
      <td>3.8</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
</div>



Short films are the best rated with Adult films being the least favourite.

**Visualizing the top 10 genres**


```python
# creating figure and axis
fig, ax = plt.subplots()

# plotting the bar plot
sns.barplot(
    data=genre_ratings[:10],
    y='genres',
    x='averagerating',
    ax=ax,
    orient='h'
)
i = 0
votes = genre_ratings[:10].numvotes
# Labelling the number of votes and including the ratings
for p in ax.patches:
    # labelling the ratings
    ax.annotate(
        f'{p.get_width():.2f}',
        (p.get_width() + .25, p.get_y() + p.get_height()),
        ha='center', va='center',
        xytext=(0, 9),
        textcoords='offset points'
    )
    # labelling the number of votes
    ax.annotate(
        f'{votes[i]} votes',
        (p.get_width() / 2., p.get_y() + p.get_height()),
        ha='center', va='center',
        xytext=(0, 9),
        textcoords='offset points'
    )
    i += 1
    
# labelling the title
ax.set_title('Top 10 rated Genres');
```


    
![png](index_files/index_161_0.png)
    


The top ten genres are shown in the bar graph according to user ratings. The genres are represented by the x-axis, while the ratings are displayed on the y-axis. The quantity of votes a genre received helps to further classify genres with similar ratings, resulting in a more accurate ranking. The height of the bars reflects the average rating for each genre, and each bar is color-coded to help differentiate between them.

### (b). Best rated Directors

The best directors are those whose movies are highly rated.

We first merge the ratings and directors tables


```python
director_ratings = directors.merge(
    movie_ratings,
    on='movie_id',
    how='inner'
)
director_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>birth_year</th>
      <th>death_year</th>
      <th>primary_profession</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0285252</td>
      <td>nm0899854</td>
      <td>Tony Vitale</td>
      <td>1964.0</td>
      <td>0.0</td>
      <td>producer,director,writer</td>
      <td>3.9</td>
      <td>219</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0462036</td>
      <td>nm1940585</td>
      <td>Bill Haley</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>director,writer,producer</td>
      <td>5.5</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0835418</td>
      <td>nm0151540</td>
      <td>Jay Chandrasekhar</td>
      <td>1968.0</td>
      <td>0.0</td>
      <td>director,actor,writer</td>
      <td>5.0</td>
      <td>8147</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0878654</td>
      <td>nm0089502</td>
      <td>Albert Pyun</td>
      <td>1954.0</td>
      <td>0.0</td>
      <td>director,writer,producer</td>
      <td>5.8</td>
      <td>875</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0878654</td>
      <td>nm2291498</td>
      <td>Joe Baile</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>producer,director,camera_department</td>
      <td>5.8</td>
      <td>875</td>
    </tr>
  </tbody>
</table>
</div>



From the data, some directors are deceased. We filter the data to include only directors who are alive.


```python
director_ratings = director_ratings.loc[
    director_ratings.death_year == 0
]

director_ratings.death_year.value_counts()
```




    death_year
    0.0    85331
    Name: count, dtype: int64



We first get the number of movies each director has featured in.


```python
director_movie_count = director_ratings.groupby(
    ['person_id']
).size().sort_values(ascending=False)

# resetting the index and naming count column
director_movie_count = director_movie_count.reset_index(name='moviecount')

print('Highest movie count:', director_movie_count.moviecount.iloc[0])
print('Lowest movie count:', director_movie_count.moviecount.iloc[-1])

director_movie_count.head()
```

    Highest movie count: 39
    Lowest movie count: 1
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>moviecount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm5954636</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm2551464</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm3583561</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm4341114</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm2563700</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



The data includes directors with varying levels of experience, ranging from those who have directed only one film to those with over 200 movies. To categorize their experience, we create a new column with the following classifications:

- Beginner: 1-3 movies
- Intermediate: 4-5 movies
- Experienced: 6-15 movies
- Highly Experienced: 16-20 movies
- Veteran: 20+ movies


```python
# function to categorize the experience
def set_experience(val):
    if val <= 3:
        return 'beginner'
    elif val > 3 and val <= 5:
        return 'intermediate'
    elif val > 5 and val <= 15:
        return 'experienced'
    elif val > 15 and val <=20:
        return 'highly experienced'
    else:
        return 'veteran'
```

**Creating the experience column**


```python
director_movie_count['experience'] = director_movie_count.moviecount.map(
    lambda x: set_experience(x)
)
director_movie_count.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[155], line 1
    ----> 1 director_movie_count['experience'] = director_movie_count.moviecount.map(
          2     lambda x: set_experience(x)
          3 )
          4 director_movie_count.head()
    

    NameError: name 'director_movie_count' is not defined


Next, we group the data by directors and calculate the average ratings of their movies as well as the average number of votes. This allows us to analyze the performance of each director based on the reception and popularity of their films.


```python
# grouping by the directiors
ratings_by_directors = director_ratings.groupby(
    ['person_id', 'primary_name']
)[
    ['averagerating', 'numvotes']
].mean()

# reseting the index
ratings_by_directors.reset_index(inplace=True)


ratings_by_directors.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm0000095</td>
      <td>Woody Allen</td>
      <td>6.700000</td>
      <td>106068.375000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm0000108</td>
      <td>Luc Besson</td>
      <td>6.350000</td>
      <td>113490.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm0000110</td>
      <td>Kenneth Branagh</td>
      <td>6.928571</td>
      <td>160110.714286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0000118</td>
      <td>John Carpenter</td>
      <td>5.600000</td>
      <td>38287.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0000123</td>
      <td>George Clooney</td>
      <td>6.266667</td>
      <td>118783.000000</td>
    </tr>
  </tbody>
</table>
</div>



The average number of votes and the average rating of each director's film are then determined by grouping the data by director. This makes it possible for us to evaluate each director's work in light of the reviews and box office success of their respective projects. Next, we transform the average number of votes to integer numbers and round the average ratings to one decimal place.


```python
# rounding the ratings column
ratings_by_directors.averagerating = ratings_by_directors.averagerating.round(1)

# convert votes to integers
ratings_by_directors.numvotes = ratings_by_directors.numvotes.astype(int)

# sorting
ratings_by_directors.sort_values(
    by=['averagerating', 'numvotes'],
    ascending=False,
    inplace=True
)

ratings_by_directors

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31643</th>
      <td>nm3388005</td>
      <td>Stephen Peek</td>
      <td>10.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>51014</th>
      <td>nm7223265</td>
      <td>Loreto Di Cesare</td>
      <td>10.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>52109</th>
      <td>nm7633303</td>
      <td>Lindsay Thompson</td>
      <td>10.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>50148</th>
      <td>nm6925060</td>
      <td>Tristan David Luciotti</td>
      <td>10.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>54795</th>
      <td>nm8791543</td>
      <td>Emre Oran</td>
      <td>10.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>54832</th>
      <td>nm8809512</td>
      <td>Erik Alarik</td>
      <td>1.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>23645</th>
      <td>nm2277264</td>
      <td>Koki Ebata</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>40737</th>
      <td>nm4728793</td>
      <td>Takeo Urakami</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>28377</th>
      <td>nm2947112</td>
      <td>Shinju Funabiki</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>44133</th>
      <td>nm5328929</td>
      <td>Samuele Dalò</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>56784 rows × 4 columns</p>
</div>



It is noted that the average rating for the best directors is 10, whereas the lowest have an average rating of 1. We also look at the total number of films that each director has starred in, since this has a big impact on the director's grade.


```python
# merging the ratings to include movies count
ratings_by_directors = ratings_by_directors.merge(
    director_movie_count,
    on='person_id',
    how='inner'
)
 
ratings_by_directors.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 56784 entries, 0 to 56783
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   person_id      56784 non-null  object 
     1   primary_name   56784 non-null  object 
     2   averagerating  56784 non-null  float64
     3   numvotes       56784 non-null  int32  
     4   moviecount     56784 non-null  int64  
     5   experience     56784 non-null  object 
    dtypes: float64(1), int32(1), int64(1), object(3)
    memory usage: 2.4+ MB
    


```python
# change data type of moviecount column
ratings_by_directors.moviecount = ratings_by_directors.moviecount.astype('Int32')
# include sorting by movie count
ratings_by_directors.sort_values(
    by=['averagerating', 'moviecount', 'numvotes'],
    ascending=False,
    inplace=True
)

ratings_by_directors
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>moviecount</th>
      <th>experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm3388005</td>
      <td>Stephen Peek</td>
      <td>10.0</td>
      <td>20</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm7223265</td>
      <td>Loreto Di Cesare</td>
      <td>10.0</td>
      <td>8</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm7633303</td>
      <td>Lindsay Thompson</td>
      <td>10.0</td>
      <td>7</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm6925060</td>
      <td>Tristan David Luciotti</td>
      <td>10.0</td>
      <td>6</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm8791543</td>
      <td>Emre Oran</td>
      <td>10.0</td>
      <td>6</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56779</th>
      <td>nm8809512</td>
      <td>Erik Alarik</td>
      <td>1.0</td>
      <td>8</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>56780</th>
      <td>nm2277264</td>
      <td>Koki Ebata</td>
      <td>1.0</td>
      <td>7</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>56781</th>
      <td>nm4728793</td>
      <td>Takeo Urakami</td>
      <td>1.0</td>
      <td>7</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>56782</th>
      <td>nm2947112</td>
      <td>Shinju Funabiki</td>
      <td>1.0</td>
      <td>6</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>56783</th>
      <td>nm5328929</td>
      <td>Samuele Dalò</td>
      <td>1.0</td>
      <td>5</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
  </tbody>
</table>
<p>56784 rows × 6 columns</p>
</div>



We can now group the directors by experience and compare them.


```python
# top 5 beginner ditectors
top_5_beginner_directors = ratings_by_directors.loc[
    ratings_by_directors.experience == 'beginner'
][:5]

# top 5 intermediate ditectors
top_5_intermediate_directors = ratings_by_directors.loc[
    ratings_by_directors.experience == 'intermediate'
][:5]

# top 5 experienced ditectors
top_5_experienced_directors = ratings_by_directors.loc[
    ratings_by_directors.experience == 'experienced'
][:5]

# top 5 highly experienced ditectors
top_5_highly_directors = ratings_by_directors.loc[
    ratings_by_directors.experience == 'highly experienced'
][:5]

# top 5 veteran ditectors
top_5_veteran_directors= ratings_by_directors.loc[
    ratings_by_directors.experience == 'veteran'
][:5]
```


```python
# create figure and axes
fig, axes = plt.subplots(nrows=5, figsize=(7, 25))

# list of all dataframes
data_list = [
    top_5_beginner_directors,
    top_5_intermediate_directors,
    top_5_experienced_directors,
    top_5_highly_directors,
    top_5_veteran_directors
]

# plotting the data
for i, data in enumerate(data_list):
    sns.barplot(
        data=data,
        x='averagerating',
        y='primary_name',
        ax=axes[i]
    )
    # including the number of votes and movie count
    n = 0
    for p in axes[i].patches:
        # labelling the rating
        axes[i].annotate(
            f'{p.get_width():.2f}',
            (p.get_width() + .25, p.get_y() + p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points'
        )
        # labelling the number of votes
        axes[i].annotate(
            f'{data.moviecount.iloc[n]} movie(s), {data.numvotes.iloc[n]} votes',
            (p.get_width() / 2., p.get_y() + p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points'
        )
        n += 1
    
    # labeling axes
    axes[i].set_title(f'Top 5 {data.experience.iloc[i]} directors')
    axes[i].set_xlabel('Average Rating')
    axes[i].set_ylabel('Director Name')
    axes[i].set_xlim(0, 10)
```


    
![png](index_files/index_183_0.png)
    


Above is the list of graphs of the top directors based on their movie ratings, also categorized by their experience levels.

Various factors can vary based on the directors' experience, such as:
- salary
- quality of movies
- audience reception
- production budgets

Beginner directors might have lower salaries and fewer resources, but as their experience grows, they tend to produce higher quality films, receive better audience ratings, and command higher salaries. Veteran directors, with extensive experience, often have established reputations, allowing them to secure larger budgets and attract top talent, further enhancing the quality and success of their movies.

Interestingly, beginners sometimes tend to have higher ratings. This can be attributed to the effect of having directed only a few movies, which may result in their ratings being skewed by a small sample size. A single highly-rated movie can disproportionately elevate their average rating. As directors gain more experience and their body of work grows, their ratings might normalize and provide a more comprehensive view of their overall performance.

These graphs provide a comprehensive view of how directors' experience levels correlate with their average movie ratings, showcasing the impact of experience on their career achievements and industry recognition.

### (c). Best Rated writers

Just like the directors we use the same method to get the top rated writers.

#### i. Merge writers and ratings table


```python
# merging writers and movie ratings
writer_ratings = writers.merge(
    movie_ratings,
    on='movie_id',
    how='inner'
)
writer_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>birth_year</th>
      <th>death_year</th>
      <th>primary_profession</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0285252</td>
      <td>nm0899854</td>
      <td>Tony Vitale</td>
      <td>1964.0</td>
      <td>0.0</td>
      <td>producer,director,writer</td>
      <td>3.9</td>
      <td>219</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0462036</td>
      <td>nm1940585</td>
      <td>Bill Haley</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>director,writer,producer</td>
      <td>5.5</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0835418</td>
      <td>nm0310087</td>
      <td>Peter Gaulke</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>writer,actor,director</td>
      <td>5.0</td>
      <td>8147</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0835418</td>
      <td>nm0841532</td>
      <td>Gerry Swallow</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>writer,actor,miscellaneous</td>
      <td>5.0</td>
      <td>8147</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0878654</td>
      <td>nm0284943</td>
      <td>Randall Fontana</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>writer,director,actor</td>
      <td>5.8</td>
      <td>875</td>
    </tr>
  </tbody>
</table>
</div>



#### ii. Remove dead writers


```python
writer_ratings = writer_ratings.loc[
    writer_ratings.death_year == 0
]

# check the death year collumn
writer_ratings.death_year.value_counts()
```




    death_year
    0.0    109319
    Name: count, dtype: int64



#### iii. Get movies count for each writer

Getting the number of movies each writer has written.


```python
writer_movie_count = writer_ratings.groupby(
    ['person_id']
).size().sort_values(ascending=False)

# resetting the index and naming count column
writer_movie_count = writer_movie_count.reset_index(name='moviecount')

# Creating the experience column
writer_movie_count['experience'] = writer_movie_count.moviecount.map(
    lambda x: set_experience(x)
)

print('Highest movie count:', writer_movie_count.moviecount.iloc[0])
print('Lowest movie count:', writer_movie_count.moviecount.iloc[-1])

writer_movie_count.head()
```

    Highest movie count: 40
    Lowest movie count: 1
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>moviecount</th>
      <th>experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm5954636</td>
      <td>40</td>
      <td>veteran</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm3057599</td>
      <td>32</td>
      <td>veteran</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm3583561</td>
      <td>32</td>
      <td>veteran</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0893128</td>
      <td>32</td>
      <td>veteran</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0598531</td>
      <td>32</td>
      <td>veteran</td>
    </tr>
  </tbody>
</table>
</div>



#### iv. Getting average ratings of each writer


```python
# grouping by the writers
ratings_by_writers = writer_ratings.groupby(
    ['person_id', 'primary_name']
)[
    ['averagerating', 'numvotes']
].mean()

# reseting the index
ratings_by_writers.reset_index(inplace=True)


ratings_by_writers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm0000092</td>
      <td>John Cleese</td>
      <td>7.450000</td>
      <td>89365.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm0000095</td>
      <td>Woody Allen</td>
      <td>6.700000</td>
      <td>106068.375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm0000101</td>
      <td>Dan Aykroyd</td>
      <td>5.200000</td>
      <td>186788.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0000108</td>
      <td>Luc Besson</td>
      <td>5.905556</td>
      <td>87079.500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0000116</td>
      <td>James Cameron</td>
      <td>6.950000</td>
      <td>161411.000</td>
    </tr>
  </tbody>
</table>
</div>



we then round the ratings to one decimal point and convert the votes to integer values.


```python
# rounding the ratings column
ratings_by_writers.averagerating = ratings_by_writers.averagerating.round(
    1)

# convert votes to integers
ratings_by_writers.numvotes = ratings_by_writers.numvotes.astype(int)

# sorting
ratings_by_writers.sort_values(
    by=['averagerating', 'numvotes'],
    ascending=False,
    inplace=True
)

ratings_by_writers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63710</th>
      <td>nm6680574</td>
      <td>Brian Baucum</td>
      <td>10.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>66116</th>
      <td>nm7223265</td>
      <td>Loreto Di Cesare</td>
      <td>10.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>67762</th>
      <td>nm7633303</td>
      <td>Lindsay Thompson</td>
      <td>10.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>71714</th>
      <td>nm8791543</td>
      <td>Emre Oran</td>
      <td>10.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15236</th>
      <td>nm10616933</td>
      <td>Ivana Diniz</td>
      <td>10.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71763</th>
      <td>nm8809512</td>
      <td>Erik Alarik</td>
      <td>1.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>34829</th>
      <td>nm2947112</td>
      <td>Shinju Funabiki</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>60173</th>
      <td>nm6008960</td>
      <td>Eva Toulová</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>74437</th>
      <td>nm9854007</td>
      <td>Giueppe di Giorgio</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>74438</th>
      <td>nm9854008</td>
      <td>Roberto Attolini</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>74705 rows × 4 columns</p>
</div>



We then include the movie count and experience by merging with the movie count df.


```python
# merging the ratings to include movies count
ratings_by_writers = ratings_by_writers.merge(
    writer_movie_count,
    on='person_id',
    how='inner'
)

ratings_by_writers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 74705 entries, 0 to 74704
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   person_id      74705 non-null  object 
     1   primary_name   74705 non-null  object 
     2   averagerating  74705 non-null  float64
     3   numvotes       74705 non-null  int32  
     4   moviecount     74705 non-null  int64  
     5   experience     74705 non-null  object 
    dtypes: float64(1), int32(1), int64(1), object(3)
    memory usage: 3.1+ MB
    

We then sort the records according to the rating, then movie count and finally the number of votes.


```python
# change data type of moviecount column
ratings_by_writers.moviecount = ratings_by_writers.moviecount.astype(
    'Int32')
# include sorting by movie count
ratings_by_writers.sort_values(
    by=['averagerating', 'moviecount', 'numvotes'],
    ascending=False,
    inplace=True
)

ratings_by_writers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>moviecount</th>
      <th>experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm6680574</td>
      <td>Brian Baucum</td>
      <td>10.0</td>
      <td>8</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm7223265</td>
      <td>Loreto Di Cesare</td>
      <td>10.0</td>
      <td>8</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm7633303</td>
      <td>Lindsay Thompson</td>
      <td>10.0</td>
      <td>7</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm8791543</td>
      <td>Emre Oran</td>
      <td>10.0</td>
      <td>6</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm10616933</td>
      <td>Ivana Diniz</td>
      <td>10.0</td>
      <td>5</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>74700</th>
      <td>nm8809512</td>
      <td>Erik Alarik</td>
      <td>1.0</td>
      <td>8</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>74701</th>
      <td>nm2947112</td>
      <td>Shinju Funabiki</td>
      <td>1.0</td>
      <td>6</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>74702</th>
      <td>nm6008960</td>
      <td>Eva Toulová</td>
      <td>1.0</td>
      <td>5</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>74703</th>
      <td>nm9854007</td>
      <td>Giueppe di Giorgio</td>
      <td>1.0</td>
      <td>5</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
    <tr>
      <th>74704</th>
      <td>nm9854008</td>
      <td>Roberto Attolini</td>
      <td>1.0</td>
      <td>5</td>
      <td>1</td>
      <td>beginner</td>
    </tr>
  </tbody>
</table>
<p>74705 rows × 6 columns</p>
</div>



#### v. Grouping according to experience


```python
# top 5 beginner ditectors
top_5_beginner_writers = ratings_by_writers.loc[
    ratings_by_writers.experience == 'beginner'
][:5]

# top 5 intermediate ditectors
top_5_intermediate_writers = ratings_by_writers.loc[
    ratings_by_writers.experience == 'intermediate'
][:5]

# top 5 experienced ditectors
top_5_experienced_writers = ratings_by_writers.loc[
    ratings_by_writers.experience == 'experienced'
][:5]

# top 5 highly experienced ditectors
top_5_highly_writers = ratings_by_writers.loc[
    ratings_by_writers.experience == 'highly experienced'
][:5]

# top 5 veteran ditectors
top_5_veteran_writers = ratings_by_writers.loc[
    ratings_by_writers.experience == 'veteran'
][:5]
```

#### vi. Plotting the Top Writers


```python
# create figure and axes
fig, axes = plt.subplots(nrows=5, figsize=(7, 25))

# list of all dataframes
data_list = [
    top_5_beginner_writers,
    top_5_intermediate_writers,
    top_5_experienced_writers,
    top_5_highly_writers,
    top_5_veteran_writers
]

# plotting the data
for i, data in enumerate(data_list):
    sns.barplot(
        data=data,
        x='averagerating',
        y='primary_name',
        ax=axes[i]
    )
    # including the number of votes and movie count
    n = 0
    for p in axes[i].patches:
        # labelling the rating
        axes[i].annotate(
            f'{p.get_width():.2f}',
            (p.get_width() + .25, p.get_y() + p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points'
        )
        # labelling the number of votes
        axes[i].annotate(
            f'{data.moviecount.iloc[n]} movie(s), {data.numvotes.iloc[n]} votes',
            (p.get_width() / 2., p.get_y() + p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points'
        )
        n += 1

    # labeling axes
    axes[i].set_title(f'Top 5 {data.experience.iloc[i]} writers')
    axes[i].set_xlabel('Average Rating')
    axes[i].set_ylabel('Writer Name')
    axes[i].set_xlim(0, 10)
```


    
![png](index_files/index_205_0.png)
    


graphs showing the top writers according to their movie ratings, further divided into categories based on their experience levels. There are a number of variables that can change depending on the writers' experience, including: 
- salary;
- the calibre of their scripts;
 - the reception from audiences; 
- production budgets.
 Newer writers may have less resources and a lower salary, but as their experience grows, they tend to write better scripts, get higher ratings from audiences, and command higher salaries. Experienced writers, who have a lot of work under their belts, frequently have established reputations that enable them to secure larger budgets and draw in top talent, further improving the calibre and success of the films they do.
# Conclusion
## Recommendations for New Movie Studio

## 1. Acquiring talent
- The company has the ability to work with exceptionally skilled writers and directors connected to highly acclaimed movies. The best writers and directors for each experience level have been determined by the analysis. It's crucial to weigh the trade-offs, though, as seasoned experts could have higher expenses despite their potential for success and recognition. Less seasoned workers, on the other hand, might provide new insights and a more affordable option, but they also carry a greater chance of failure.
## 2. Gaining knowledge from renowned studios.
-It's critical to imitate the business models of successful studios, like BV Studios, Warner Bros., and Fox Studios. Examine their methods for finding talent, choosing genres, marketing, and distribution in order to implement the best strategies that have continuously led to their success.



