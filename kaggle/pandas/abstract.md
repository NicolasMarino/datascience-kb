
# Pandas



Different ways to create a dataframe


One:

``` python
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
```
| |Bob	|Sue|
-----------|-----------|-----------
|0|	I liked it.	|Pretty good.|
|1|	It was awful.|	Bland.|

Second:

``` python
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

```


| |Bob	|Sue|
-----------|-----------|-----------
|Product A|	I liked it.	|Pretty good.|
|Product B|	It was awful.|	Bland.|

Series:

A Series, by contrast, is a sequence of data values. If a DataFrame is a table, a Series is a list. And in fact you can create one with nothing more than a list:

``` python
pd.Series([1, 2, 3, 4, 5])
```

A Series is, in essence, a single column of a DataFrame. So you can assign row labels to the Series the same way as before, using an index parameter. However, a Series does not have a column name, it only has one overall name:

``` python
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
```


- | -
-----------|-----------|
2015 Sales    |30
2016 Sales    |35
2017 Sales    |40
Name: Product A, dtype: int64

## Reading files

Csv.


``` python
wine_review = pd.read_csv({path-to-file});

#check large of Dataframe
wine_review.shape
##(129971, 14) for {129971 records in 14 columns, something like 2 M entries}

#Get first 5
wine_reviews.head()

#Select index of column to read.
wine_reviews = pd.read_csv({path-to-file}, index_col=0)
wine_reviews.head()
```


# Indexing in pandas

## Index-based selection


Selecting data based on numerical position, <`iLoc`> works on this paradigm.

``` python

#Select first row of data in a DatFrame

reviews.iloc[0]


#Both loc and iloc are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second.
#This means that it's marginally easier to retrieve rows, and marginally harder to get retrieve columns. To get a column with iloc, we can do the following:

#Gets fist column all elements
reviews.iloc[;, 0]

# get first column only first 3 elems starting from 0.
reviews.iloc[:3, 0]
reviews.iloc[1:3, 0]
reviews.iloc[[0, 1, 2], 0]

# get last 5 elements
reviews.iloc[-5:]

#Excersises
#Select the records with index labels `1`, `2`, `3`, `5`, and `8`, assigning the result to the variable `sample_reviews`.
#Solution: sample_reviews = reviews.loc[[1,2,3,5,8]]

```

## Label-based selection

``` python

# first row of country column.
reviews.loc[0, 'country']

# all elements of given column names
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

```


# Choosing between loc and iloc

When choosing or transitioning between loc and iloc, there is one "gotcha" worth keeping in mind, which is that the two methods use slightly different indexing schemes.

iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.

Why the change? Remember that loc can index any stdlib type: strings, for example. If we have a DataFrame with index values Apples, ..., Potatoes, ..., and we want to select "all the alphabetical fruit choices between Apples and Potatoes", then it's a lot more convenient to index df.loc['Apples':'Potatoes'] than it is to index something like df.loc['Apples', 'Potatoet'] (t coming after s in the alphabet).

This is particularly confusing when the DataFrame index is a simple numerical list, e.g. 0,...,1000. In this case df.iloc[0:1000] will return 1000 entries, while df.loc[0:1000] return 1001 of them! To get 1000 elements using loc, you will need to go one lower and ask for df.loc[0:999].

Otherwise, the semantics of using loc are the same as those for iloc.

# Manipulating the index

Label-based selection derives its power from the labels in the index. Critically, the index we use is not immutable. We can manipulate the index in any way we see fit.

The set_index() method can be used to do the job. Here is what happens when we set_index to the title field:

``` python

# It creates an new row with this name.
reviews.set_index("title")

```
This is useful if you can come up with an index for the dataset which is better than the current one.

# Conditional selection

``` python

reviews.loc[reviews.country == 'Italy']

reviews.loc[(reviews.country == 'Italy') (& or | {like known}) (reviews.points >= 90)]

reviews.loc[reviews.country.isin(['Italy', 'France'])]

reviews.loc[reviews.price.notnull()]

```

# Assigning data


``` python

# constant value in each row
reviews['critic'] = 'everyone'

# itrable between values.
reviews['index_backwards'] = range(len(reviews), 0, -1)

```

# Summary Functions and Maps


``` python

# Similar to

reviews.points.describe()
# count    129971.000000
# mean         88.447138
#              ...      
# 75%          91.000000
# max         100.000000
# Name: points, Length: 8, dtype: float64

# For string
reviews.taster_name.describe()

# count         103727
# unique            19
# top       Roger Voss
# freq           25514
# Name: taster_name, dtype: object

# Get unique names
reviews.taster_name.unique()
# Returns an array of strings

# Get counts of every string
reviews.taster_name.value_counts()

```

## Maps

```python
# map() is the first, and slightly simpler one. For example, suppose that we wanted to remean the scores the wines received to 0. We can do this as follows:

review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)

# Faster way

review_points_mean = reviews.points.mean()
reviews.points - review_points_mean

# Pandas will also understand what to do if we perform these operations between Series of equal length. For example, an easy way of combining country and region information in the dataset would be to do the following:

reviews.country + " - " + reviews.region_1
# 0            Italy - Etna
# 1                     NaN
#                ...       
# 129969    France - Alsace
# 129970    France - Alsace
# Length: 129971, dtype: object

# These operators are faster than map() or apply() because they use speed ups built into pandas. All of the standard Python operators (>, <, ==, and so on) work in this manner.

# However, they are not as flexible as map() or apply(), which can do more advanced things, like applying conditional logic, which cannot be done with addition and subtraction alone.


```

### Exercise

```python

# Which wine is the "best bargain"? Create a variable `bargain_wine` with the title of the wine with the highest points-to-price ratio in the dataset.

bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, "title"]

# There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series `descriptor_counts` counting how many times each of these two words appears in the `description` column in the dataset. (For simplicity, let's ignore the capitalized versions of these words.)
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

# We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.

# Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.

# Create a series `star_ratings` with the number of stars corresponding to each review in the dataset.

def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')

```

# Grouping and sorting


## Groupwise analysis


One function we've been using heavily thus far is the value_counts() function. We can replicate what value_counts() does by doing the following:

``` python
reviews.groupby('points').points.count()
#We can use any of the summary functions
reviews.groupby('points').price.min()

# We also can apply lambda functions to each element
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])

# Get the best wine by country and province
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

# agg()
# Another groupby() method worth mentioning is agg(), which lets you run a bunch of different functions on your DataFrame simultaneously.
# For example, we can generate a simple statistical summary of the dataset as follows:

reviews.groupby(['country']).price.agg([len, min, max])


```

## Multi-indexes

``` python
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed

mi = countries_reviewed.index
type(mi)
# pandas.core.indexes.multi.MultiIndex


## MultiIndex / advanced indexing
# https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html

```

## Sorting

``` python
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')

# sort_values() defaults to an ascending sort, where the lowest values go first. However, most of the time we want a descending sort,  where the higher numbers go first. That goes thusly:

countries_reviewed.sort_values(by='len', ascending=False)

# To sort by index values, use the companion method sort_index().
countries_reviewed.sort_index()

# Sort by more than one column at a time
countries_reviewed.sort_values(by=['country', 'len'])

```

## Excercises

``` python
# get series with one index, and count how many elements.
reviews_written = reviews.groupby('taster_twitter_handle').size()

# What is the best wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).
best_rating_per_price = reviews_written = reviews.groupby('price')['points'].max().sort_index()

# What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.

price_extremes = reviews.groupby('variety').price.agg(['min','max'])

# What are the most expensive wine varieties? Create a variable `sorted_varieties` containing a copy of the dataframe from the previous question where varieties are sorted in descending order based on minimum price, then on maximum price (to break ties).
sorted_varieties = reviews.groupby('variety').price.agg(['min','max']).sort_values(by=['min','max'], ascending=False)

# Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()


# What combination of countries and varieties are most common? Create a `Series` whose index is a `MultiIndex`of `{country, variety}` pairs. For example, a pinot noir produced in the US should map to `{"US", "Pinot Noir"}`. Sort the values in the `Series` in descending order based on wine count.

country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)

```

## Data types and missing values


``` python

reviews.price.dtype
dtype('float64')

# get every column dtype.
reviews.dtypes

# points column from its existing int64 data type into a float64 data type
reviews.points.astype('float64')

# A DataFrame or Series index has its own dtype, too:
reviews.index.dtype
dtype('int64')


```

### Missing data



``` python

# Get each row which has null country
reviews[pd.isnull(reviews.country)]

# Fill na with unknow
reviews.region_2.fillna("Unknown")

# Replace: 
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")

```


#### Excercises


``` python

# Sometimes the price column is null. How many reviews in the dataset are missing a price?
n_missing_prices = pd.isnull(reviews.price).sum()


# What are the most common wine-producing regions? Create a Series counting the number of times each value occurs in the `region_1` field. This field is often missing data, so replace missing values with `Unknown`. Sort in descending order.  Your output should look something like this:

reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)

```


# Renaming and Combining

## Renaming

``` python
# Rename column or change index name.
reviews.rename(columns={'points': 'score'})

# Rename specific indexes
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# You'll probably rename columns very often, but rename index values very rarely. For that, set_index() is usually more convenient.

# This creates a new row with index named wines
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

```

## Combining


``` python
# Pandas has three core methods for doing this. In order of increasing complexity, these are concat(), join(), and merge(). Most of what merge() can do can also be done more simply with join(), so we will omit it and focus on the first two functions here.

## concat()
# The simplest combining method is concat(). Given a list of elements, this function will smush those elements together along an axis.

canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])


# join()
# The middlemost combiner in terms of complexity is join(). join() lets you combine different DataFrame objects which have an index in common.

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])


# For example, to pull down videos that happened to be trending on the same day in both Canada and the UK, we could do the following:
left.join(right, lsuffix='_CAN', rsuffix='_UK')
## The lsuffix and rsuffix parameters are necessary here because the data has the same column names in both British and Canadian datasets. If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't need them.
```


### Excercises


``` python

## Set the index name in the dataset to `wines`.
reviews.rename_axis('wines', axis='rows')

## Both tables include references to a `MeetID`, a unique key for each meet (competition) included in the database. Using this, generate a dataset combining the two tables into one.
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")

powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))

```