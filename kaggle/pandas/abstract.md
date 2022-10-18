
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





