
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
