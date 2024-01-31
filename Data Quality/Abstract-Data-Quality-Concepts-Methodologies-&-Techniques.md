
`- [Abstract](#abstract)
- [Abstract](#abstract)
  - [Data Quality](#data-quality)
    - [Concepts, Methodologies and Techniques](#concepts-methodologies-and-techniques)
  - [Data Quality Dimensions](#data-quality-dimensions)
    - [Accuracy](#accuracy)
    - [Completeness](#completeness)
      - [Of relational data](#of-relational-data)
        - [without null values with OWA](#without-null-values-with-owa)
        - [with null values with CWA](#with-null-values-with-cwa)
    - [Time related dimensions: Currency timeliness and Volatility](#time-related-dimensions-currency-timeliness-and-volatility)
      - [Possible metrics:](#possible-metrics)
    - [Consistency](#consistency)
      - [Integrity constraints](#integrity-constraints)
          - [Data edits](#data-edits)
    - [Other Data Quality dimensions](#other-data-quality-dimensions)
    - [Accesibility](#accesibility)
    - [Quality information sources](#quality-information-sources)
    - [Evoluition of information toward networked web based niformation systems.](#evoluition-of-information-toward-networked-web-based-niformation-systems)
    - [Brief description of correctness, minimality, completeness, pertinence, readability and normalization.](#brief-description-of-correctness-minimality-completeness-pertinence-readability-and-normalization)


# Abstract 
## Data Quality
### Concepts, Methodologies and Techniques

## Data Quality Dimensions

Some definitions of data dimensions are independent of the data model used to represent the data.

Accuracy and time -related dimensions are model independent.

### Accuracy

Closeness between value v and v', considered as the correct representation of the real-life phenomenon that v aims to represent.
Kinds of accuracy:
- Syntactic:
  - Closeness of a value v to the elements of the corresponding definition domain D.
  - Not interesed in comparing v with the true value of v'; rather we are interested in checking whether v is any one of the values in D; whatevetr it is. So if v= Jack, even if v'=John, v is considered syntactically correct, as Jack is an admissible value in the domain of persons' names.
  - Mesaured by means of functions, called comparision functions more in chapter 5.
  - min numb of character ins, del, replacement to convert from s to s'. More complex comparision functions exist, similar sounds or character transpositions.
- Semantic:
  - Closenees for the value v to the true value of v'
  - Coincides with the concept of correctness.
  - More complex to calculate than syntactic accuracy.
  - One way for checking it:
    -  looking for the same data in different data sources and finding the correct data by comparisons.
    -  The object identification problem:
       -  Trying to identify whether two tuples refer to the same real-world entity or not; Chapter 5 more info.

Accuracy described above is referred to a single value of a relation attribute.
In practical cases, coarser accuracy definitions and metrics may be applied.
As an example it is possible to calculate the accuracy of an attribute called attribute or column accuracy, fo a relation relation accuracy, or of a whole database, database accuracy.
With more than one attr, can be introduced namely duplication.
For relation and database accuracy, for both syntactic and semantic acccuracy, a ratio is tipically calculated between accurate values and the total number of values.

Different metrics:
- Weak acc error
- Strong acc error
- % of accurate tuples matched w the reference table.
- Degree of syntactic accuracy of the relational instance r, by actually considering the fraction of accurate qi = 0 matched si=0 tuples.

### Completeness


The extend to which data are of sufficient breadth, depth and scope for the task at hand.

Types:
- Schema completeness
  - Not missing concepts and their properties in the schema
- Column completeness
  - missing values for a specific property or column
- Population completeness
  - missing values w respect to a reference population

#### Of relational data

It is important to know why a value is null, is it because it is unknown, ooes not exists at all it may exists buy it is not actually known whether it exists or not.
The closed world assumption **CWA** states that only the values actually present in a relational table r , and no other values represent facts of the real world.
In open world assumption **OWA** we can state neither the truth nor the falsity of facts not repesented in the tuples of r.

##### without null values with OWA

Introduced reference relation; relation containing all the tuples that satisfy the relational schema of r, that represent objects of the real world that constitute the present true extension of the schema.

The completeness of a relation r is measured in a model without null values as the fraction of tuples actually represented in the relation r, namely, its size with respect to the total number of tuples in ref(r):

For example consider the citizens of Rome, overall number of people is six million. We have a company that stores data on Rome's citizens, we have 5.400.000 data points, then C(r) = 0.9.
- value of completeness, capture the presence of null values for some fields of a tuple;
- tuple completeness, to characterize the completeness of a tuple with resepect to the values of all its fields;
- attribute completeness, measure the number of null values of a specific attribute in a relation;
- relation completeness, capture the presence of null values in a whole relation


##### with null values with CWA

### Time related dimensions: Currency timeliness and Volatility

All are between 0-1.

-  Currency
   -  How proplty data are updated.
- Volatility
  - frequency with which data vary in time.
- Timeliness
  - How current data are for the task at hand
  - We may have that that it is useleess because they are late for a specific usage.

#### Possible metrics:

- Currency
  - last update metadata.
- Volatility
  - Length of time that data remain valid.
- Timeliness
  - Data not only are current but also in time for event that correspond to their usage, a currency measurement and a check that data are available before the planned usage time.
- More complex metrics:
  - One that threee dimensions are linked by defining timeliness as a function of currency and volatility.
  - Currency = age + (DeliveryTime - InputTime)
  - where age measures how old the data unit is when recived DeliveryTime is the time the information product is delivered to the customer and InputTime is the time that data unit is obtained.
  - timeliness is defined as;
    - max 0,1 minus currency/volatility.
    - Ranges from 0- to 1 where 0 means bad timeliness...
    - Relevance of currency depends on volatility: data that are highly volatile must be current, while currency is less important for data with low volatility.

### Consistency

Captures the violtion of semantic rules defined over a set of data items.
Reference to relational theory integrity constraints are an instantiation of such semantic rules. In statistics data edits are another example of semantic rules that allow for the checking of consistency.

#### Integrity constraints

Are typically defined on shcemas, they can at the same time be checked on a specific instance of te schema that presently represents the extension fo the database.

Main types:
- Key dependency
  - Evit duplicates
- Inclusion dependency aka referencial constraint
  - foreign key si an example of 
- Functional dependeny
  - Given a relationl instance r, let x and y be two nonempty set of atttributes in r. r stisfies the functional dependney X -> Y, if the follwing holds for every pair of tuples, t1 and t2 in r:
  - if t1.X = t2.X then t1.Y = t2.Y


###### Data edits


In non relational data we can define consistency rules.Is defineda s tehtask of detecting inconsistencies by formulatingruyles that muste be respected by every correct set of answers.
chuch rules are expressed as edits which denote error conditions.

example

marital status = ''marrried'', age = ''5 years old''

Rule is to detect this kind of errors could be the following:

If maritial status is married, age must be not be less than 14.

The act of correcting erroneous fields by restoring correct values is called imputation.
Problem is called *edit-imputation* problem. More info in chapter 4.

### Other Data Quality dimensions


There are so many new data pipelines and we have more information now than ever.
Is it dificult to controll everything.

But one thing is important, and that is interpretability.

To maximize interpretability following type of documentation should be available.

1. schema fo files or dataases
2. integrity constraints
3. metadata for cross domain information
4. certificate describing available mesures of data quality dimensions and shcema dimensions
5. history and provenance of data, where created produced and maintained.
Sychronization between different time series concerns proper integration of data 

### Accesibility

World Wide Web Consortium defined the individual disabilities.
- may be not be able to see, hear, move or process some types of information easility or at all;
- may have difficulty reading or comprenhending text;
- may not have to or be able to use a keyboard or mouse;
- may have a text-only screen, a small screen or a slow internet connection;
- may not speak or understand a natural language fluently.

In order to be more accesible:
- avoid use of color as the only means to express semantics, helping daltopnic people appreciate meaning of data
- clear natural language by providing expansions of acronyms improving readability a frequent use of plain terms;
- design a web site that ensure device independance using features that enable activation of page element via variety of input devices;
- provide context and orientation information to help users understand complex pages or elements.

### Quality information sources

Wang and Strong to know dimensions how trustable is the information source providing the data.

- Believability
  - considers whether a certain source provides data that can be regarded as true, real and creible.
- Reputation
  - How trustable is the information source
- Objectivity
  - impartially of sources in data provisioning.
  
- Reliability
  - Whether a source provides data conveying the right information.

This takes a lot of importance in peer to peer connection, if one peer is giving bad information an error could be introduced. More info about this trust issues in chapter 9.

### Evoluition of information toward networked web based niformation systems.

### Brief description of correctness, minimality, completeness, pertinence, readability and normalization.

