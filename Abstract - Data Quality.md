# Abstract - Data Quality Concepts Methodologies and Techniques by Carlo Batini and Monica Scannapieco


### The polygen model for data manipulation

Basically you have a set of ordered triples:

1. datum drawn from a simple domain
2. set of originating databases denoting the local databases from which te datum originates; and
3. set of intermediate databases in which the data led to the selection of the datum

Primitive operators: project, cartesian product, restrict, union, difference.

Each one used to certain things:
- difference to not enter duplicates, 
- restrict select ones that satisfy certain condition and sent to intermediate db
- select join, defined in termos of restrict operator so theyy also involve intermediate sources
- new operators are included like coaesce, takes two columns and merges them into one column with consistency.

### Data provenance

Definition: Description of the origins of a piece of data and the process by which it arrived it in the database.
