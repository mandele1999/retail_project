Dataset Title: Online Retail

Source of Dataset: UC Irvine Machine Learning Repository Link: [https://archive.ics.uci.edu/dataset/352/online+retail]
Authors: Daqing Chen, chend@lsbu.ac.uk, School of Engineering, London South Bank University
Category: Business/Sales

Data Info:
This is a transactional data set which contains all the transactions occurring between 01/12/2010 
and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique
all-ocassion gifts. Many customers of the company are wholesalers.

Variables Table

Variable Name	Role	Type	        Description	Units	Missing Values
InvoiceNo	    ID	    Categorical	    a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation		no
StockCode	    ID	    Categorical	    a 5-digit integral number uniquely assigned to each distinct product		no
Description	    Feature	Categorical	    product name		no
Quantity	    Feature	Integer	        the quantities of each product (item) per transaction		no
InvoiceDate	    Feature	Date	        the day and time when each transaction was generated		no
UnitPrice	    Feature	Continuous	    product price per unit	sterling	no
CustomerID	    Feature	Categorical	    a 5-digit integral number uniquely assigned to each customer		no
Country	        Feature	Categorical	    the name of the country where each customer resides		no