__author__ = 'danmullaguru'
import sqlite3
import pandas as pd
#read using pandas and convert to sqlite
c = sqlite3.connect(":memory:")
df_pd = pd.read_csv("Data/product_descriptions.csv").fillna(" ")
df_pd.to_sql("pd",c)
df_pd=""
df_at = pd.read_csv("Data/attributes.csv").fillna(" ")
df_at.to_sql("at",c)
df_at=""
df_tr = pd.read_csv("Data/train.csv", sep=",", encoding="ISO-8859-1").fillna(" ")
df_tr.to_sql("tr",c)
df_tr=""
df_te = pd.read_csv("Data/test.csv", sep=",", encoding="ISO-8859-1").fillna(" ")
df_te.to_sql("te",c)
df_te=""
#df_ss = pd.read_csv("../input/sample_submission.csv").fillna(" ")
print("Attribute --> Brands: ")
print(pd.read_sql("SELECT [value], count(*) as c FROM at WHERE [name]='MFG Brand Name' GROUP BY [value] ORDER BY c DESC LIMIT 10;", c).head(10))
print("-----------------------------------")
print("Top 10 Search Terms from Train : ")
print(pd.read_sql("SELECT search_term, count(*) as c FROM tr GROUP BY search_term ORDER BY c DESC LIMIT 10;", c).head(10))
print("-----------------------------------")
print("Top 10 Search Terms from Test : ")
print(pd.read_sql("SELECT search_term, count(*) as c FROM te GROUP BY search_term ORDER BY c DESC LIMIT 10;", c).head(10))