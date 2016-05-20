__author__ = 'danmullaguru'

import pandas as pd
import numpy as np
import re
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn import preprocessing

stop_words = set(stopwords.words('english'))

'''
stop_words = ('in.','fu.')
word_tokens = word_tokenize(example_sent)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
'''

import time
start_time = time.time()

df_train = pd.read_csv("Data/train.csv", encoding="ISO-8859-1")
#print(df_train.head())
train_data_count = len(df_train.index)
#print("Train\n",df_train.shape)

df_test = pd.read_csv("Data/test.csv", encoding="ISO-8859-1")
#print(df_test.head())
#print("Test\n",df_test.shape)

df_prod_desc = pd.read_csv("Data/product_descriptions.csv", encoding="ISO-8859-1")
#print(df_test.head())
#print(df_prod_desc.shape)

df_prod_attribs = pd.read_csv("Data/attributes.csv", encoding="ISO-8859-1")
#print(df_test.head())
#print(df_prod_attribs.shape)

df_prod_attribs_brands = df_prod_attribs[df_prod_attribs['name'] == "MFG Brand Name"]
df_prod_attribs_brands = df_prod_attribs_brands.drop('name', axis=1)
df_prod_attribs_brands.rename(columns={'value': 'brand_name'}, inplace=True)

def concatenateMaterials(x):
    x = ' '.join(str(x))
    return x

#print(df_prod_attribs_brands.head(20))
df_prod_attribs_material = df_prod_attribs[df_prod_attribs['name'] == "Material"]
print(df_prod_attribs_material.columns)
df_prod_attribs_material = df_prod_attribs_material.drop('name', axis=1)
df_prod_attribs_material.rename(columns={'value': 'material'}, inplace=True)
df_prod_attribs_material = df_prod_attribs_material[~ df_prod_attribs_material.duplicated(['product_uid'], take_last=True)]
#print(df_prod_attribs_material[~ df_prod_attribs_material.duplicated(['product_uid'], take_last=True)])
#print(df_prod_attribs_material[df_prod_attribs_material['product_uid']==100245] )



print(df_prod_attribs_material.shape)
print(df_prod_attribs_material.columns)
#df_prod_attribs_material = df_prod_attribs_material.groupby('product_uid')['material'].apply(concatenateMaterials)
#print(df_prod_attribs_material.shape)
#print(df_prod_attribs_material.head())
#print(df_prod_attribs_material.shape)

def collectOtherAttribs(x):
    return pd.Series(dict(other_attribs =''.join(x)))

#collect other attributes
#df_other_attribs = df_prod_attribs.head(100).groupby('product_uid')['value'].apply(lambda x:"%s" % ' '.join(x))
#print(df_other_attribs)

#combine test and train
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#print("TrainPlusTest\n",df_all.shape)




#print(df_all.head())
#print(df_all.tail())
#print(df_all.shape)




#Add new columns
#include productid, as some products are popular
df_all["search_words"] = np.nan #df_all["search_term"]
df_all["search_numbers"] = np.nan #df_all["search_term"]

df_all["rest_of_features"] = np.nan
df_all["sw_in_title_ratio"] = 0
df_all["sw_in_desc_ratio"] = 0
df_all["sw_in_brand_name_ratio"] = 0
df_all["sw_in_material_ratio"] = 0
df_all["sw_in_rest_of_features_ratio"] = 0
df_all["sn_in_title_ratio"] = 0
df_all["sn_in_desc_ratio"] = 0

df_all = pd.merge(df_all, df_prod_desc, on='product_uid', how='left')
print("After merging prod desc :TrainPlusTest\n",df_all.shape)
df_all = pd.merge(df_all, df_prod_attribs_brands, on='product_uid', how='left')
print("After merging prod brand :TrainPlusTest\n",df_all.shape)
df_all = pd.merge(df_all, df_prod_attribs_material, on='product_uid', how='left')
print("After merging prod material :TrainPlusTest\n",df_all.shape)



#print(df_all.head())


#print(df_all.head())

def separateNumbers(str1):
    numbers = re.findall("[\D]?\d+[\.\/\-xX*]?\d*", str1)
    filtered_numbers = [n.strip().replace('*','x') for n in numbers]
    return ' '.join(filtered_numbers)

def separateWords(str1):
    words =  re.findall("\D+", str1)
    filtered_words = [w.strip().replace('*','x') for w in words if (len(w.strip())>1 )]
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #print(filtered_words)
    return ' '.join(filtered_words)

def populateSearch_words_numbers():

    for index, row in df_all.iterrows():
        searchString = row['search_term']
        numbers = re.findall("\d+[\./]?[xX*]?\d*", searchString)
        words =  re.findall("\D+", searchString)
        #print(numbers,"\n")
        #row['search_numbers']= ' '.join(numbers)
        #row['search_words']= ''.join(words)
        df_all.xs(index)['search_numbers'] = ' '.join(str(numbers))
        df_all.xs(index)['search_words'] = ' '.join(words)
    return ()


def find_common_words(str1,str2):
    #str2 = str(str2)
    count = 0.3
    str1_words = str1.split()
    total_match_wrds = 0.1 + len(str1.split()) + len(str2.split())
    #print(str1_words,"\n",str2,"\n")
    for word in str1_words:
        #if (len(word)>1 and word not in stop_words and str2.find(word)>=0):
        if (len(word)>2 and word not in stop_words and re.search(word, str2, re.IGNORECASE)):
            #print(word)
            count+=1
            #print(word)
    return count

def find_common_numbers(str1,str2):
    #str2 = str(str2)
    count = 0.3
    str1_words = str1.split()
    total_matched_wrds = 0.1 + len(str1.split()) + len(str2.split())
    #print(str1_words,"\n",str2,"\n")
    for word in str1_words:
        #if (len(word)>1 and word not in stop_words and str2.find(word)>=0):
        if (len(word)>0 and re.search(word, str2, re.IGNORECASE)):
            #print(word)
            count+=1
            #print(word)
    return count

def find_brand_match_ratio(str1,brand_name):
    #str2 = str(str2)
    count = 0.2
    search_words = str1.split()
    total_matched_wrds = 0.2
    #print(search_words,"\n",brand_name,"\n")
    for word in search_words:
        #if (len(word)>1 and word not in stop_words and str2.find(word)>=0):
        if (len(word)>1 and word not in stop_words and re.search(word, brand_name, re.IGNORECASE)):
            #print(word,"-",brand_name)
            count = 1
            #print(word)
        #if(count>total_matched_wrds):
        #    count = total_matched_wrds
    return count

def find_sw_in_title_ratio(x):
    count = 0.3
    product_title = x['product_title']
    search_words = x['search_words']
    count = find_common_words(search_words,product_title)
    return count

def find_sw_in_desc_ratio(x):
    count = 0.3
    product_desc = x['product_description']
    search_words = x['search_words']
    count = find_common_words(search_words,product_desc)
    return count

def find_sw_in_brand_name_ratio(x):
    count = 0.5
    brand_name = x['brand_name']
    #print(brand_name)
    if(isinstance(brand_name, str)):
        search_words = x['search_words']
        #print("\n-->",search_words,"\n",brand_name)
        count = find_brand_match_ratio(search_words,brand_name)
    #else:
        #print(brand_name)
    return count

def find_sw_in_material_ratio(x):
    count = 0.3
    material = x['material']
    if(isinstance(material, str)):
        search_words = x['search_words']
        count = find_brand_match_ratio(search_words,material)
    return count

def find_sn_in_title_ratio(x):
    count = 0.3
    product_title = x['product_title']
    search_numbers = x['search_numbers']
    count = find_common_numbers(search_numbers,product_title)
    return count

def find_sn_in_desc_ratio(x):
    count = 0.3
    product_desc = x['product_description']
    search_numbers = x['search_numbers']
    count = find_common_numbers(search_numbers,product_desc)
    return count

df_all['search_words'] = df_all['search_term'].map(lambda x:separateWords(x))


df_all['search_numbers'] = df_all['search_term'].map(lambda x:separateNumbers(x))


df_all['sw_in_title_ratio'] = df_all.apply(find_sw_in_title_ratio, axis=1)

df_all['sw_in_desc_ratio'] = df_all.apply(find_sw_in_desc_ratio, axis=1)
df_all['sn_in_title_ratio'] = df_all.apply(find_sn_in_title_ratio, axis=1)
df_all['sn_in_desc_ratio'] = df_all.apply(find_sn_in_desc_ratio, axis=1)
df_all['sw_in_brand_name_ratio'] = df_all.apply(find_sw_in_brand_name_ratio, axis=1)
df_all['sw_in_material_ratio'] = df_all.apply(find_sw_in_material_ratio, axis=1)




#str = "wire fencing 6 ft high"
#print(separateWords(str))
#str = "owens corning 7-3"
#print(separateNumbers(str))

#print(df_all['search_term'].head(100))
#print(df_all['search_words'].head(100))
#print(df_all['search_numbers'].head(100))
#print(df_all['sw_in_title_ratio'].head(100))
#print(df_all['sw_in_desc_ratio'].head(100))

#print(df_all.loc[df_all['sw_in_brand_name_ratio'] > 27])
#Write to pickle

'''
product_uid
sw_in_title_ratio
sw_in_desc_ratio
sw_in_brand_name_ratio
sw_in_material_ratio
sn_in_title_ratio
sn_in_desc_ratio
relevance
'''
column_list = ['id','product_uid','sw_in_title_ratio','sw_in_desc_ratio','sw_in_brand_name_ratio','sw_in_material_ratio','sn_in_title_ratio','sn_in_desc_ratio','relevance']

'''
df_all_relevance = df_all['relevance']
df_all = df_all.ix[:, column_list]
print(df_all.head())
min_max_scaler = preprocessing.MinMaxScaler()
df_all = pd.DataFrame(min_max_scaler.fit_transform(df_all), columns=df_all.columns)
df_all['relevance'] = df_all_relevance
'''

df_train_massaged = df_all.ix[:train_data_count-1, column_list]
#print(df_train_massaged.head())
#print("train shape\n",df_train_massaged.shape)

df_test_massaged = df_all.ix[train_data_count:, column_list]
#print(df_test_massaged.head())
#print("test shape\n",df_train_massaged.shape)

df_train_pickle_object = open("Data/df_train_merged.pickle","wb")
pickle.dump(df_train_massaged,df_train_pickle_object)
df_train_pickle_object.close()
print("Pickled Train data")

df_test_pickle_object = open("Data/df_test_merged.pickle","wb")
pickle.dump(df_test_massaged,df_test_pickle_object)
df_test_pickle_object.close()
print("Pickled Test data")

#read from pickle
