__author__ = 'danmullaguru'

import pandas as pd

from sklearn import preprocessing
'''
train_data = pd.read_csv('Data/train.csv',  encoding='latin-1')
train_data.to_pickle('train_data.pickle')
print(list(train_data.columns.values))
'''
train_data = pd.read_pickle('Data/df_train_merged.pickle')
print(list(train_data.columns.values))
print(train_data.head());
print("---------------------------------------------");


df_all_relevance = train_data['relevance']
min_max_scaler = preprocessing.MinMaxScaler()
train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data), columns=train_data.columns)
train_data['relevance'] = df_all_relevance

print(train_data.head());
print("---------------------------------------------");


prodDesc_data = pd.read_pickle('prodDesc_data.pickle')
print(list(prodDesc_data.columns.values))
print(prodDesc_data.head());
print("---------------------------------------------");

attributes_data = pd.read_pickle('attributes_data.pickle')
print(list(attributes_data.columns.values))
print(attributes_data.head());
print("---------------------------------------------");
print("Unique Attributes:\n",len(pd.unique(attributes_data.name.ravel())))
for name in pd.unique(attributes_data.name.ravel()):
    print (name,"\n")
print("---------------------------------------------");

test_data = pd.read_pickle('test_data.pickle')
#print(list(test_data.columns.values))
#print(test_data.head());
#print("---------------------------------------------");

sampleSubmission_data = pd.read_pickle('sampleSubmission_data.pickle')
#print(list(sampleSubmission_data.columns.values))
#print(sampleSubmission_data.head());
#print("---------------------------------------------");

#print(train_data.loc[train_data['id'] == 1894])

#print(attributes_data.loc[attributes_data['product_uid'] == 100337])

'''
#print(sessions.info())
#print(sessions.describe())
##print column names
print(list(sessions.columns.values))
##print rows columns count
print(sessions.shape)
print(sessions.head(50))
print(sessions[sessions['user_id']=='d1mm9tcy42'])
g = sessions.groupby('user_id')
#print(g.count().head())
print(g.size())

'''