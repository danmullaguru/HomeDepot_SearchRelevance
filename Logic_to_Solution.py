__author__ = 'danmullaguru'

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

'''
combineTrainTestData


Search Words

        stop words in. fu. cu.


Search numbers

        add numbers (23.45, 23, 2/3)
        4*3 4X8 separate as 4 8



BrandName

Material

RestOfFeatures

---------
SW in Title Ratio

SW in Desc Ratio

SW in BrandName ratio

SW in Material ratio

SW in RestOfFeatures ratio
--------
SplitTestData

PreapareNewTestDataForFit
--------
Try with

Random Forest
Linear Regression

Calculate performance on CV data
--------
'''

def find_common_words(str1,str2):
    #str2 = str(str2)
    count = 0
    str1_words = str1.split()
    #print(str1_words,"\n",str2,"\n")
    for word in str1_words:
        #print(word,"\n",str2)
        #if (len(word)>1 and word not in stop_words and str2.find(word)>=0):
        if (len(word)>1 and word not in stop_words and re.search(word, str2, re.IGNORECASE)):
            count+=1
            #print("\n",word)
    return count

def separateWords(str1):
    words =  re.findall("\D+", str1)
    filtered_words = [w.strip() for w in words if (len(w.strip())>1 )]
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #print(filtered_words)
    return ' '.join(filtered_words)

search_term = 'Hampton Bay Ceiling Fan with remote'
search_words = separateWords(search_term)
print(search_words,"\n")

prod_title = 'Hampton Bay Ceiling Fan Remote Control'
common_word_count = find_common_words(search_words,prod_title)
print(common_word_count)
