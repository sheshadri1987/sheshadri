pip install ibm_watson

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 \
import Features, SentimentOptions, EntitiesOptions, KeywordsOptions,ConceptsOptions

pip install nltk

authenticator = IAMAuthenticator('037RCFfB95tyai-d-Jk-EdmU4tu57f0MWc9p5CHUF15j')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)
natural_language_understanding.set_service_url('https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/f2b2d3df-d747-4c68-bff4-5a3439346dfa')

#this gives option to upload your file from local
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io
 
df = pd.read_csv(io.BytesIO(uploaded['Salesforce.csv']))
print(df)
df['Subject']=df['Subject'].fillna(' ')

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
list=df["Subject"]
df['comment'] = df["Subject"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])).str.lower()
df['comment']

def remove_whitespace(text):
    return  " ".join(text.split())

df['comment']=df['comment'].apply(remove_whitespace)

text1=""
for i in range(len(df)):
    text1=text1+" "+df["comment"][i]

l=df['comment']
l

list1=[]
list2=[]

for i in range(0,len(l)):
    k=""
    try:
        sentiment=natural_language_understanding.analyze(text=l[i],features=Features(sentiment=SentimentOptions(targets=[l[i]]),keywords=KeywordsOptions( sentiment=True,
                                     limit=2))).get_result()
        s=json.dumps(sentiment)
        s1=json.loads(s)
        #print(s1)
        s2=s1["sentiment"]["document"]["label"]
        list2=list2+[s2]
        k=""
        for i in range(len(s1["keywords"])):
            k=k+(s1["keywords"][i]["text"])+" ,"
        list1=list1+[k]
        print("list1 :",list1)
    except:
        k=" "
        list1=list1+[k]
        list2=list2+["neutral"]

nltk.download('punkt')

import matplotlib.pyplot as plt
%matplotlib inline

from nltk import word_tokenize
wordlist=word_tokenize(text)
wordlist

filtered_sentence = [w for w in wordlist]
filtered_sentence

pip install wordcloud

from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=60).generate(text)
plt.figure(figsize=(16,12))

'''plot wordcloud in matplotlib'''

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

import gensim
from gensim import corpora

# Creating the term dictionary of our corpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(wordlist)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_sents]

wordlist

Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=100)

# Print the model output
print(ldamodel.print_topics(num_topics=2, num_words=20))

text

