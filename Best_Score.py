#!/usr/bin/env python
# coding: utf-8

# In[57]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import io


# In[120]:


df_train = pd.read_csv('Trainset.csv', encoding='Latin-1')   
df_train=df_train.drop(columns="id")
df_test = pd.read_csv('Testset without answer.csv', encoding='Latin-1')
reviewsTest = df_test["review"]

labels, texts = [], []
labels = df_train["rating"]
texts = df_train["review"]
from nltk.corpus import stopwords
stopword = set("a about above after again against ain all am an and any are aren arent as at be because been before being below between both but by can couldn couldnt d did didn didnt do does doesn doesnt doing don dont down during each few for from further had hadn hadnt has hasn hasnt have haven havent having he her here hers herself him himself his how i if in into is isn isnt it its its itself just ll m ma me mightn mightnt more most mustn mustnt my myself needn neednt no nor not now o of off on once only or other our ours ourselves out over own re s same shan shant she shes should shouldve shouldn shouldnt so some such t than that thatll the their theirs them themselves then there these they this those through to too under until up ve very was wasn wasnt we were weren werent what when where which while who whom why will with won wont wouldn wouldnt y you youd youll youre youve your yours yourself yourselves could hed hell hes heres hows id ill im ive lets ought shed shell thats theres theyd theyll theyre theyve wed well were weve whats whens wheres whos whys would able abst accordance according accordingly across act actually added adj affected affecting affects afterwards ah almost alone along already also although always among amongst announce another anybody anyhow anymore anyone anything anyway anyways anywhere apparently approximately arent arise around aside ask asking auth available away awfully b back became become becomes becoming beforehand begin beginning beginnings begins behind believe beside besides beyond biol brief briefly c ca came cannot cant cause causes certain certainly co com come comes contain containing contains couldnt date different done downwards due e ed edu effect eg eight eighty either else elsewhere end ending enough especially et etc even ever every everybody everyone everything everywhere ex except f far ff fifth first five fix followed following follows former formerly forth found four furthermore g gave get gets getting give given gives giving go goes gone got gotten h happens hardly hed hence hereafter hereby herein heres hereupon hes hi hid hither home howbeit however hundred id ie im immediate immediately importance important inc indeed index information instead invention inward itd itll j k keep keeps kept kg km know known knows l largely last lately later latter latterly least less lest let lets like liked likely line little ll look looking looks ltd made mainly make makes many may maybe mean means meantime meanwhile merely mg might million miss ml moreover mostly mr mrs much mug must n na name namely nay nd near nearly necessarily necessary need needs neither never nevertheless new next nine ninety nobody non none nonetheless noone normally nos noted nothing nowhere obtain obtained obviously often oh ok okay old omitted one ones onto ord others otherwise outside overall owing p page pages part particular particularly past per perhaps placed please plus poorly possible possibly potentially pp predominantly present previously primarily probably promptly proud provides put q que quickly quite qv r ran rather rd readily really recent recently ref refs regarding regardless regards related relatively research respectively resulted resulting results right run said saw say saying says sec section see seeing seem seemed seeming seems seen self selves sent seven several shall shed shes show showed shown showns shows significant significantly similar similarly since six slightly somebody somehow someone somethan something sometime sometimes somewhat somewhere soon sorry specifically specified specify specifying still stop strongly sub substantially successfully sufficiently suggest sup sure take taken taking tell tends th thank thanks thanx thats thatve thence thereafter thereby thered therefore therein therell thereof therere theres thereto thereupon thereve theyd theyre think thou though thoughh thousand throug throughout thru thus til tip together took toward towards tried tries truly try trying ts twice two u un unfortunately unless unlike unlikely unto upon ups us use used useful usefully usefulness uses using usually v value various ve via viz vol vols vs w want wants wasnt way wed welcome went werent whatever whatll whats whence whenever whereafter whereas whereby wherein wheres whereupon wherever whether whim whither whod whoever whole wholl whomever whos whose widely willing wish within without wont words world wouldnt www x yes yet youd youre z zero as aint allow allows apart appear appreciate appropriate associated best better cmon cs cant changes clearly concerning consequently consider considering corresponding course currently definitely described despite entirely exactly example going greetings hello help hopefully ignored inasmuch indicate indicated indicates inner insofar itd keep keeps novel presumably reasonably second secondly sensible serious seriously sure ts third thorough thoroughly three well wonder ourselves hers between yourself but again there about once during out very having with they own an be some for do its yours such into of most itself other off is s am or who as from him each the themselves until below are we these your his through don nor me were her more himself this down should our their while above both up to ours had she all no when at any before them same and been have in will on does yourselves then that because what over why so can did not now under he you herself has just where too only myself which those i after few whom t being if theirs my against a by doing it how further was here than".split())
#stopword=stopwords.words('english')
s = []
for str in texts: 
    s_list = [word for word in str.split() if word not in stopword]
    str_ = ' '.join(s_list)   
    s.append(str_) 
print(s) 
#print(stopword)
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels.shape


# In[133]:


cnt_vectorizer = CountVectorizer(ngram_range=(1, 2),max_features=20000) 
cnt_vectorizer = cnt_vectorizer.fit(texts)
features = cnt_vectorizer.transform(texts)
features_nd = features.toarray()

features_nd.shape


# In[134]:


nb = MultinomialNB()
nb.fit(features_nd, labels)


# In[135]:


features = cnt_vectorizer.transform(reviewsTest)
features_nd = features.toarray()


features_nd.shape


# In[136]:


nb_pred= nb.predict(features_nd)


# In[137]:


trainDF1 = pd.DataFrame()
trainDF1['id'] = df_test['id']
inv_pd = encoder.inverse_transform(nb_pred)
trainDF1['rating'] = inv_pd


# In[138]:


export_csv = trainDF1.to_csv('submission_nb2.csv', index = None)

