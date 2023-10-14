#載入所需的程式套件
import os
import pandas as pd
import jieba
#print(jieba.__version__)
import jieba.analyse
import html
import re
from collections import Counter
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from textacy.preprocessing import replace
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn import preprocessing 
from sklearn.feature_extraction.text import TfidfTransformer

#載入資料
os.chdir("C:\\Users\\Downloads")
WealthNews=pd.read_excel('使用資料.xlsx')
WealthNews.columns
#WealthNews.head()
#WealthNews.dtypes
WealthNews.info()

#資料探索-----------------------------------------------------------------------
#1.處理Source
WealthNews["Source"] = WealthNews["Source"].str.replace("NowNews","NOWnews") # 直接改正
#新增NewSource，將"財訊xxxx"都換成"財訊"
WealthNews['NewSource']=WealthNews['Source']
#pandas.DataFrame.loc[condition, column_label] = new_value
#condition：這個引數返回使條件為真的值。
#column_label：該引數用於指定要更新的目標列。WealthNews['NewSource']=WealthNews['Source']
WealthNews.loc[WealthNews['NewSource'].str.startswith('財訊'),'NewSource']='財訊'

#為了讓圓餅圖顯示得比較漂亮，將篇數為1的Source編列為其他
WealthNews['NewSource'] = WealthNews['NewSource'].str.replace("健康醫療網","其他")
WealthNews['NewSource'] = WealthNews['NewSource'].str.replace("廣告企劃部","其他")
WealthNews['NewSource'] = WealthNews['NewSource'].str.replace("愛姆斯的醫材藥品法規世界","其他")
WealthNews['NewSource'] = WealthNews['NewSource'].str.replace("科技新報","其他")
WealthNews['NewSource'] = WealthNews['NewSource'].str.replace("英國《金融時報》精選","其他")
WealthNews['NewSource'] = WealthNews['NewSource'].str.replace("鍶科技","其他")
WealthNews.groupby('NewSource').size()
 #解決圖表中文亂碼
from pylab import mpl
mpl.rcParams['font.sans-serif']=['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus']=False
#NewSource圓餅圖
#WealthNews['NewSource'].value_counts().plot(kind='pie')
explode=[0.08, 0, 0, 0, 0, 0, 0, 0, 0, 0]  
plotdata=list(WealthNews['NewSource'].value_counts())
df1 = pd.DataFrame(WealthNews['NewSource'].value_counts().reset_index())
plt.pie(data=df1,x='NewSource', labels='index',autopct='%.0f%%',textprops = {"fontsize" : 11}, explode=explode,labeldistance=1.58,pctdistance=1.25)
plt.show()

#2.計算length統計量
WealthNews['length'] = WealthNews['Context'].str.len()
length_stat = pd.DataFrame() 
length_stat['N']=WealthNews.groupby('NewSource')['length'].count()
length_stat['length_mean']=WealthNews.groupby('NewSource')['length'].mean()
length_stat['length_sd']=WealthNews.groupby('NewSource')['length'].std()
length_stat['length_min']=WealthNews.groupby('NewSource')['length'].min()
length_stat['length_median']=WealthNews.groupby('NewSource')['length'].median()
length_stat['length_max']=WealthNews.groupby('NewSource')['length'].max()

length_stat = length_stat.sort_values('N', ascending=False)
length_stat
#變異大是因為中英夾雜，英文一個字母算一次

#前5多的來源中，它們的文章的箱型圖
WealthNews['NewSource'].value_counts()
w = WealthNews['NewSource'].isin(['財訊','Genet','NOWnews','Money DJ','財經新報'])
g = sns.catplot(data=WealthNews[w], x="NewSource", y="length", kind='box',height=6)

#3.以年月當X軸(eg. 202001,202002...202006)去看y(生醫有幾篇報導) 折線圖 
import datetime
WealthNews['year_month']=WealthNews['Date'].dt.strftime('%Y-%m')
#WealthNews['year_month']=pd.to_datetime(WealthNews['year_month'])
  
report = pd.DataFrame() 
report['N']=WealthNews.groupby('year_month')['ID'].count()
report=report.reset_index()
report

plt.figure(figsize=(10,5),dpi=100,linewidth = 2)
plt.plot(report['year_month'],report['N'],'o-',color = 'g',label="Count")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("年-月", fontsize="10")
plt.ylabel("刊登生醫類文章數", fontsize="10")
plt.xticks(range(30), ['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
                        '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
                        '2022-01','2022-02','2022-03','2022-04','2022-05','2022-06'], rotation=90)
plt.yticks([2,4,6,8,10,12,14,16,18,20])
plt.title("202001-202206生醫相關文章數")
plt.legend()
plt.show()

#4.將Contex 文字取代成一致
WealthNews['Context'] = WealthNews['Context'].str.replace("(","（")
WealthNews['Context'] = WealthNews['Context'].str.replace(")","）")
WealthNews['Context'] = WealthNews['Context'].str.replace("Covid","新冠肺炎")
WealthNews['Context'] = WealthNews['Context'].str.replace("COVID","新冠肺炎")
WealthNews['Context'] = WealthNews['Context'].str.replace("19","nineteen")
WealthNews['Context'] = WealthNews['Context'].str.replace("1期","一期")
WealthNews['Context'] = WealthNews['Context'].str.replace("2期","二期")
WealthNews['Context'] = WealthNews['Context'].str.replace("3期","三期")
WealthNews['Context'] = WealthNews['Context'].str.replace("Moderna","莫德納")

#Part I: Identify the Noise----------------------------------------------------
RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\《》「」]')    #[&#<>{}\[\]\\]老師寫的
#定義資料不純度
def impurity(text, min_len=10):
    """returns the share of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

#計算每篇text 的不純度
WealthNews['impurity'] = WealthNews['Context'].apply(impurity, min_len=10)
WealthNews[['Context', 'impurity']].sort_values(by='impurity', ascending=False).head(10)

#Part II: Remove noise with regular expression---------------------------------
def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text) #in this example, this part does nothing
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r' ', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    #去除 （英文）字，()內的當做重複原譯前面的字，補充說明的英文取代呈空白
    text = re.sub(r'（+[\w* ]+）',' ', text)   #r'（+[\w*]+）|（+[\d*]+）'
    #刪除<,>,《,》
    text = re.sub(r'[<,>,.,%,《,》/]','', text)
    
    #直接去除數字 
    text = re.sub(r'\d','', text)   
    text = re.sub(r'[()0-9-\s]','', text)
    
    #刪除英文字
    #text = re.sub(r'[a-zA-Z,\. ]','', text)
    return text.strip()

WealthNews['Clean_Context']=WealthNews['Context'].apply(clean)
#再次看清除noise後的impurity
WealthNews['impurity2'] = WealthNews['Clean_Context'].apply(impurity, min_len=10) 
WealthNews[['Clean_Context', 'impurity2']].sort_values(by='impurity2', ascending=False).head(3)  

#Part III: Character Normalization with textacy (中文不用)----------------------
#Part IV: Character Masking with textacy---------------------------------------
from textacy.preprocessing import replace
WealthNews['Clean_Context']=WealthNews['Clean_Context'].apply(replace.urls)

#Liguistic Processing----------------------------------------------------------
jieba.set_dictionary('./dict.txt.big.txt') #繁體中文用字
jieba.load_userdict("./userdict_medical.txt") #自訂字典
jieba.analyse.set_stop_words("./stopWords.txt") #停用字 

#去除停用字函數
#用自己整理的stop words
stopwords1 = [line.strip() for line in open('./stopWords.txt', 'r', encoding='utf-8').readlines()]
def remove_stop(text):
    c1=[]
    for w in text:
        if w not in stopwords1:
            c1.append(w)
    c2=[i for i in c1 if i.strip() != '']
    return c2

#斷句斷詞
WealthNews['tokens']=WealthNews['Clean_Context'].apply(jieba.cut)     
#去除停用字
WealthNews['tokens_new']=WealthNews['tokens'].apply(remove_stop)    

from collections import Counter
counter = Counter()
WealthNews['tokens_new'].map(counter.update)
print(counter.most_common(15))

min_freq=5 #(老師設2)
freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
freq_df = freq_df.query('freq >= @min_freq')
freq_df.index.name = 'token'
freq_df = freq_df.sort_values('freq', ascending=False)
freq_df.head(10)

#Token Word的長條圖
import seaborn as sns
sns.set(font="SimSun")
ax = freq_df.head(15).plot(kind='barh', width=0.95, figsize=(8,3))
ax.invert_yaxis()
ax.set(xlabel='Frequency', ylabel='Token', title='Top Words')

#Token Word的文字雲
from matplotlib import pyplot as plt
from wordcloud import WordCloud 
wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(freq_df['freq'])
plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)

#相似度------------------------------------------------------------------------
def list_to_string(org_list, seperator=' '):
    return seperator.join(org_list)
WealthNews['News_seg']=WealthNews['tokens_new'].apply(list_to_string)

cv = CountVectorizer(decode_error='ignore', min_df=2) 
dt01 = cv.fit_transform(WealthNews['News_seg'])
print(cv.get_feature_names())
fn=cv.get_feature_names()
#稀疏矩陣
dtmatrix=pd.DataFrame(dt01.toarray(), columns=fn)  
dtmatrix

cosine_similarity(dt01[1], dt01[3])
#sm 相似度矩陣 186x186
#計算相似度
sm = pd.DataFrame(cosine_similarity(dt01, dt01)) 

#Tfidf-------------------------------------------------------------------------
#from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

tfidf_dt = tfidf.fit_transform(dt01)
tfidfmatrix = pd.DataFrame(tfidf_dt.toarray(), columns=cv.get_feature_names())
#計算某2則報導的相似度
cosine_similarity(tfidf_dt[72], tfidf_dt[27]) 

#匯出sm1
sm1 =pd.DataFrame(cosine_similarity(tfidf_dt, tfidf_dt))
sm1.to_excel('cosine_similarity_tfidf.xlsx', sheet_name='sheet1', index=False)
#sm1

sm2 = pd.DataFrame(cosine_similarity(tfidf_dt.transpose(), tfidf_dt.transpose()))
sm2

tfidfsum=tfidfmatrix.T.sum(axis=1)
#tfidf的文字雲
wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(tfidfsum)
plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)

#Kmeans------------------------------------------------------------------------
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(preprocessing.normalize(tfidf_dt))
    distortions.append(km.inertia_)

from matplotlib import pyplot as plt
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

km = KMeans(
    n_clusters=7, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(preprocessing.normalize(tfidf_dt))

g0 = WealthNews['Context'][y_km==0]
print("the number of cluster 0:",len(g0))
g0.head()
#疫苗防治政策相關

g1 = WealthNews['Context'][y_km==1]
print("the number of cluster 1:",len(g1))
g1.head()
#生醫產業、研發相關

g2 = WealthNews['Context'][y_km==2]
print("the number of cluster 2:",len(g2))
g2.head()
#各國疫苗研發、Omicron病毒相關

g3 = WealthNews['Context'][y_km==3]
print("the number of cluster 3:",len(g3))
g3.head()
#疫情所影響到的社會事件相關

g4 = WealthNews['Context'][y_km==4]
print("the number of cluster 4:",len(g4))
g4.head()

g5 = WealthNews['Context'][y_km==5]
print("the number of cluster 5:",len(g5))
g5.head()

g6 = WealthNews['Context'][y_km==6]
print("the number of cluster 6:",len(g6))
g6.head()

#額外補充
#觀察Token Words的出現次數and時間點----------------------------------------------
#拿詞頻的freq來用
freq_df.head(20)

import datetime
WealthNews['year_month']=WealthNews['Date'].dt.strftime('%Y-%m')
WealthNews['year_month']=pd.to_datetime(WealthNews['year_month'])

#主觀選取跟covid-19比較有相關的詞
date = pd.DataFrame(WealthNews['year_month'])
date['tokens_new']=WealthNews['tokens_new']
date['count_疫苗']=0
for i in range(0,len(date['tokens_new'])):
    date['count_疫苗'][i]=date['tokens_new'][i].count('疫苗')
date['count_治療']=0
for i in range(0,len(date['tokens_new'])):
    date['count_治療'][i]=date['tokens_new'][i].count('治療')
date['count_研發']=0
for i in range(0,len(date['tokens_new'])):
    date['count_研發'][i]=date['tokens_new'][i].count('研發')
date['count_疫情']=0
for i in range(0,len(date['tokens_new'])):
    date['count_疫情'][i]=date['tokens_new'][i].count('疫情')
date['count_臨床試驗']=0
for i in range(0,len(date['tokens_new'])):
    date['count_臨床試驗'][i]=date['tokens_new'][i].count('臨床試驗')
date['count_病毒']=0
for i in range(0,len(date['tokens_new'])):
    date['count_病毒'][i]=date['tokens_new'][i].count('病毒')

group_fin=date.groupby('year_month').agg('sum').reset_index()

# import matplotlib相關套件
import matplotlib.pyplot as plt
# import字型管理套件
from matplotlib.font_manager import FontProperties
month=group_fin['year_month']

plt.figure(figsize=(25,10),dpi=100,linewidth = 2)
plt.plot(group_fin['year_month'],group_fin['count_疫苗'],color = 'r', label="疫苗")
plt.plot(group_fin['year_month'],group_fin['count_治療'],color = 'g', label="治療")
plt.plot(group_fin['year_month'],group_fin['count_研發'],color = 'b', label="研發")
plt.plot(group_fin['year_month'],group_fin['count_疫情'],color = 'm', label="疫情")
plt.plot(group_fin['year_month'],group_fin['count_臨床試驗'],color = 'y', label="臨床試驗")
plt.plot(group_fin['year_month'],group_fin['count_臨床試驗'],color = 'k', label="病毒")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Covid-19相關字詞出現時間點觀察折線圖", fontsize="25")
plt.xlabel("month", fontsize="20")
plt.ylabel("count", fontsize="20")
plt.legend()
plt.show()
#可以看出，在2020/04，covid-19相關疫苗剛出產時，"疫苗"有一波討論
#到了2021/05，台灣疫情大爆發，"疫苗"的討論度又暴增了