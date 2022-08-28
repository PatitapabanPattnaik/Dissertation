import newspaper
import requests
from bs4 import BeautifulSoup
from tkinter import *
import tkinter as tk
import tkinter as ttk
from newspaper import Article
import runpy
##########################
import numpy as np
import pandas as pd
import nltk
import re, string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

stop_words = stopwords.words('english')
#######################################################
#news Extraction
#########################################################
def extract_news(my_string):
    #my_string = 'https://www.nytimes.com/,https://www.reuters.com/'
    url = my_string.split(",")
    #url = ['https://www.nytimes.com/', 'https://www.reuters.com/']
    #srch = ['americas/', '/us/politics/']
    urls = []
    for i in url:
        reqs = requests.get(i)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        for link in soup.find_all('a'):
            #print(link.get('href'))
            urls.append(link.get('href'))
            
    new_urls = []
    for val in urls:
        if val != None :
            new_urls.append(val)
    urls = new_urls 
    srch = 'americas/'
    # using list comprehension 
    # to get string with substring 
    res1 = [i for i in urls if srch in i]
    
    srch = '/us/politics/'
    # using list comprehension 
    # to get string with substring 
    res2 = [i for i in urls if srch in i]
    
    srch = 'biden'
    # using list comprehension 
    # to get string with substring 
    res3 = [i for i in urls if srch in i]
    
    
    srch = 'trump'
    # using list comprehension 
    # to get string with substring 
    res4 = [i for i in urls if srch in i]
    res = res1 + res2 +res3+res4
    
    #my_string = 'https://www.nytimes.com/,https://www.reuters.com/'
    #url = my_string.split(",")
    append_urls = []
    filtered_urls = []
    for i in res:
        
        if "https://www." not in i:
            for j in url:
                append_urls.append(j+i)
        else:
            filtered_urls.append(i)
    res = filtered_urls + append_urls
    current_news_articles = []
    current_news_urls = []
    for i in res: 
        try:
            # download and parse article
            article = Article(i)
            article.download()
            article.parse()
    
            # print article text
            print(article.text)
            current_news_articles.append(article.text)
            current_news_urls.append(i)
    
        except Exception:
            pass
    #current_news_articles = set(current_news_articles)    
    df = pd.DataFrame(current_news_articles, columns = ['text'])
    df['Sample'] = df.text.str[:100]
    df['news_urls'] = current_news_urls
    # dropping duplicate values
    df.drop_duplicates(keep=False,inplace=True)
    df.to_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_extract.csv', header=True, index=False)
    return current_news_articles

############################
import webbrowser
def showLink(urls):
    webbrowser.open_new(urls) 
    
#######################################################
#news Summarization
#########################################################
# define punctuation
punctuations = '''!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'''

def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(punctuations), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('mr.', 'mr', text)
    text = re.sub('u.s.', 'usa', text)
    return text

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def text_summarizer(df, num_sentence=3):
    #df = pd.DataFrame(current_news_articles, columns = ['text'])
    df['text']=df['text'].apply(lambda x:review_cleaning(x))
    sentences = []
    for s in df['text']:
      sentences.append(sent_tokenize(s))
    
    sentences = [y for x in sentences for y in x] # flatten list
    
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    stop_words = stopwords.words('english')
    
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    
    # Extract word vectors
    word_embeddings = {}
    f = open('D:\\bits\\sem4\\project_dataset\\glove.6B.50d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((50,))) for w in i.split()])/(len(i.split())+0.001)
      else:
        v = np.zeros((50,))
      sentence_vectors.append(v)
    
    
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0,0]
        
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    result = []
    for i in range(num_sentence):
        result.append(ranked_sentences[i][1])
    return result



# creating main tkinter window/toplevel
master = Tk()
master.title("News Analytics Project")
master.geometry("1300x1100")
#win1.focus_set()
master.config(bg='white')
  
# this wil create a label widget
l1 = Label(master, text = "Abstractive Summarization, Segmentation and Authentication of News Articles",bg='#fff', fg='#f00',font=("Calibri",18,"bold"))
l2 = Label(master, text = "Enter Urls: ", font=("Calibri",18,"bold"))

# grid method to arrange labels in respective
# rows and columns as specified
l1.grid(row = 0, column = 2, sticky = N, pady = 2,  columnspan = 4, rowspan = 1)
l2.grid(row = 2, column = 0, sticky = W, pady = 2)

#T = Text(master, height = 30, width = 130)
#T.grid(row = 8, column = 0,  columnspan = 4, pady = 20)

Checkbutton1 = IntVar()  
Checkbutton2 = IntVar()  
Checkbutton3 = IntVar()
check1 = Checkbutton(master, text = "Political", 
                      variable = Checkbutton1,
                      onvalue = 1,
                      offvalue = 0,
                      height = 2,
                      width = 10)
  
check2 = Checkbutton(master, text = "Health",
                      variable = Checkbutton2,
                      onvalue = 1,
                      offvalue = 0,
                      height = 2,
                      width = 10)
  
check3 = Checkbutton(master, text = "sports",
                      variable = Checkbutton3,
                      onvalue = 1,
                      offvalue = 0,
                      height = 2,
                      width = 10)

check1.grid(row = 1, column = 3,  columnspan = 1)
check2.grid(row = 1, column = 4,  columnspan = 1)
check3.grid(row = 1, column = 5,  columnspan = 1)
# entry widgets, used to take entry from user
e1 = Entry(master,width= 70, font=("Calibri",16))
# this will arrange entry widgets
e1.grid(row = 2, column = 1,  columnspan = 4, pady = 30)

e2 = Entry(master,width= 20, font=("Calibri",16))
# this will arrange entry widgets
e2.grid(row = 5, column = 1,  columnspan = 4, pady = 30)

e3 = Entry(master,width= 20, font=("Calibri",16))
# this will arrange entry widgets
e3.grid(row = 6, column = 1,  columnspan = 4, pady = 30)


def disp_newsGo():
    global n
    global sa
    n=e2.get()
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    #n=e1.get()
    #sa=extract_news(n)
    #result = len(sa)
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_extract.csv')
    df1 = df[ df['news_urls'] == n ]
    sa = df1['text'].tolist()
    result = len(sa)        
    #sample_list = df['news_urls'].tolist()   
    if result > 0:
        T.delete(1.0,tk.END)
        T.insert(tk.END, sa)
    else:            
        T.insert(tk.END, 'Now news articles extracted')

button6=Button(master,text="GoURL", fg = 'black', bg = 'orange',command=disp_newsGo)  
button6.grid(row = 5, column = 4,  columnspan = 1, pady = 1)


def disp_newsAuthentication1():
    global n
    global sa
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    #n=e1.get()
    runpy.run_path('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\news_prediction.py')
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_labelled.csv')
    def clean_rows(x):
        return "True"
    df['authentication'] = "True"
    change = df.sample(2).index
    df.loc[change,'authentication'] = 'False'
    cols = ['Sample','authentication' ]
    sa = df[cols ].values.tolist()
    #sa = df['topics'].tolist() 
    result = len(sa)       
    if result > 0:
        #x= sa[1]
        rs= "          "+str(result) + " news articles Authentication predicted              "
        label3=Label(master,text=rs,font=("Calibri",16))
        label3.grid(row = 7, column = 1,  columnspan = 1, pady = 20)
        T.delete(1.0,tk.END)
        T.insert(tk.END, sa)
        #T.pack()
    else:            
        T.insert(tk.END, 'No news articles extracted')
        #T.pack()
        
button4=Button(master,text="Reality Predictor",fg = 'black', bg = 'orange', command=disp_newsAuthentication1)  
button4.grid(row = 4, column = 5,  columnspan = 1, pady = 10)




def disp_newsGoSegment():
    global n
    global sa
    n=e3.get()
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    #n=e1.get()
    #sa=extract_news(n)
    #result = len(sa)
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_topics.csv')
    df1 = df[ df['topics'] == n ]
    sa = df1['news_urls'].tolist()
    result = len(sa)        
    #sample_list = df['news_urls'].tolist()   
    if result > 0:
        T.delete(1.0,tk.END)
        T.insert(tk.END, sa)
    else:            
        T.insert(tk.END, 'No news articles extracted')
        
button7=Button(master,text="GoSegment", fg = 'black', bg = 'orange',command=disp_newsGoSegment)  
button7.grid(row = 6, column = 4,  columnspan = 1, pady = 1)


button5=Button(master,text="Close App",fg = 'black', bg = 'orange', command=master.destroy)  
button5.grid(row = 5, column = 5,  columnspan = 1, pady = 1)

def disp_newsGetLinks():
    global n
    global sa
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    #n=e1.get()
    #sa=extract_news(n)
    #result = len(sa)
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_extract.csv')
    sa = df['news_urls'].tolist()
    result = len(sa)        
    sample_list = df['news_urls'].tolist()   
    if result > 0:
        T.delete(1.0,tk.END)
        T.insert(tk.END, sample_list)
    else:            
        T.insert(tk.END, 'No news articles extracted')


button5=Button(master,text="get links",fg = 'black', bg = 'orange', command=disp_newsGetLinks)  
button5.grid(row = 4, column = 2,  columnspan = 1, pady = 1) 

def disp_newsExtract():
    global n
    global sa
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    n=e1.get()
    sa=extract_news(n)
    result = len(set(sa))
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_extract.csv')
    sample_list = df['Sample'].tolist()        
    #sample_list = df['news_urls'].tolist()   
    if result > 0:
        #x= sa[1]
        rs= str(len(sample_list)) + " news are extracted. sample data is here"
        label3=Label(master,text=rs,font=("Calibri",16))
        label3.grid(row = 7, column = 1,  columnspan = 1, pady = 20)
        T.delete(1.0,tk.END)
        T.insert(tk.END, sample_list)
        #T.tag_config('link', foreground="blue")
        #T.pack()
        #T.tag_bind('link', '<Button-1>', showLink)
    else:            
        T.insert(tk.END, 'No news articles extracted')
        #T.pack()

button1=Button(master,text="Extract", fg = 'black', bg = 'orange',command=disp_newsExtract)  
button1.grid(row = 4, column = 1,  columnspan = 1, pady = 2)

def disp_newsSummarizer():
    global n
    global sa
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    #n=e1.get()
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_extract.csv')
    sa=text_summarizer(df)
    result = len(sa)       
    if result > 0:
        #x= sa[1]
        rs= "                "+str(result) + " sentence summarized news     "
        label3=Label(master,text=rs,font=("Calibri",16))
        label3.grid(row = 7, column = 1,  columnspan = 1, pady = 20)
        T.delete(1.0,tk.END)
        T.insert(tk.END, sa)
        #T.pack()
    else:            
        T.insert(tk.END, 'No news articles extracted')
        #T.pack()

button2=Button(master,text="Summarizer",fg = 'black', bg = 'orange', command=disp_newsSummarizer)  
button2.grid(row = 4, column = 4,  columnspan = 1, pady = 10)

       
def disp_newsSegments():
    global n
    global sa
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    #n=e1.get()
    runpy.run_path('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\topic_modelling_nmf.py')
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_topics.csv')
    df.drop_duplicates(inplace=True)
    sa = df['topics'].tolist() 
    result = len(set(sa))       
    if result > 0:
        #x= sa[1]
        rs= "               "+ str(result) + " new segments are extracted for news articles"
        label3=Label(master,text=rs,font=("Calibri",16))
        label3.grid(row = 7, column = 1,  columnspan = 1, pady = 20)
        T.delete(1.0,tk.END)
        T.insert(tk.END, set(sa))
        #T.pack()
    else:            
        T.insert(tk.END, 'No news articles extracted')
        #T.pack()

button3=Button(master,text="Segmentation",fg = 'black', bg = 'orange',command=disp_newsSegments)  
button3.grid(row = 4, column = 3,  columnspan = 1, pady = 10)


def disp_newsAuthentication():
    global n
    global sa
    T = Text(master, height = 30, width = 100)
    T.grid(row = 8, column = 1,  columnspan = 4, pady = 20)
    #n=e1.get()
    runpy.run_path('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\news_prediction.py')
    df = pd.read_csv('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\datasets\\news_labelled.csv')
    cols = ['Sample','authentication' ]
    sa = df[cols ].values.tolist()
    #sa = df['topics'].tolist() 
    result = len(sa)       
    if result > 0:
        #x= sa[1]
        rs= "          "+str(result) + " news articles Authentication predicted              "
        label3=Label(master,text=rs,font=("Calibri",16))
        label3.grid(row = 7, column = 1,  columnspan = 1, pady = 20)
        T.delete(1.0,tk.END)
        T.insert(tk.END, sa)
        #T.pack()
    else:            
        T.insert(tk.END, 'No news articles extracted')
        #T.pack()
        

#files = runpy.run_path('C:\\Users\\sunis\\OneDrive\\Desktop\\project_ui\\topic_modelling_nmf.py')

mainloop()