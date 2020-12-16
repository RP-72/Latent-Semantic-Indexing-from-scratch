import pandas as pd
import math
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from string import punctuation
import nltk
pi = math.pi

news_group="sci.med"
docs=60 #number of documents which we are going to compare
categories = [news_group]
# print(list(twenty_train.target_names))
twenty_train = fetch_20newsgroups(subset='train',
        categories=categories,
        #remove=('headers', 'footers', 'quotes'),
        shuffle=True)

def transpose(A):
    '''
    Parameter: 
        A - 2D matrix having dimension m*n
    Process:
        Calculates transpose of given matrix
    Output:
        ans - 2D matrix having dimension n*m
    '''
    ans = np.zeros((len(A[0]),len(A)))
    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[j][i] = A[i][j]
    return ans

stopwords = nltk.corpus.stopwords.words('english')
def rem_punct(s):
    '''
    Parameter: 
        s - string
    Process:
        First we initialize an empty output string str1.

        The function iterates through s and if the character is not a punctation, 
        it is added to the outpur string str1.
    Output:
        str1 - string
    '''
    str1 = ''
    for char in s:
        if(char not in punctuation):
            str1 = str1 + char
    return str1

def rem_nums(s):
    '''
    Parameter: 
        s - string
    Process:
        First we initialize an empty output string str1.

        The function iterates through s and if the character is not a number, 
        it is added to the outpur string str1.
    Output:
        str1 - string
    '''
    str1 = ''
    for num in s:
        if(not num.isdigit()):
            str1 = str1 + num
    return str1

all_docs=[nltk.tokenize.wordpunct_tokenize(rem_nums(rem_punct(twenty_train.data[i]).lower())) for i in range(docs)]

bow=[] #bag of words
for j in range(docs):
    temp = []
    for i in all_docs[j]: #all_docs[j] is the jth document's list of words
        if i not in stopwords and len(i)>0 and i!=None:
            temp.append(i)
    bow.append(temp)

def unique(bow):
    '''
    Parameter: 
        bow - 2D list
    Process:
        a is initialized as first list of bow
        
        The function iterates through all other lists and sets
        a = union(a,cur_list)
        Here cur_list is the iterator
    Output:
        a - set of all unique words in bow
    '''
    a = bow[0]
    for i in range(1,len(bow)):
        a = set(a).union(set(bow[i]))
    return a

wordset = unique(bow)
worddict = [dict.fromkeys(wordset,0) for i in range(len(bow))]

def term_document_matrix():
    '''
    Parameter: 
        Nothing
    Process:
        Iterates through bow and worddict at the same time 

        bow - 2D list  (iterator is bow_i)
        worddict = list of dictionaries    (iterator is worddict_i)

        increments value of each word found in bow_i
    Output:
        pandas dataframe
    '''
    for bow_i,worddict_i in zip(bow,worddict): #zip takes first row of bow and key of worddict
        for word in bow_i:
            worddict_i[word]+=1 #increments value of each word when found in the document
        
    return pd.DataFrame(worddict)
#The above function returns a document term matrix.
#from this we get number of times a unique word is found in each document respectively.

docterm = term_document_matrix()

def term_freq(worddict,bow):
    '''
    Parameter: 
        worddict - dictionary
        bow - list of strings 
    Process:
        initialize empty dictionary called tfdict
        bowcount = number of elements(words) in bow

        traverse through key value pairs of worddict
        Here,
            word - key
            count - value
        Divide count of a word by total number of elements in that document and store it as a 
        key value pair in tfdict
    Output:
        tfdict - dictionary
    '''
    #here worddict is a single dictionary. NOT A LIST OF DICTIONARIES
    #bow is a single list. NOT A NESTED LIST
    tfdict = {} #tfdict -- term frequency dict
    bowcount = len(bow) #bowcount = total number of words in the document 
    for word,count in worddict.items(): 
        tfdict[word] = count/float(bowcount)
    return tfdict

tfbow = []
for i,j in zip(worddict,bow): #worddict is a list of dictionaries
    tfbow.append(term_freq(i,j))

#tfbow is a list of dictionaries. ith dictionary in tfbow is the tfdict of ith document 

def idf(doclist):
    '''
    Parameters:
        doclist - list of dictionaries 
    Process:
        returns a dictionary containing key value pairs of words and number of documents
        that words occurs in. 
        The process is described in detail below.
    Output:
        idfdict - dictionary 
    '''
    idfdict={}
    n = len(doclist)
    idfdict = dict.fromkeys(doclist[0].keys(),0)
    #initializes idfdict as a dictionary which has same keys as doclist[0] and value of each key is 0
    for doc in doclist:
        for word,val in doc.items():
            if val>0:
                idfdict[word]+=1
    #now idfdict has total occurences of each word in all of the documents
    #note that multiple occurences of a word in one document is considered as 1
    #hence if value of a key is... for eg: {'king':2,....} 
    #this means that the word king has appeared in 2 documents
    for word,val in idfdict.items():
        idfdict[word]=math.log(n/float(val)) 
        #computes log (total num of documents/no. of documents that contain a particular word)n
    return idfdict

idfs = idf(worddict) #stores idf value of all words


def tfidf(tfbow_dict,idfs):
    '''
    Parameter: 
        tfbow_dict - dictionary 
        idfs - dictionary
    Process:
        multiplies term frequency with idf of each term 
    Output: 
        tfidf - dictionary
    '''
    tfidf = {}
    for word,val in tfbow_dict.items():
        tfidf[word]=val*idfs[word] #multiply term freq with idf for each term 
    return tfidf

tfidfl = [] 
for i in tfbow:
    tfidfl.append(tfidf(i,idfs))  
#Above loop iterates through list of dictionaries tfbow. 
#Appends output of tfidf function(a dictionary) to tfidfl
X = pd.DataFrame(tfidfl).T
l_2d = X.T.values.tolist()
l_2d = np.array(l_2d)
l_2d = transpose(l_2d)
L,S,R=np.linalg.svd(l_2d)

def zero_padding(n,U,S,V):
    '''
    Parameter: 
        n - integer
        U - Matrix having dimension m*m
        S - List of singular values
        V - Matrix having dimension n*n
    Process:
        The aim of this function is to use S to make a matrix A which contains n singular 
        values on its diagonal. Rest all values are 0. Dimensions of A are m*n
        Basically, we want to make A such that multiply(U,A,V) is possible
    Output:
        A - Matrix having dimension m*n
    '''
    t = []
    for i in range(0,n):
        t.append(S[i])
    #t is the tuple consisting of first n characters of the tuple S (S is sigma)
    for i in range(len(S)-n):
        t.append(0) #zero_padding t with zeros
    
    #convert the non-selected singular values to 0 and form a diagonal matrix, store as A
    A=[]
    for i in range (0,len(S)):
        temp = []
        for j in range(0,len(S)):
            if(j == i):
                temp.append(float(t[j]))
            else:
                temp.append(float(0))
        A.append(temp)
    newrow = []
    for i in range(0,len(S)):
        newrow.append(0)
    #newrow is a list of zeros having same length as S
    for i in range(len(U)-len(S)): 
        A.append(newrow)
        
    return A
        
def reconstruct(u,s,v,n):
    '''
    Parameter: 
        u - Matrix having dimension m*m
        s - List of singular values
        v - Matrix having dimension n*n
        n - integer
    Process:
        produces matrix A having m*n dimensions.
        Returns the product of multiply(u,A,v)
    Output:
        m*n matrix
    '''
    A=zero_padding(n,u,s,v)
    return np.round((u.dot(A)).dot(v),decimals=3)
    
def frobenius(a,a2): #finds how similar two matrices are
    '''
    Parameter: 
        a - Matrix having dimension m*n
        a2 - Matrix having dimension m*n
    Process:
        computes the frobenius norm of the matrix
    Output:
        an integer (frobenius norm of the two matrices)
    '''
    return (np.sqrt(np.sum((a-a2)**2)))/np.sqrt(np.sum(a**2))

def find_k():
    '''
    Parameter:
        None 
    Process:
        iterates through 1 to number of singular values - 1.
        compares the two matrices l_2d and reconstructed matrix of l_2d

        Note that l_2d is the original matrix (the one passed into SVD)
    Output:
        an integer - the number of singular values required to reconstruct a matrix
                     whose frobenius norm with l_2d is less than 0.38 
    '''
    for i in range(1,len(S)):
        f=frobenius(l_2d,reconstruct(L,S,R,i))
        if f<0.35 :
            return i


def search(q):
    q=rem_punct(q)
    q=q.lower().split(" ")
    terms = X.index
    query=[]
    for i in terms:
        if(i in q):
            query.append(1)
        else:
            query.append(0) 
    query=np.asarray(query)
    if 1 not in query:
        print("Could not find any documents")
    
    else:
        k = find_k()
        reconstructed_A = reconstruct(L,S,R,k) 
        score = query.dot(reconstructed_A)
        sort = {}
        for i in range(len(score)):
            if(score[i]<0):
                score[i]=0
            sort[i+1] = score[i]
        last = {}
        for w in sorted(sort, key=sort.get, reverse=True):
            last[w]=sort[w]
        xAxis=[]
        title="Document-wise weightage of the string: "
        for i in q: 
            title=title+" "+i
        yAxis=[]
        
        for document,sc in last.items():
            print("Document: ",document)
            xAxis.append(document)
            yAxis.append(sc)
        
        plt.bar(xAxis, yAxis, color ='red',  
                width = 0.4) 
        
        plt.xlabel("Document number") 
        plt.ylabel("Relevance-Score") 
        plt.title(title) 
        plt.show() 
while(1):
    print("Please choose an option:\n"
    + "1: Search keywords \n"
    + "2: View Documents \n"
    + "3: Exit")
    opt = int(input())
    if(opt == 1):
        print("Enter keyword you want to search\n")
        s = input()
        print("Loading your search results...")
        search(s)
        print("\n")
    elif (opt == 2):
        print("Enter Document number:\n")
        doc_n = int(input())
        if(doc_n>docs):
            print("There are only ",docs," documents")
            continue
        print(twenty_train.data[doc_n - 1])
        print("\n")
    elif( opt == 3):
        print("THANK YOU")
        exit()
    else:
        print("Enter valid input. ")
