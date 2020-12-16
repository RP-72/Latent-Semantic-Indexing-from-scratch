# Latent Semantic Indexing

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Latent semantic indexing (LSI) or Latent Semantic Analysis(LSA) is an indexing and information retrieval method. It is one of the major analysis approaches in the field of text mining. 

It helps in finding out the documents which are most relative with the specified keyword. Similarly it also helps the search engines to give most appropriate results for the search query. LSI uses "Singular Value Decomposition" to do this.
# Concepts used throughout the program
### Singular Value Decomposition (SVD):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SVD breaks down an input matrix A (having dimensions m*n) into 3 parts:
1. Orthogonal matrix U (Dimensions: m*m)
2. Diagonal matrix S (Dimensions: m*n)
3. Orthogonal matrix V (Dimensions: n*n)

SVD is widely used to compress data. Its interpretation on text data can be done as follows: 
- Documents are shown as V rows.
- The similarity of documents can be calculated by analysis of the VS rows.
- Words are shown as U rows.
- The similarity of terms can be defined by analysing the rows of the US matrix.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The matrix S is always a diagonal matrix with non-negative descending values, known as the **singular values**. Each non zero value represents **concepts**. There can't be more concepts than there are documents. The magnitude of the values describes how much variance each feature describes in the data. Interpretation of SV describes the **relationship between concepts and documents**. It is a set of column vectors and each eigenvector describes a concept. Components of each vector represent the weightage/contribution that each document has, in describing that particular concept. The matrix product VS describes the **relation between documents (VS's rows) and the features (VS's columns)**.

### Frobenius Norm:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The Frobenius norm of a matrix is nothing but the square root of the sum of squares of its elements. In the case of LSI, it is used to compare how different two matrices are. Let’s say that the matrices we want to compare are *A1* and *A2*. This can be done so by computing the Frobenius norm of *(A1 - A2)*  (**F1**). Then we compute the Frobenius norm of *A1* (**F2**). We divide both of them **F1/F2** and according to the answer we get, we can say how different those two matrices are. This answer will always be between 0 and 1. The lesser the answer, the lesser will be the difference between A1 and A2. (For better understanding: as if A1 = A2, our answer will be 0 since F1 will be equal to 0)

### Jacobi Eigenvalue Theorem:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It is an iterative method to find eigenvalues and eigenvectors of symmetric matrices. It is based on the series of rotations. Here we apply similarity transformations on a matrix such that the given matrix gets converted into a diagonal matrix. Diagonal elements of the final diagonal matrix will be the approximation of eigenvalues of the original given matrix. 
You can see the detailed description on how to implement the Jacobi Eigenvalue theorem in the report PDF. 

# Approach
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To perform LSI, we first load and preprocess the text data. We remove all punctuations from it and store it in a python list. Then we tokenized each string in this list. After that, we found unique words in each string and stored all of these words in  a wordset. This wordset is a set of unique words of all documents. 

Now, we will find the frequency of all words in each document one by one. We measured term frequency by *(count of word)/(total words in respective document)* for all the documents. After that, we stored the **IDF** ([Inverse Document Frequency](https://www.geeksforgeeks.org/tf-idf-model-for-page-ranking/)) value of all words.
Then, we multiplied the term frequency of words in the respective documents with the IDF value of each word. This way, we measured the significance of each word in that document.

We stored these values in a dataframe called X. **Rows of X transpose are words** and **columns are document numbers**. 
We converted this dataframe into a matrix and performed SVD on it.

We iterated from 1 to min(m,n) where (m = number of rows, n = number of columns) singular values and chose the first i (iterator) number of singular values. Then, we reconstructed the l_2d matrix by using the first i singular values. **As we reduce the number of singular values, the reconstructed matrix we get gets more and more compressed.** 

After deciding a threshold for the Frobenius norm, we compared the original matrix and the reconstructed matrix. If the value returned by the frobenius function of two matrices is less than this threshold, we will use the reconstructed matrix for the search.

For the search function, we simply ask for an input. If any word of that input is not present in any of the documents, we say that no matching documents were found. If it is found, we calculate the score of each document (a higher score means that the document is more relevant). This is done so by using the reconstructed and compressed matrix we get from running the functions above.

**Note**: In this program, since we wanted to avoid the use of libraries as much as possible, we have performed SVD completely from scratch with the help of the Jacobi Eigen value theorem. Due to this, the SVD function takes O(n^4) time to return the U, S and V matrices. Hence, this makes our code a bit inefficient. 

# Result
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In order to compare 5 documents, the program takes about 1 minute time to compute U, S and V. After these three are computed, the program proceeds to run smoothly. 

Check out the report to view the detailed output.
