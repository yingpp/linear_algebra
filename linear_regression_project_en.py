
# coding: utf-8

# # 1 Matrix operations
# 
# ## 1.1 Create a 4*4 identity matrix

# In[26]:


#This project is designed to get familiar with python list and linear algebra
#You cannot use import any library yourself, especially numpy

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO create a 4*4 identity matrix 
I = [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]


# ## 1.2 get the width and height of a matrix. 

# In[27]:


#TODO Get the height and weight of a matrix.
def shape(M):
    height = len(M)
    weight = len((M)[0])
    return height,weight


# In[28]:


# run following code to test your shape function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 round all elements in M to certain decimal points

# In[29]:


# TODO in-place operation, no return value
# TODO round all elements in M to decPts

from decimal import Decimal
# from decimal import *

def matxRound(M, decPts=4):
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j] = round(M[i][j], decPts)
        


# In[30]:


# run following code to test your matxRound function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 compute transpose of M

# In[31]:


#TODO compute transpose of M
def transpose(M):
    if M == []:
        return []
    else:
        new_m = []
        for j in range(len(M[0])):
            lst = []
            for i in range(len(M)):
                lst.append(M[i][j])
            new_m.append(lst)
        return new_m


# In[32]:


# run following code to test your transpose function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 compute AB. return None if the dimensions don't match

# In[33]:


#TODO compute matrix multiplication AB, return None if the dimensions don't match
def matxMultiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError
    else:
        new_matrix = []
        for i in range(len(A)):
            lst = []
            for k in range(len(B[0])):
                result = 0
                for j in range(len(A[0])):
                    result += A[i][j]*B[j][k]
                lst.append(result)
            new_matrix.append(lst)
        return new_matrix
        


# In[34]:


# run following code to test your matxMultiply function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussian Jordan Elimination
# 
# ## 2.1 Compute augmented Matrix 
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# Return $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[35]:


#TODO construct the augment matrix of matrix A and column vector b, assuming A and b have same number of rows
def augmentMatrix(A, b):
    if len(A) != len(b):
        raise ValueError
    else:
        Ab = []
        for i in range(len(A)):
            row = []
            for val in A[i]:
                row.append(val)
            for val in b[i]:
                row.append(val)
            Ab.append(row)
        return Ab


# In[36]:


# run following code to test your augmentMatrix function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 Basic row operations
# - exchange two rows
# - scale a row
# - add a scaled row to another

# In[37]:


# TODO r1 <---> r2
# TODO in-place operation, no return value
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]


# In[38]:


# run following code to test your swapRows function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[39]:


# TODO r1 <--- r1 * scale
# TODO in-place operation, no return value
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    else:
        for i in range(len(M[r])):
            M[r][i] = M[r][i]*scale


# In[40]:


# run following code to test your scaleRow function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[41]:


# TODO r1 <--- r1 + r2*scale
# TODO in-place operation, no return value
def addScaledRow(M, r1, r2, scale):
    if scale == 0:
        raise ValueError
    else:
        for i in range(len(M[r1])):
            M[r1][i] = M[r2][i]*scale + M[r1][i]


# In[42]:


# run following code to test your addScaledRow function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gauss-jordan method to solve Ax = b
# 
# ### Hint：
# 
# Step 1: Check if A and b have same number of rows
# Step 2: Construct augmented matrix Ab
# 
# Step 3: Column by column, transform Ab to reduced row echelon form [wiki link](https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form)
#     
#     for every column of Ab (except the last one)
#         column c is the current column
#         Find in column c, at diagonal and under diagonal (row c ~ N) the maximum absolute value
#         If the maximum absolute value is 0
#             then A is singular, return None （Prove this proposition in Question 2.4）
#         else
#             Apply row operation 1, swap the row of maximum with the row of diagonal element (row c)
#             Apply row operation 2, scale the diagonal element of column c to 1
#             Apply row operation 3 mutiple time, eliminate every other element in column c
#             
# Step 4: return the last column of Ab
# 
# ### Remark：
# We don't use the standard algorithm first transfering Ab to row echelon form and then to reduced row echelon form.  Instead, we arrives directly at reduced row echelon form. If you are familiar with the stardard way, try prove to yourself that they are equivalent. 

# In[43]:


#TODO implement gaussian jordan method to solve Ax = b

""" Gauss-jordan method to solve x such that Ax = b.
        A: square matrix, list of lists
        b: column vector, list of lists
        decPts: degree of rounding, default value 4
        epsilon: threshold for zero, default value 1.0e-16
        
    return x such that Ax = b, list of lists 
    return None if A and b have same height
    return None if A is (almost) singular
"""
def find_max_diag(A, j):
    current = abs(A[j][j])
    i = j+1
    max_row = j
    while i < len(A):
        val = abs(A[i][j])
        if val > current:
            max_row = i
        i += 1
    return max_row

def check_singular(M, j):
    epsilon = 1.0e-16
    result = []
    for i in range(j, len(M)):
        result.append(M[i][j])
    biggest = 0
    for i in result:
        if abs(i) > biggest:
            biggest = abs(i)
    if biggest < epsilon:
        return True
    
def is_zero(num, epsilon = 1.0e-16 ):
    if num < epsilon:
        return True

        
def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    if len(A) != len(b):
        return None
    else:
        result = []
        Ab = augmentMatrix(A, b)
        n_col = len(Ab[0])
        n_row = len(Ab)
        for j in range(n_col - 1):  # equals n_row
            if check_singular(Ab, j) == True:
                return None
            else:
                max_diag_row = find_max_diag(Ab, j)
                if j != max_diag_row:
                    Ab[j], Ab[max_diag_row] = Ab[max_diag_row], Ab[j]
                # normalized row by the j-th column value
                for k in range(n_row):
                    denom = float(Ab[k][j])
                    if is_zero(denom, epsilon = 1.0e-16):
                        continue
                    for i in range(n_col):
                        Ab[k][i] = Ab[k][i] / denom
                # cancel j-th column value except the j-th row
                for k in range(n_row):
                    if k == j:
                        continue
                    for i in range(n_col):
                        Ab[k][i] = Ab[k][i] - Ab[j][i]
        for k in range(n_row):
            denom = float(Ab[k][k])
            if is_zero(denom, epsilon = 1.0e-16):
                continue
            for i in range(n_col):
                Ab[k][i] = Ab[k][i] / denom
        for i in range(n_row):
            result.append([Ab[i][n_col-1]])
        matxRound(result, decPts)
        return result


# In[44]:


# run following code to test your addScaledRow function
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## 2.4 Prove the following proposition:
# 
# **If square matrix A can be divided into four parts: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} $, where I is the identity matrix, Z is all zero and the first column of Y is all zero, 
# 
# **then A is singular.**
# 
# Hint: There are mutiple ways to prove this problem.  
# - consider the rank of Y and A
# - consider the determinate of Y and A 
# - consider certain column is the linear combination of other columns

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
#     A =
#   \begin{bmatrix}
#   A_{1,1} & \cdots & A_{1,k-1} & \cdots & \cdots & \cdots \\
#   \vdots  & \ddots & \vdots  & \cdots& \ddots& \cdots \\
#   A_{k-1,1} & \cdots & A_{k-1,k-1} & \cdots& \cdots& \cdots \\
#   A_{k,1} & \cdots & A_{k,k-1} &  A_{k,k} & \cdots & A_{k,m}\\
#   \vdots  & \ddots & \vdots & \vdots  & \ddots & \vdots \\
#   A_{m,1} & \cdots & A_{m,k-1} & A_{m,k} & \cdots & A_{m,m}
#  \end{bmatrix}
# 
#     I = 
#  \begin{bmatrix}
#   A_{1,1} & \cdots & A_{1,k-1} \\
#   \vdots  & \ddots & \vdots  \\
#   A_{k-1,1} & \cdots & A_{k-1,k-1} 
#  \end{bmatrix}
#  
#      Z = 
#  \begin{bmatrix}
#   A_{k,1} & \cdots & A_{k,k-1} \\
#   \vdots  & \ddots & \vdots  \\
#   A_{m,1} & \cdots & A_{m,k-1} 
#  \end{bmatrix}
#  
#      Y = 
#  \begin{bmatrix}
#   A_{k,k} & \cdots & A_{k,m} \\
#   \vdots  & \ddots & \vdots  \\
#   A_{m,k} & \cdots & A_{m,m} 
#  \end{bmatrix}
#  
#  In Matrix A, assume that :
#  
#  the number of column is m and the number of row is m,
#  
#  the number of first column is k and the number of first row is k.
#  
#  (1) In Matrix I, $A_{1,1} = A_{2,2} \cdots = A_{k-1,k-1} = 1 ;$
#  
#  (2) In Matrix Z, all elements are zero;
#  
#  (3) In Matrix Y, $A_{k,k} = A_{k,k+1} \cdots = A_{k,m} = 0 ;$
#  
#  Because of (3), apply any elementary row operations from row k to row m, it cannot make $A_{k,k}$ into 1, so we will have 
#  
#  0 = $x_k$, $x_k$ will become a free variables. As a result, A is a singular.
#  
#  

# ---
# 
# # 3 Linear Regression: 
# 
# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# 
# We define loss funtion $E$ as 
# $$
# E(m, b) = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# and we define vertex $Y$, matrix $X$ and vertex $h$ :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$
# 
# 
# Proves that 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
# Because:
# 
# $2X^TXh$ = $\begin{bmatrix}
# 2m\sum_{i=1}^{n}x_i^2 + 2b\sum_{i=1}^{n}x_i \\
# 2m\sum_{i=1}^{n}x_i + 2b
# \end{bmatrix}$
# 
# $2X^TY$ = $\begin{bmatrix}
# 2\sum_{i=1}^{n}x_iy_i \\
# 2\sum_{i=1}^{n}y_i
# \end{bmatrix}$
# 
# $2X^TXh - 2X^TY$ = $\begin{bmatrix}
# \sum_{i=1}^{n}-2x_i(y_i-mx_i-b) \\
# \sum_{i=1}^{n}-2(y_i-mx_i-b) 
# \end{bmatrix}$
# 
# Also:
# 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# As a result:
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# We define loss funtion $E$ as 
# $$
# E(m, b) = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# and we define vertex $Y$, matrix $X$ and vertex $h$ :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$
# 
# Proves that 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$
# 
# $$
# \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：

# ## 3.2  Linear Regression
# ### Solve equation $X^TXh = X^TY $ to compute the best parameter for linear regression.

# In[53]:


#TODO implement linear regression 
'''
points: list of (x,y) tuple
return m and b
'''

    
def linearRegression(points):
    Y = []
    X = []
    for i in points:
        Y.append([i[1]])
        X.append([i[0], 1])
    X_trans = transpose(X)
    m = matxMultiply(X_trans, X)
    b = matxMultiply(X_trans, Y)
    return gj_Solve(m, b)
 


# ## 3.3 Test your linear regression implementation

# In[55]:


#TODO Construct the linear function

#TODO Construct points with gaussian noise
import random

m = random.randint(0, 100)

b = random.randint(0, 100)

number = 0

points = []

while number < 50:
    x = random.randint(-10,10)
    y = m*x + b + random.gauss(0,1)
    points.append([x,y])
    number +=1
    
print (m, b)
print (linearRegression(points))

#TODO Compute m and b and compare with ground truth

