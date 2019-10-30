# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:00:13 2019

@author: Victor Biazon
Regressão Linear - Metodo dos Minimos Quadrados
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit


def MultMatrix (A, B): #multiplicação de matrizes
    if len(A[0]) != len(B): #verifica se asmatrizes podem ser multiplicadas
        print("Não é possivel multiplicar")
        return
    else:
        result = np.zeros((len(A), len(B[0])), float)
        for k in range(len(B[0])): #executa a multiplicação de elemento a elemento e soma no result.
            for l in range(len(A[:])):
                for m in range(len(B)):
                    result[l][k] += A[l][m]*B[m][k]
    return result


def DeterminantMatrix(mat): #calcula determinante da matriz
    if len(mat) != len(mat[0]): # verifica se é uma matriz quadrada
        print("Matriz não quadrada")
        return
    if len(mat) == 2: #se for uma matriz 2x2 retorna a determinante simples
        det = mat[0,0]*mat[1,1] - mat[1,0]*mat[0,1]
        return det
    Aux = np.c_[mat,mat[:,:-1]] #se nao for 2x2 copia as primeiras colunbas para o final de uma matriz auxiliar
    det = 0
    for m in range(0, len(mat)): #parte positiva 
        lin = 1
        for n in range(0, len(mat[0])):
            lin *= Aux[n, m + n]
        det += lin
    for m in range(0, len(mat)): #parte negativa
        lin = 1
        for n in range(0, len(mat)):
            lin *= Aux[len(mat) - n -1, m + n]
        det -= lin
    return det

def InverseMatrix (mat): #calcula a inversa da matriz por sistemas lineares
    if len(mat) != len(mat[:]):
        print("Matriz não quadrada")
        return
    if DeterminantMatrix(mat) != 0:       
        i = len(mat)
        j = len(mat[:])
        ident = np.zeros((i,j), int)
        for index in range(0,len(ident)):
            ident[index,index] = 1
        inv = np.linalg.solve(mat, ident)   #encontra a matriz inversa por sistemas lineares com a matriz identidade.
        return inv
    
def InverseMatrixCofactor (mat): #inversa usando cofatores
    if len(mat) != len(mat[:]):
        print("Matriz não quadrada")
        return
    D = np.linalg.det(mat) #calcula determinante da matriz
    if  D != 0:       #se for 0 não há inversão
        i = len(mat)
        j = len(mat[:])
        cofact = np.zeros((i,j), float)
        for m in range(0, i): 
            for n in range(0, j):
                aux = np.delete(np.delete(mat, m, 0), n, 1)
                cofact[m,n] = np.linalg.det(aux) * (-1)**(m+n)           
        tcofact = MatrixTranspose(cofact)
        inv = tcofact / D
        return inv
    else: return

def MatrixTranspose (mat): #calcula a transposta da matriz
    transp = np.zeros((len(mat[0]), len(mat)),float)
    for m in range(0, len(mat[0])): #troca elementos de linha e coluna por coluna e linha
        for n in range(0, len(mat)):
            transp[m,n] = mat[n,m]
    return transp

def LinearRegression(data): #regressão linear simples
    X = np.asarray(data.iloc[:,:-1]) #separa as varaiveis independentes no vetor X
    y = np.asarray(data.iloc[:,-1:]) #separa as variaveis dependentes no vetor Y
    XBeta = np.c_[np.ones((len(X),1), float), X] #adiciona vetor de 1's no vetor X
    XBetat = MatrixTranspose(XBeta) #transpoe o vetor XBeta para executar a multiplicação
    Xtx = MultMatrix(XBetat,XBeta) #multiplica XBetat por XBeta
    Xty = MultMatrix(XBetat,y) #multiplica XBetat pelo vetor y
    Beta = MultMatrix(InverseMatrix(Xtx), Xty) #Multiplica a inversa do vetor Xtx pelo vetor Xty
    
    X0 = 200
    X0_a = np.array([[1, X0]])
    Y_pred = MultMatrix(X0_a, Beta) #preve a pressão para a temperatura de 200ºF
    print("X0(200ºF) = ", Y_pred)
    
    #Plotar o Grafico
    plt.figure()
    plt.scatter(X, y, color = 'red')
    plt.scatter(X0, Y_pred, color = 'black')
    plt.plot(XBeta[:,1], MultMatrix(XBeta, Beta), color = 'blue')
    plt.title('Dados de ponto de ebulição da água por P x Temperatura')
    plt.xlabel('Temperature (ºF)')
    plt.ylabel('BPt')
    plt.show()
    return

def QuadLinearRegression(data): # regressão linear quadratica
    X = np.asarray(data.iloc[:,:-1]) #separa as varaiveis independentes no vetor X
    y = np.asarray(data.iloc[:,-1:]) #separa as variaveis dependentes no vetor Y
    XBeta = np.c_[np.ones((len(X),1), float), X, np.square(X)] #adiciona vetor de 1's no vetor X e vetor X² na coluna seguinte
    XBetat = MatrixTranspose(XBeta) #transpoe o vetor XBeta para executar a multiplicação
    Xtx = MultMatrix(XBetat,XBeta) #multiplica XBetat por XBeta
    Xty = MultMatrix(XBetat,y) #multiplica XBetat pelo vetor y
    Beta = MultMatrix(InverseMatrix(Xtx), Xty) #Multiplica a inversa do vetor Xtx pelo vetor Xty
    X0 = 200
    X0_a = np.array([[1, X0, X0**2]])
    Y_pred = MultMatrix(X0_a, Beta) #preve a pressão para a temperatura de 200ºF
    print("X0(200ºF) = ", Y_pred)
    
    #Plotar o Grafico
#    plt.figure()
    plt.scatter(X, y, color = 'red')
    plt.scatter(X0, Y_pred, color = 'black')
    plt.plot(XBeta[:,1], MultMatrix(XBeta, Beta), color = 'purple')
    plt.title('Dados de ponto de ebulição da água por P x Temperatura')
    plt.xlabel('Temperature (ºF)')
    plt.ylabel('BPt')
    plt.show()
    return

def RobustLinearRegression(data): #regressão linear robusta
    X = np.asarray(data.iloc[:,:-1]) #separa as varaiveis independentes no vetor X
    y = np.asarray(data.iloc[:,-1:]) #separa as variaveis dependentes no vetor Y
    XBeta = np.c_[np.ones((len(X),1), float), X] #adiciona vetor de 1's no vetor X
    XBetat = MatrixTranspose(XBeta) #transpoe o vetor XBeta para executar a multiplicação
    Xtx = MultMatrix(XBetat,XBeta) #multiplica XBetat por XBeta
    Xty = MultMatrix(XBetat,y) #multiplica XBetat pelo vetor y
    Beta = MultMatrix(InverseMatrix(Xtx), Xty) #Multiplica a inversa do vetor Xtx pelo vetor Xty
    Wi = np.ones((len(X),1)) #cria vetor de pesos
    WX = np.zeros_like(X, dtype = float) #cria vetor de X ponderado pelos pesos
    WY = np.zeros_like(y, dtype = float) #cria vetor de Y ponderado pelos pesos
    Y_pred = MultMatrix(XBeta, Beta) #realiza predição inicial para comparar com os valores de Y reais
    for l in range(0,1):
        for k in range(0,len(X)):
            Wi[k] = abs(1/(y[k] - Y_pred[k])) #calcula o peso Wi do elemento k
            WY[k] = Wi[k] * y[k] #multiplica o elemento y correspondente ao peso W por este
            for j in range(0, len(X[0])):
                WX[k,j] = Wi[k] * X[k,j] #multiplica o elemento X correspondente ao peso W por este
        XBeta = np.c_[Wi, WX] #cria novo vetor XBeta
        XBetat = MatrixTranspose(XBeta) #transpoe o vetor XBeta
        Xtx = MultMatrix(XBetat,XBeta) 
        Xty = MultMatrix(XBetat, WY)
        Beta = MultMatrix(InverseMatrix(Xtx), Xty)    #calcula novos coeficientes Beta
    
    X0 = 200.0
    X0_a = np.array([[1, X0]])
    Y_pred = MultMatrix(X0_a, Beta) #preve a pressão para a temperatura de 200ºF
    print("X0(200ºF) = ", Y_pred)
    Y_pred_a = MultMatrix(np.c_[np.ones((len(X),1), float), X], Beta)
    #Plotar o Grafico
#    plt.figure()
    plt.scatter(X, y, color = 'red')
    plt.scatter(X0, Y_pred, color = 'black')
    plt.plot(X, Y_pred_a, color = 'green')
    plt.title('Dados de ponto de ebulição da água por P x Temperatura')
    plt.xlabel('Temperature (ºF)')
    plt.ylabel('BPt')
    plt.show()
    return


## MAIN
data = pd.read_table('alpswater.txt', decimal  = ",")
LinearRegression(data)
QuadLinearRegression(data)
RobustLinearRegression(data)

