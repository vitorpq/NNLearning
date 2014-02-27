# -*- coding: utf-8 -*-
"""
Process Data Dynamic function:
    It transform a data vector in a matrix using the own vector itens as anteriors
    
Created on Thu Jan 30 21:24:19 2014

@author: Vitor Emmanuel Andrade


-------------------------MATLAB Function---------------------
function data = pddin(Dados, ant)
    for j = 1:ant
        if j == 1
            data = Dados;
        else
            data = cat(2, data(1:end-1,1:end-2), Dados(j:end,:));
        end
    end
end
"""

def pddin(Data, anterior,initial=0):
    """
    Process Data Dynamic function:
    It transform a data vector in a matrix using the own vector itens as anteriors
    """
#   Check if array is unidimensional and fix it. Check if 
#   is a line or column vector. I prefer column vector
#   if Data.ndim == 1:
    from numpy import concatenate
     
    for j in xrange(anterior):
        if j == 0:
            temp = Data
        else:
            temp = concatenate((temp[:-1,], Data[j:,]), axis=1)
    return temp

if __name__ == '__main__':
    from numpy import loadtxt
    from numpy import array
    dados = loadtxt('data/exchanger.dat', delimiter=';')
    Data = dados[:20,0]
    Data = array([Data]).T
    r = pddin(Data, 4)
    print(r)