# -*- coding: utf-8 -*-
"""
Created on Sun May 30 18:47:14 2021

@author: Lukas
"""
"""
Bei Continous Variable (CV) quantum computation werden statt qubits qumodes verwendet.
Bei diesen handelt es sich um kontinuierliche Zustände. Mögliche Basen sind bspw. über
die Darstellung im Fock-Raum, als Wignerfunktion im Phasenraum gegeben.
Hier wird hauptsächlich Darstellung im Phasenraum (x,p) benutzt.

Dieser Code orientiert sich an https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html#id9
insbesondere die Funktion orth()
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops

tf.random.set_seed(2021)
np.random.seed(2021)






"""
Um eine parametrisierte orthogonale Matrix auf den Input anzuwenden
werden BeamSplitter (BSgate's) und Rotationen (Rgate) verwendet.
Siehe hierzu arXiv:1603.08788v2
Dabei muss ein BeamSplitter je zwei qumodes des Inputvektor verbinden. 
Hat der Vektor N einträge, so bedeutet dies N(N-1)/2 BSGates. Da jedes Gate 2 
Parameter hat, benötigt man hierfür N(N-1) Parameter.
(Der zweite Parameter bestimmt die Phase nach Anwendung des BSGates. Klassische
Neuronale Netze besitzen diese Möglichkeit nicht.)

Um aber allgemeine orthoganle Matrizen zu ermöglichen, ist es notwendig
auch einzelne Rotationen der qumodes zu ermöglichen. Daher werden am Ende 
der Operation RGate angewendet. Hierbei reicht es aus nur N-1 Rgates anzuwenden,
da diese eine relative Rotation zu den übrigen qumodes ermöglichen sollen.
"""         

def orth(params, q):
    N = len(q)
    
    
    theta = params[:N*(N-1)//2] # Ersten N*(N-1)//2
    phi = params[N*(N-1)//2:N*(N-1)] # Die nächsten N*(N-1)//2
    rphi = params[-(N-1):] # Die letzten N-1
    
    #Für N>1 müssen die qumodes miteinander verbunden werden.
    #Insgesamt werden N(N-1)/2 BS-Gates hierfür benötigt.
    #Für die Anordnung gibt es dabei mehrere Möglichkeiten, allerdings wird
    #hier die Anordnung nach arXiv:1603.08788v2 verwendet, da diese bei realen
    #Geräten vorkommende Verluste am gleichmäßigsten verteilt.
    

    n=0
    
    if(N>1):
        for i in range(N):
            
            #Bilde Paare aus den qumodes. Je zwei Nachbarn werden miteinander verbunden
            #(1,2),(2,3),(3,4),(4,5)...
            
            for j, (qi, qj) in enumerate(zip(q[:-1], q[1:])):
                
                #Nun wird abwechselnd (1,2) und (3,4)... oder (2,3) und (4,5)... 
                #miteinander verbunden
                
                if( (i+j)%2 == 0 ):
                    ops.BSgate(theta[n],phi[n]) | (qi,qj)
                    n+=1
        #Am Ende wird eine Rotation auf N-1 qumodes angewandt.
        for i in range(N-1):
            ops.Rgate(rphi[i]) | q[i]
    else:
        ops.Rgate(rphi[0]) | q[0]
        
        
        
"""
Fasse die verschiedenen Operationen einer Schicht zusammen
x -> f(Wx+a)
"""

def layer(params,q):
    N=len(q)
    M = N * (N - 1) + max(1, N - 1) #Anzahl der Parameter für Ortho
    
    ort1=params[:M]
    diag=params[M:M+N]
    ort2=params[M+N:2*M+N] 
    br=params[2*M+N:2*M+2*N]
    bi=params[2*M+2*N:2*M+3*N]
    activ=params[2*M+3*N:2*M+4*N]
    
    
    """Rechne Mx aus"""
    
    #Führe orthonormale Matrix aus
    orth(ort1,q)
    
    #Multipliziere die Diagonalmatrix mit x
    for i in range(N):
        ops.Sgate(diag[i]) | q[i]
    
    #Führe orthonormale Matrix aus
    orth(ort2,q)
    
    for i in range(N):
        
        #Addiere den Bias der Neuronen bi ist bei einem klassischen NN 0
        ops.Dgate(br[i],bi[i]) | q[i]
        
        #Wende eine nicht lineare Aktivierungsfunktion an. Bei klassischen NN
        #ist dies beispielsweise die Sigmoid-Funktion oder ReLU
        
        #Hier wird ein non-gaussian verwendet um zu ermöglichen, dass die
        #Gesamtschaltung auch non-gaussian Operationen lernen kann. Aus 
        #recheneffizienzgründen wird das Kerr-Gate verwendet
        
        ops.Kgate(activ[i]) | q[i]
        
"""
    N: Dimension des Inputs (Anzahl der verwendeten Qumodes)
    l: Anzahl der Schichten des QNN
"""
def init(N,l):
    std=0.1
    M=N*(N-1)+max(1,N-1)    #Um den Fall N=1 abzudecken
    
    #initializiere die Parameter als Tensorflowvariablen mit einer Normalverteilung
    
    ort1_weights = tf.random.normal(shape=[l, M], stddev=std)
    
    ort2_weights = tf.random.normal(shape=[l, M], stddev=std)
    
    diag= tf.random.normal(shape=[l, N], stddev=std)
    br= tf.random.normal(shape=[l, N], stddev=std)
    bi= tf.random.normal(shape=[l, N], stddev=std)
    activ= tf.random.normal(shape=[l, N], stddev=std)
    
    weights=tf.concat([ort1_weights,ort2_weights,diag,br,bi,activ], axis=1)
    
    weights = tf.Variable(weights)

    return weights