# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:35:28 2021

@author: Lukas
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import basis
import time

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


tf.random.set_seed(2021)
np.random.seed(2021)



#==============================================================
#                           Trainingsdaten
#==============================================================
#Größe des Trainingssamples
batch = 20
#Größe des Intervalls
a = -1
b = 1

#Bestrafung von nicht erwünschten Eigenschaften der Lösung
reg=1
#Lernrate
lr = 0.03


#Funktionen die gelernt werden sollen

#Rauschen (Normalverteilt)
e=0.0

#1 dimensional
def f1(x,y,e):
    return x*y + e*np.random.normal(size=x.shape)

def f2(x,y,e):
    return np.abs(np.pi*x) + e*np.random.normal(size=x.shape)

def f3(x,y,e):
    return x**2-x+2 + e*np.random.normal(size=x.shape)

def f4(x,y,e):
    return np.exp(-x**2) + e*np.random.normal(size=x.shape)


#Bestimme welche Funktion gelernt werden soll
def f(x,y,e):
    return f1(x,y,e)

#Ordner in dem Bilder gespeichert werden
ordner="multiplication/"
#==============================================================
#Erstelle Trainings und Testdaten
train_data_x = np.linspace(a, b, num=batch)
train_data_y = np.linspace(a, b, num=batch)
test_data_x = np.linspace(a-0.01, b+0.01, num=batch)
test_data_y = np.linspace(a-0.01, b+0.01, num=batch)

X,Y = np.meshgrid(train_data_x,train_data_y)
tX,tY = np.meshgrid(test_data_x,test_data_y)

train_data_x=X.flatten()
train_data_y=Y.flatten()
train_Z = f(train_data_x,train_data_y,e)

train_data_x = tf.constant(train_data_x,tf.float32)
train_data_y = tf.constant(train_data_y,tf.float32)
train_Z = tf.constant(train_Z,tf.float32)
testX = tf.constant(tX.flatten(),tf.float32)
testY = tf.constant(tY.flatten(),tf.float32)

#==============================================================
#                           Netzparameter
#==============================================================

#Größe des Netzes
in_dim = 3
layers = 7
#Genauigkeit
cutoff_dim = 11
      
#==============================================================      

# zum Ausführen des Programms wird ein Simulator benötigt. Hier wird das backend von tensorflow verwendet
#cutoff_dim gibt an wieviele Dimensionen des Fock-Raums für die Simulation benutzt werden sollen
#Je höher die Zahl, desto kleiner ist der Fehler auf Operationen, aber desto mehr Zeit wird benötigt

eng = sf.Engine('tf', backend_options={"cutoff_dim": cutoff_dim, "batch_size": batch**2})


#==============================================================
#                           Initialisierung
#==============================================================

#Erstelle ein Programm mit N qumodes
qnn = sf.Program(in_dim)

# initialisiere Parameter zufällig
weights = basis.init(in_dim, layers) 
num_params = np.prod(weights.shape)   # Gesamtzahl an Parametern
    
#Erstelle einen Array mit symbolischen Variabeln die später optimiert werden
sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
sf_params = np.array([qnn.params(*i) for i in sf_params])

#symbolischer Parameter für den Input
x_data = qnn.params("input1")
y_data = qnn.params("input2")
#==============================================================

#Baue die Struktur des Netzes auf
with qnn.context as q:
    
    #Setze den Input des Netzes als Verschiebung im Ortsraum
    ops.Dgate(x_data) | q[0]
    ops.Dgate(y_data) | q[1]
    
    
    for l in range(layers):
        basis.layer(sf_params[l], q)
        

#==============================================================
#                           Kostenfunktion
#==============================================================        
    
def costfunc(weights):
    #Um Tensorflow benutzen zu können muss ein Dictionary zwischen den symbolischen
    #Variablen und den Tensorflowvariablen erstellt werden
    mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))} 
    mapping["input1"] = train_data_x
    mapping["input2"] = train_data_y
    
    # benutze den Tensorflowsimulator
    state = eng.run(qnn, args=mapping).state

    #Ortsprojektion und Varianz
    output = state.quad_expectation(2)[0]
   
    #Größe die minimiert werden soll
    loss = tf.reduce_mean(tf.abs(output - train_Z) ** 2)

    
    #Stelle sicher, dass der Trace des Outputs nahe bei 1 bleibt
    #Es wird also bestraft, wenn der Circuit Operationen benutzt
    #die für große Rechenfehler sorgen (dazu führen, dass der Anteil an höheren Fockstates zunimmt)
    trace = tf.abs(tf.reduce_mean(state.trace()))
    
    cost = loss + reg * (tf.abs(trace - 1) ** 2)

    return cost, loss, trace, output



#Das Training dieses Netzes dauert mehrere Stunden!
#==============================================================
#                           Training
#==============================================================

weights = tf.Variable(weights)
history = []
start_time = time.time()

#Nutze einen Optimierer von Tensorflow. Genauer gesagt: Adam (arXiv:1412.6980v9)

opt= tf.keras.optimizers.Adam(learning_rate=lr)

epochs=1000
# Führe das Training 1000 mal durch
for i in range(epochs):
        
    # wenn das Programm gelaufen ist, dann resete die Engine
    if eng.run_progs:
        eng.reset()
        
    with tf.GradientTape() as tape:
        cost, loss, trace, output = costfunc(weights)
        
    gradients = tape.gradient(cost, weights)
    opt.apply_gradients(zip([gradients], [weights]))
    
    history.append(loss)
    
    #alle 10 Schritte 
    if i % 10 == 0:
        print("Rep: {} Cost: {:.4f} Loss: {:.4f} Trace: {:.4f}".format(i, cost, loss, trace))  
        #Speichere grafisch den Trainingsfortschritt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, np.reshape(output,(batch,batch)), cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
        ax.plot_surface(X, Y, np.reshape(train_Z,(batch,batch)), cmap="Greys", lw=0.5, rstride=1, cstride=1,alpha=0.2)
        fig.set_size_inches(4.8, 5)
        name=ordner+str(i)+".png"
        fig.savefig(name, format='png', bbox_inches='tight')
        plt.close(fig)

        
        
#Gebe die Dauer des Trainings aus           
end_time = time.time()        
print("Dauer: ",np.round(end_time-start_time),"Sekunden") 
np.save("weights_mult",weights)
eng.reset()

# %matplotlib inline
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
plt.style.use('default')

#Erstelle einen Plot des Trainingsverlaufes
plt.plot(history)
plt.ylabel('Kosten')
plt.xlabel('Epoche')
plt.show()

#Teste den Algorithmus an nicht gelernten Trainingsdaten
#==============================================================
#                           Test
#==============================================================

weights=np.load("weights_mult.npy")

"""
#Simuliere fehlerhafte Gates durch Veränderung einzelner Parameter
from random import randint


for fehler in range(1):
    print(fehler)
    for anz in range(1):
        weights=np.load("weights_mult.npy")
        for z in range(8):
            i=randint(0,6)
            j=randint(0,27)
            weights[i,j] += 0.1*np.random.normal(size=1)
        
        cost, loss, trace, output = costfunc(weights)
        eng.reset()
        print(loss)

"""
mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))} 
mapping["input1"] = testX
mapping["input2"] = testY


# benutze den Tensorflowsimulator
state = eng.run(qnn, args=mapping).state

#Ortsprojektion der Ausgabe
output = state.quad_expectation(2)[0]


#Visualisiere die Ausgabe für alle Testdaten
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
#ax.plot_surface(tX, tY, np.reshape(output,(batch,batch)), cmap="RdYlGn", lw=0.5, rstride=1, cstride=1,alpha=0.8)
ax.plot_surface(X, Y, np.reshape(output,(batch,batch)), cmap="RdYlGn", lw=0.5, rstride=1, cstride=1,alpha=0.8)
ax.plot_surface(X, Y, np.reshape(train_Z,(batch,batch)), cmap="Greys", lw=0.5, rstride=1, cstride=1,alpha=0.4)
fig.set_size_inches(4.8, 5)
name=ordner+"Test"+".png"
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
fig.savefig(name, format='png', bbox_inches='tight')






    
    