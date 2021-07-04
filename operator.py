# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:01:52 2021

@author: Lukas
"""
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import basis
import time
from strawberryfields.utils import random_interferometer
import matplotlib.pyplot as plt



tf.random.set_seed(2021)
np.random.seed(2021)

#Dimension ab der der Fock-Raum abgeschnitten wird (für Simulation)
cutoff_dim = 10

#==============================================================
#                           Trainingsdaten
#==============================================================

#Bestrafung von nicht erwünschten Eigenschaften der Lösung
reg=1
#Lernrate
lr = 0.025

#Anzahl der Epochen
epochs=200

#Dimension des zu lernenden Unitary
dim_oper = 4

#erzeugt eine zufällige 4x4 Matrix
unit_r = random_interferometer(dim_oper)
print(unit_r)

#fülle den Operator bis zum Cutoff mit der Identität auf
unit_z = np.identity(cutoff_dim, dtype=np.complex128)
unit_z[:dim_oper, :dim_oper] = unit_r

#Spalten der Matrix
zielkets = []
for i in range(dim_oper):
    zielkets.append(unit_z[:,i])
zielkets = tf.constant(zielkets, dtype=tf.complex64)


#==============================================================
#                           Netzparameter
#==============================================================

#Größe des Netzes
in_dim = 1  
layers = 15


#==============================================================      
eng = sf.Engine('tf', backend_options={"cutoff_dim": cutoff_dim, "batch_size": dim_oper})


#==============================================================
#                           Initialisierung
#==============================================================

#Erstelle ein Programm mit N qumodes
qnn = sf.Program(in_dim)

# initialisiere Parameter zufällig
weights = basis.init(in_dim, layers) 
anzahl = np.prod(weights.shape)   # Gesamtzahl an Parametern
    
#Erstelle einen Array mit symbolischen Variabeln die im QNN verwendet werden
params = np.arange(anzahl).reshape(weights.shape)
params = params.astype(np.str)   #Variablen sind einfach numeriert

par = []
for i in params:
    par.append(qnn.params(*i))

params = np.array(par)

#symbolischer Parameter für den Input
x_data = qnn.params("input")

#erzeuge Basisvektoren mit der richtigen Dimension
basis_vektoren = np.zeros([dim_oper,cutoff_dim])
np.fill_diagonal(basis_vektoren,1)

#==============================================================

#Baue die Struktur des Netzes auf
with qnn.context as q:
      
    #initialisiert die Basisvektoren
    ops.Ket(basis_vektoren) | q   
    
    #baut Layer des QNN
    for l in range(layers):
        basis.layer(params[l], q)
        
        
#==============================================================
#                           Kostenfunktion
#==============================================================        
    
def costfunc(weights):
    #Um Tensorflow benutzen zu können muss ein Dictionary zwischen den symbolischen
    #Variablen und den Tensorflowvariablen erstellt werden
    dictio = {}
    for symb, var in zip(params.flatten(), tf.reshape(weights, -1)):
        dictio[symb.name] = var

    
    # benutze den Tensorflowsimulator
    state = eng.run(qnn, args=dictio).state

    #Ausgabe-Ket
    ket = state.ket()
   
    #Mittlerer Überlapp
    ueberlapp =  tf.math.real( tf.einsum('bi,bi->b', tf.math.conj(zielkets),ket) )

    loss = tf.abs(tf.reduce_sum(ueberlapp - 1))

    #Stelle sicher, dass der Trace des Outputs nahe bei 1 bleibt
    #Es wird also bestraft, wenn der Circuit Operationen benutzt
    #die für große Rechenfehler sorgen (dazu führen, dass der Anteil an höheren Fockstates zunimmt)
    trace = tf.abs(tf.reduce_mean(state.trace()))
    
    cost =  loss + reg * (tf.abs(trace - 1) ** 2)

    return cost, loss, trace, ket

#==============================================================
#                           Training
#==============================================================
history = []
start_time = time.time()

#Nutze einen Optimierer von Tensorflow. Genauer gesagt: Adam (arXiv:1412.6980v9)

opt= tf.keras.optimizers.Adam(learning_rate=lr)


# Führe das Training 200 mal durch
for i in range(epochs):
        
    # wenn das Programm gelaufen ist, dann resete die Engine
    if eng.run_progs:
        eng.reset()
        
    with tf.GradientTape() as tape:
        cost, loss, trace, ket = costfunc(weights)

    gradients = tape.gradient(cost, weights)
    opt.apply_gradients(zip([gradients], [weights]))
    history.append(loss)
    #alle 10 Schritte 
    if i % 10 == 0:
        print("Epochen: {} Gesamtkosten: {:.4f} Loss: {:.4f} Trace: {:.4f}".format(i, cost, loss, trace))  
        
end_time = time.time()  
      
print("Dauer: ",np.round(end_time-start_time),"Sekunden") 
np.save("weights_unitary",weights)
eng.reset()  

# %matplotlib inline

plt.plot(history)
plt.ylabel('Kosten')
plt.xlabel('Epochen')
plt.show()

#Teste das QNN durch einen Vergleich des gelernten Operators mit
#dem tatsächlichen Operator
#==============================================================
#                           Test
#==============================================================

#lade Gewichte
weights=np.load("weights_unitary.npy")


dictio = {}
for symb, var in zip(params.flatten(), tf.reshape(weights, -1)):
    dictio[symb.name] = var

# benutze den Tensorflowsimulator
state = eng.run(qnn, args=dictio).state

#Ausgabe-Ket
ket = state.ket()

#Extrahiere aus der Ausgabe den relevanten Teil des Operators
learnt_unitary = ket.numpy().T[:dim_oper, :dim_oper]

#Stelle die beiden Operatoren grafisch dar
#Real und Imaginärteil werden getrennt betrachtet
fig, ax = plt.subplots(1, 4, figsize=(7, 4))
ax[0].matshow(unit_r.real, cmap=plt.get_cmap('Blues'))
ax[1].matshow(unit_r.imag, cmap=plt.get_cmap('Reds'))
ax[2].matshow(learnt_unitary.real, cmap=plt.get_cmap('Blues'))
ax[3].matshow(learnt_unitary.imag, cmap=plt.get_cmap('Reds'))

ax[0].set_xlabel(r'$\mathrm{Re}(U_{Ziel})$')
ax[1].set_xlabel(r'$\mathrm{Im}(U_{Ziel})$')
ax[2].set_xlabel(r'$\mathrm{Re}(U_{gelernt})$')
ax[3].set_xlabel(r'$\mathrm{Im}(U_{gelernt})$')
fig.show()