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
from matplotlib import rcParams


tf.random.set_seed(2021)
np.random.seed(2021)



#==============================================================
#                           Trainingsdaten
#==============================================================
#Größe des Trainingssamples
batch = 50
#Größe des Intervalls
a = -1
b = 1

# Führe das Training 100 mal durch
epochs=100

#Bestrafung von nicht erwünschten Eigenschaften der Lösung
reg = 1
reg_err = 0.00
#Lernrate
lr = 0.05


#Funktionen die gelernt werden sollen

#Rauschen (Normalverteilt)
e=0.1

#1 dimensional
def f1(x,e):
    return np.sin(np.pi*x) + e*np.random.normal(size=x.shape)

def f2(x,e):
    return np.abs(np.pi*x) + e*np.random.normal(size=x.shape)

def f3(x,e):
    return x**2-x+x**5 + e*np.random.normal(size=x.shape)

def f4(x,e):
    return np.exp(-5*x**2) + e*np.random.normal(size=x.shape)


#Bestimme welche Funktion gelernt werden soll
def f(x,e):
    return f1(x,e)

#Ordner in dem der Bilder des Trainingsverlauf abgespeichert werden
ordner="Training_sin/"
#==============================================================
train_data_x = np.linspace(a, b, num=batch)
test_data_x = np.linspace(a-0.01, b+0.01, num=batch)

train_data_y = f(train_data_x,e)

train_data_x = tf.constant(train_data_x,tf.float32)
train_data_y = tf.constant(train_data_y,tf.float32)
test_data_x = tf.constant(test_data_x,tf.float32)

#==============================================================
#                           Netzparameter
#==============================================================

#Größe des Netzes
in_dim = 1
layers = 6
#Genauigkeit
cutoff_dim = 10
      
#==============================================================      

# zum Ausführen des Programms wird ein Simulator benötigt. Hier wird das backend von tensorflow verwendet
#cutoff_dim gibt an wieviele Dimensionen des Fock-Raums für die Simulation benutzt werden sollen
#Je höher die Zahl, desto kleiner ist der Fehler auf Operationen, aber desto mehr Zeit wird benötigt

eng = sf.Engine('tf', backend_options={"cutoff_dim": cutoff_dim, "batch_size": batch})


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

#==============================================================

#Baue die Struktur des Netzes auf
with qnn.context as q:
    
    #Setze den Input des Netzes als Verschiebung im Ortsraum
    for i in range(in_dim):
        ops.Dgate(x_data) | q[i]
    
    #Konstruiere Schichten des Circuits
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
    
    dictio["input"] = train_data_x

    
    # benutze den Tensorflowsimulator
    state = eng.run(qnn, args=dictio).state

    #Ortsprojektion und Varianz
    output, var = state.quad_expectation(0)
    
    error = tf.sqrt(var)
   
    #Größe die minimiert werden soll
    loss = tf.reduce_mean(tf.abs(output - train_data_y) ** 2)
    
    err = tf.reduce_mean(error)

    
    #Stelle sicher, dass der Trace des Outputs nahe bei 1 bleibt
    #Es wird also bestraft, wenn der Circuit Operationen benutzt
    #die für große Rechenfehler sorgen (dazu führen, dass der Anteil an höheren Fockstates zunimmt)
    trace = tf.abs(tf.reduce_mean(state.trace()))
    
    cost = loss + reg * (tf.abs(trace - 1) ** 2) + reg_err * err

    return cost, loss, trace, output




#==============================================================
#                           Training
#==============================================================
history = []
start_time = time.time()

#Nutze einen Optimierer von Tensorflow. Genauer gesagt: Adam (arXiv:1412.6980v9)
opt= tf.keras.optimizers.Adam(learning_rate=lr)

for i in range(epochs):
        
    # wenn das Programm gelaufen ist, dann resete die Engine
    if eng.run_progs:
        eng.reset()
        
    with tf.GradientTape() as tape:
        cost, loss, trace, output = costfunc(weights)

    gradients = tape.gradient(cost, weights)
    opt.apply_gradients(zip([gradients], [weights]))
    
    history.append(loss)
    
    # alle 10 Schritte 
    if i % 10 == 0:
        print("Epochen: {} Gesamtkosten: {:.4f} Loss: {:.4f} Trace: {:.4f}".format(i, cost, loss, trace))  
        
        
        x = np.linspace(a, b, 200)
        
        fig, ax = plt.subplots(1,1)
        
        # Funktion in Schwarz
        ax.plot(x, f(x,0), color='black', zorder=1, linewidth=2)
        
        # Trainingsdaten in Rot
        ax.scatter(train_data_x, train_data_y, color='red', marker='o', zorder=2, s=20)
        
        # Vorhersagen in Blau
        ax.scatter(train_data_x, output, color='blue', marker='x', zorder=3, s=20)
        
        # Achsenbeschriftung
        ax.set_xlabel('Input', fontsize=18)
        ax.set_ylabel('Output', fontsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        
        # erstelle Bilddateinamen
        name=ordner+str(i)+".png"
        fig.savefig(name, format='png', bbox_inches='tight')
        plt.close(fig)

        
        
#Gebe die Dauer des Trainings aus   
end_time = time.time()        
print("Dauer: ",np.round(end_time-start_time),"Sekunden") 

#speichere die Gewichte ab
np.save("weights",weights)
eng.reset()


#Erstelle einen Plot des Trainingsverlaufes
plt.plot(history)
plt.ylabel('Kosten')
plt.xlabel('Epoche')
plt.show()

#==============================================================
#                           Test
#==============================================================

#Lade die trainierten Parameter
weights=np.load("weights.npy")

# Führe das Programm mit nicht gelernten Test-Daten durch
dictio = {}
for symb, var in zip(params.flatten(), tf.reshape(weights, -1)):
    dictio[symb.name] = var
        
dictio["input"] = test_data_x


# benutze den Tensorflowsimulator
state = eng.run(qnn, args=dictio).state

#speichere Ortserwartungswert nach Ablauf des Programms
output = state.quad_expectation(0)[0]


x = np.linspace(a, b, 200)       

fig, ax = plt.subplots(1,1)

# Funktion in Schwarz
ax.plot(x, f(x,0), color='black', zorder=1, linewidth=2)

# Trainingsdaten in Rot
ax.scatter(train_data_x, train_data_y, color='red', marker='o', zorder=2, s=20)

# Vorhersagen in Blau
ax.scatter(test_data_x, output, color='blue', marker='x', zorder=3, s=20)

ax.set_xlabel('Input', fontsize=18)
ax.set_ylabel('Output', fontsize=18)
ax.tick_params(axis='both', which='minor', labelsize=16)
name = ordner+"Test.png"
fig.savefig(name, format='png', bbox_inches='tight')

#==============================================================
#                           Klasssisches Netz
#==============================================================

from tensorflow.keras import layers

#Erstelle ein klassisches neuronales Netz mit Tensorflow
def build_model(z):
    model = tf.keras.Sequential([
        layers.Dense(z, activation="relu"),
        layers.Dense(9, activation="relu"),
        layers.Dense(9, activation="relu"),
        layers.Dense(9, activation="relu"),
        layers.Dense(9, activation="relu"),
        layers.Dense(z)
    ])
    
    #Verwende ADAM als Optimierer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr*0.1)
    
    #mse = Mittlerer quadrierter Fehler als Kostenfunktion
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model
#Erstelle ein Netz mit einem Input-Neuron
model= build_model(1)
#Trainiere das Netz für 1000 Epochen
history = model.fit(train_data_x+1, train_data_y, epochs=1000, validation_split=0.2, verbose=0)

#Teste es an den nicht gelernten Test-Daten
predict = model.predict(test_data_x+1, verbose=0)

    
#Erstelle einen Plot
fig, ax = plt.subplots(1,1)

# Funktion in Schwarz
ax.plot(x, f(x,0), color='black', zorder=1, linewidth=2)

# Trainingsdaten in Rot
ax.scatter(train_data_x, train_data_y, color='red', marker='o', zorder=2, s=20)

# Vorhersagen in Blau
ax.scatter(test_data_x, predict, color='blue', marker='x', zorder=3, s=20)

ax.set_xlabel('Input', fontsize=18)
ax.set_ylabel('Output', fontsize=18)
ax.tick_params(axis='both', which='minor', labelsize=16)
name = ordner+"Test_klassisch.pdf"
fig.savefig(name, format='pdf', bbox_inches='tight')


    
    