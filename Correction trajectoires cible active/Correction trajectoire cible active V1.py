# -*- coding: utf-8 -*-
"""
Created on 2024-12-11 

Cas d'utilisation : 
    Cet algorithme est utilisé pour corriger une trajectoire mesurée par Leica Tracker
    avec une cible active. La correction consiste à repositionner la trajectoire au
    point de contrôle du robot (en général, milieu de l'essieu arrière projeté au sol)
    Un jour en instrumentant, on pourra corriger la correction avec l'inclinaison du robot
    
    ATTENTION :
        bien transférer toutes les infos comme le timestamp
    
    EN SORTIE :
        csv contenant la trajectoire en coordonnées tracker et la traj au point
        de contrôle du robot.

Note de version :
    v1 : 
        Draft / première version
        import depuis Tracker Pilot. Reste à faire : import depuis d'autres fichiers

@author: cmoignard
"""
#%% Libraries


import pandas as pd
import numpy as np
# import os.path
# import glob 
# from pathlib import Path
# import tkinter
# from tkinter import *
# from tkinter import filedialog
# import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
# from time import sleep
# from tqdm import tqdm
# from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.stats import linregress
import scipy.stats
from datetime import datetime

#%% variables globales / paramétrage du calcul

Fichier_Traj_Robot = "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 1.csv"
# Fichier_Obstacles = ""

# liste de colonnes à enregistrer
Liste_Col = ["DateTime [s]", "Curviligne [m]", 
            "X [m]", "Y [m]", "Z [m]", 
            "X_cible [m]", "Y_cible [m]", "Z_cible [m]", 
            "Cap [rad]", "Vitesse [m/s]", 
            ]

# Types de fichiers de Trajectoire Mère
Fichier_Traj_Robot_Direct_Changeur_Coord = False # si le fichier de la traj mère vient directement du transfert de coordonnées
Fichier_Traj_Robot_Tracker_Pilot = True # si le fichier de la traj mère vient du tracker pilot
Fichier_Traj_Robot_Spatial_Analyser = False # si le fichier de la traj mère vient de Spatial Analyser

# Types de fichiers de localisation d'obstacles
Fichier_Obstacles_Direct_Changeur_Coord = False # si le fichier de traj fille vient directement du transfert de coordonnées
Fichier_Obstacles_Tracker_Pilot = True # si le fichier de traj fille vient du tracker pilot
Fichier_Obstacles_Spatial_Analyser = False # si le fichier de traj fille vient de Spatial Analyser

# Paramétrage du calcul
Incertitude = False # True pour calculer la propagation des incertitudes
Param_MC = 10 # nombre d'itération pour le calcul de propagation des incertitudes par la méthode Monte-Carlo
Param_Filtrage_Cap = 100 # nombre de valeurs pour la moyenne glissantes pour filtrer le cap

# Paramétrage du décalage = position de la cible active dans le repère du robot
Corr_X = 1. # en [m]
Corr_Y = 1. # en [m]
Corr_Z = 1. # en [m]
Corr_inclinaison = False # true pour prendre en compte l'inclinaison du robot pour les corrections


#%% Ouverture des fichiers / mise en forme des données

if Fichier_Traj_Robot_Direct_Changeur_Coord == True :
    traj_robot = pd.read_csv(Fichier_Traj_Robot, delimiter=";", decimal=",", index_col=[0]) # le fichier contient les incertitudes (normalement), même si elles sont nulles et plein d'autres infos, tout doit être transféré (on ajoute des infos dans le csv, on n'enlève rien)
elif Fichier_Traj_Robot_Tracker_Pilot == True :
    traj_robot = pd.read_csv(Fichier_Traj_Robot, sep=";", decimal=",", encoding="cp1252", skiprows=[0], header=None, skipinitialspace=1, usecols=[1, 2, 3, 5, 6, 7, 12]) # fichier direct Tracker Pilot en mm
    traj_robot_bis = traj_robot.rename(columns={1:"X [m]", 2:"Y [m]", 3:"Z [m]", 5:"U95x", 6:"U95y", 7:"U95z", 12:"HMS"})
    traj_robot_bis[["X [m]", "Y [m]", "Z [m]"]] = traj_robot_bis[["X [m]", "Y [m]", "Z [m]"]].div(1000.)
    traj_robot_bis["sigmaX"] = traj_robot_bis["U95x"] / 2000.
    traj_robot_bis["sigmaY"] = traj_robot_bis["U95y"] / 2000.
    traj_robot_bis["sigmaZ"] = traj_robot_bis["U95z"] / 2000.
    traj_robot_bis["sigma"] = traj_robot_bis[["sigmaX", "sigmaY", "sigmaZ"]].max(axis=1)
    traj_robot_bis["DateTime [s]"] = pd.to_datetime(traj_robot_bis["HMS"], dayfirst = True)
    traj_robot = traj_robot_bis[["X [m]", "Y [m]", "Z [m]", "sigmaX", "sigmaY", "sigmaZ", "sigma", "DateTime [s]"]]
    # rajouter les incertitudes
elif Fichier_Traj_Robot_Spatial_Analyser == True :
    traj_robot = pd.read_csv(Fichier_Traj_Robot, encoding="cp1251", delimiter=";", decimal=",", usecols={1, 2, 3}) # à écrire
# rajouter les imports depuis fichiers GNSS, RosBag et DataFrame


# A partir d'ici, les dataframes sont tous les mêmes et contiennent : trajectoires ou position d'obstacles (X, Y, Z)
# Timestamp, incertitudes
# Pour ré-importer un fichier déjà traité : pd.read_csv(Nom_De_Fichier, delimiter=";", decimal=",", index_col=[0])

"""
Data columns (total 8 columns):
 #   Column        Non-Null Count   Dtype         
---  ------        --------------   -----         
 0   X [m]         214144 non-null  float64       
 1   Y [m]         214144 non-null  float64       
 2   Z [m]         214144 non-null  float64       
 3   sigmaX        214144 non-null  float64       
 4   sigmaY        214144 non-null  float64       
 5   sigmaZ        214144 non-null  float64       
 6   sigma         214144 non-null  float64       
 7   DateTime [s]  214144 non-null  datetime64[ns]
dtypes: datetime64[ns](1), float64(7)
"""
#%% Calcul du repère mobile

# décalage en moins = vers l'avenir
s1 = pd.Series(traj_robot["X [m]"].shift(periods=-1), name="X+1 [m]")
s2 = pd.Series(traj_robot["Y [m]"].shift(periods=-1), name="Y+1 [m]")
s3 = pd.Series(traj_robot["Z [m]"].shift(periods=-1), name="Z+1 [m]")
s4 = pd.Series(traj_robot["DateTime [s]"].shift(periods=1), name="DateTime+1 [s]")
traj_robot = pd.concat([traj_robot, s1, s2, s3, s4], axis=1)

# calcul du cap vers l'avenir
traj_robot["delta_Y+ [m]"] = traj_robot["Y+1 [m]"] - traj_robot["Y [m]"]
traj_robot["delta_X+ [m]"] = traj_robot["X+1 [m]"] - traj_robot["X [m]"]
traj_robot["Cap [rad]"] = np.atan2(traj_robot["delta_Y+ [m]"].to_numpy(), traj_robot["delta_X+ [m]"].to_numpy())

# filtrage du cap
segment = range(-Param_Filtrage_Cap//2, Param_Filtrage_Cap//2)
filtrage_cap = traj_robot["Cap [rad]"].shift(periods=segment)
traj_robot["Cap_moy [rad]"] = filtrage_cap.mean(axis="columns")


# décalage en plus = dans le passé
s1 = pd.Series(traj_robot["X [m]"].shift(periods=1), name="X-1 [m]")
s2 = pd.Series(traj_robot["Y [m]"].shift(periods=1), name="Y-1 [m]")
s3 = pd.Series(traj_robot["Z [m]"].shift(periods=1), name="Z-1 [m]")
s4 = pd.Series(traj_robot["DateTime [s]"].shift(periods=1), name="DateTime-1 [s]")
traj_robot = pd.concat([traj_robot, s1, s2, s3, s4], axis=1)

# calcul de la distance et abcisse curviligne (dans le passé)
traj_robot["delta_Y- [m]"] = traj_robot["Y-1 [m]"] - traj_robot["Y [m]"]
traj_robot["delta_X- [m]"] = traj_robot["X-1 [m]"] - traj_robot["X [m]"]
traj_robot["Distance [m]"] = ((traj_robot["delta_X- [m]"])**2. + (traj_robot["delta_Y- [m]"])**2. )**0.5
traj_robot["Curviligne [m]"] = traj_robot["Distance [m]"].cumsum()

# calcul de la vitesse (dans le passé)
s1 = pd.Series(traj_robot["DateTime [s]"]-traj_robot["DateTime-1 [s]"], name="delta_T [s]")
traj_robot = pd.concat([traj_robot, s1.dt.total_seconds()], axis=1)
traj_robot["Vitesse [m/s]"] = traj_robot["Distance [m]"] / traj_robot["delta_T [s]"]

#%% correction de la position de la cible active dans le repère du robot
# correction sans inclinaison (= correction selon le cap seulement)
traj_robot["Z_robot [m]"] = traj_robot["Z [m]"] - Corr_Z
traj_robot["X_robot [m]"] = traj_robot["X [m]"] - Corr_X * np.cos(traj_robot["Cap_moy [rad]"]) + Corr_Y * np.sin(traj_robot["Cap_moy [rad]"])
traj_robot["Y_robot [m]"] = traj_robot["Y [m]"] - Corr_X * np.sin(traj_robot["Cap_moy [rad]"]) - Corr_Y * np.cos(traj_robot["Cap_moy [rad]"])
traj_robot = traj_robot.rename({"X [m]":"X_cible [m]", "Y [m]":"Y_cible [m]", "Z [m]":"Z_cible [m]"}, axis="columns")
traj_robot = traj_robot.rename({"X_robot [m]":"X [m]", "Y_robot [m]":"Y [m]", "Z_robot [m]":"Z [m]"}, axis="columns")


    
# enregistrement du fichier
nom_de_fichier = Fichier_Traj_Robot.replace(".csv", "_").replace(".txt", "_") + "corr_centre_robot_" + datetime.now().isoformat(timespec='seconds').replace(":", "-") + ".csv"
traj_robot[Liste_Col].to_csv(nom_de_fichier, sep=";", decimal=",", index=True)

plot1 = traj_robot[(traj_robot["Curviligne [m]"] > 1.) & (traj_robot["Curviligne [m]"] < 100.)].plot.scatter(x="X [m]", y="Y [m]", c='Curviligne [m]', colormap='viridis', s=1)
traj_robot[(traj_robot["Curviligne [m]"] > 1.) & (traj_robot["Curviligne [m]"] < 100.)].plot.scatter(x="X_cible [m]", y="Y_cible [m]", c='Curviligne [m]', colormap='viridis', s=1, ax=plot1)



