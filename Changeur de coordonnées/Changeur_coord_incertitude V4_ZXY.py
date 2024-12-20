# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:16:53 2024

@author: cmoignard

maturation algorithme pour fitter au mieux un ensemble de points. Par exemple, une dizaine de points mesurés avec canne d'arpentage
et avec tracker. L'objectif est de trouver la fonction de changement de repère (translation et rotation) 
qui correspond le mieux. Et d'estimer l'incertitude liée à cette fonction de changement de repère, c'est à dire :
    pour une nouvelle coordonnée dans un repère, en calculant les coordonnées dans l'autre repère, quelle serait l'incertitude
    sur ces nouvelles coordonnées

Vigilance : 


Notes de version : première version. Calcul changement de coordonnées + calcul d'incertitude sur chaque coordonnée
                    Ajout d'une section spéciale "calcul inverse seulement"
                    Ajout de paramétrage d'import en fonction du logiciel d'export
                    Ajout de paramétres pour le calcul
                    Correction des fonctions TransRotInv et TransRotInv2 (erreur dans le signe des angles)
                    Ajout du transfert du DateTime donné par Tracker dans le cas de transfert vers GNSS
                        (utilisation pour comparer la loc robot avec la loc tracker)
"""

#%% # Imports et paramètres de l'algorithme

import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
import random

methode = "SLSQP" # "Powell" semble OK, "L-BFGS-B" part en live avec certains scénarii, "SLSQP" semble OK sur 10 scénarii, 4* + rapide que Powell
Nombre_It = 100 # nombre d'itération pour la méthode Montécarlo (calcul d'incertitude par bruit gaussien sur les variables)
Erreur_GPS = 0.01 # erreur de mesure GPS par défaut [mètre]
nombre_rep = 1 # nbre de répétition de mesure effectuée dans le repère R1 (= canne d'arpentage GNSS lambert étendu CC46)

GNSS_to_Tracker = False # transfert des coord de GNSS vers Tracker
Tracker_to_GNSS = True # transfert des coord de Tracker vers GNSS

Tracker_Pilot = True # export de la trajectoire avec Tracker Pilot
Spatial_Analyser = False # export de la trajectoire avec Spatial Analyser

Incertitude = False # true pour calculer l'incertitude, false pour un transfert simple

Affichage = False # True pour créer un graphique de la trajectoire

Fichier_R1 = "" # fichier contenant les coordonnées des points dans le repère 1 / "GNSS"

Liste_Fichiers_R2 = ["C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 1.csv", 
                     "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 2.csv", 
                     "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 3.csv", 
                     "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 4.csv", 
                     "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 5.csv", 
                     "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 6.csv", 
                     "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj fille 7.csv", 
                     "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Trajectoires/traj mere a la main.csv", 
                     ]

# Fichier_R2 = "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC1/OMESRO/Changement coordonnées pour illustration/suivi traj_20231220_tracker/m14_20231220_1629.csv" # fichier contenant les coordonnées dans le repère 2 / "Tracker"
Fichier_changeur = "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/TR_ZXY_GNSSS_Tracker_SLSQP_10000mc_[]sup_2024-12-04T10-32-16.csv" # fichier contenant les paramètres du changeur de coordonnées
unite = {"X":"X [m]", "Y":"Y [m]", "Z":"Z [m]", "sigma":"sigma [m]", "sigmaX":"sigmaX [m]", "sigmaY":"sigmaY [m]", "sigmaZ":"sigmaZ [m]", }

start = time.time()

#%% # définir la fonction de changement de repère vers R2 // DataFrame
# ATTENTION : formulation non canonique : la translation est effectuée avant la rotation. Donc, les paramètres de translation
# s'écrivent dans le repère 1 avant transfert dans l'orientation du repère 2

def TransRot (liste_points_rep1, parametres) :
    """
    transfère les coordonnées d'une liste de points depuis le repère 1 vers le repère 2
    sans les incertitudes
    
    Parameters
    ----------
    liste_points_rep1 : pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma) 
        exprimées dans le repère 1.
        Coordonnées + incertitude -> float en mètre
    parametres : pd.DataFrame. DF de 6 paramètres + 6 incertitudes (1 sigma).
        entree = deltaX, deltaY, deltaZ, alpha, beta, gamma en mètre et radian
        entree = valeurs du changeur de coord de Repère 1 vers Repère 2 en mètre et en radian
        delta = translation dans le repère 1 (GNSS) AVANT LA ROTATION
        Alpha = rotation autour de Z
        Beta = rotation autour de X
        Gamma = rotation autour de Y
                  DeltaX [m]  DeltaY [m]  DeltaZ [m]  Alpha(Z) [rad]  Beta(X) [rad]  Gamma(Y) [rad]
         Valeurs  float       float       float       float           float          float 
         Sigma    float       float       float       float           float          float

    Returns pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma)
        exprimées dans le répère 2
                  X           Y           Z           sigmaX   sigmaY   sigmaZ
         0        float       float       float       float    float    float           
         1        float       float       float       float    float    float    
         ...

    -------
    
    """
    
    deltaX = parametres.loc["Valeurs", "DeltaX [m]"]
    deltaY = parametres.loc["Valeurs", "DeltaY [m]"]
    deltaZ = parametres.loc["Valeurs", "DeltaZ [m]"]
    alpha  = parametres.loc["Valeurs", "Alpha(Z) [rad]"] # euler autour de Z
    beta   = parametres.loc["Valeurs", "Beta(X) [rad]"] # euler autour de X prime
    gamma  = parametres.loc["Valeurs", "Gamma(Y) [rad]"] # euler autour de Y prime prime
    
    # Translation dans le repère 1 (GNSS)
    x_0 = liste_points_rep1[["X [m]"]] + deltaX
    y_0 = liste_points_rep1[["Y [m]"]] + deltaY
    z_0 = liste_points_rep1[["Z [m]"]] + deltaZ
    
    # rotation autour de Z (alpha)
    x_1 = x_0 * math.cos(alpha) - y_0.to_numpy() * math.sin(alpha)
    y_1 = x_0.to_numpy() * math.sin(alpha) + y_0 * math.cos(alpha)
    z_1 = z_0
    
    # rotation autour de X (beta)
    x_2 = x_1
    y_2 = y_1 * math.cos(beta) - z_1.to_numpy() * math.sin(beta)
    z_2 = y_1.to_numpy() * math.sin(beta) + z_1 * math.cos(beta)
    
    # rotation autour de Y (gamma)
    x_3 = z_2.to_numpy() * math.sin(gamma) + x_2 * math.cos(gamma)
    y_3 = y_2
    z_3 = z_2 * math.cos(gamma) - x_2.to_numpy() * math.sin(gamma)
    # Rajouter les sigmas comme NaN    
    return pd.concat([x_3, y_3, z_3], axis=1)
#%% # définir la fonction de changement de repère vers R2 // DataFrame
# ATTENTION : formulation non canonique : la translation est effectuée avant la rotation. Donc, les paramètres de translation
# s'écrivent dans le repère 1 avant transfert dans l'orientation du repère 2

def TransRot2 (liste_points_rep1) :
    """
    transfère les coordonnées d'une liste de points depuis le repère 1 vers le repère 2
    sans les incertitudes
    Pour calcul incertitudes via méthode MontéCarlo
    
    Parameters
    ----------
    liste_points_rep1 : pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 6 paramètres de changeur de coord 
        exprimées dans le repère 1 vers le repère 2
        Coordonnées -> float en mètre
        paramètres -> 6* float en mètre et radians
    
    Returns pd.DataFrame. DF de n vecteurs avec 3 coordonnées en mètre
        exprimées dans le répère 2
                  X           Y           Z    
         0        float       float       float          
         1        float       float       float
         ...

    -------
    
    """
    
    deltaX = liste_points_rep1[["DeltaX [m]"]].to_numpy()
    deltaY = liste_points_rep1[["DeltaY [m]"]].to_numpy()
    deltaZ = liste_points_rep1[["DeltaZ [m]"]].to_numpy()
    alpha  = liste_points_rep1[["Alpha(Z) [rad]"]].to_numpy() # euler autour de Z
    beta   = liste_points_rep1[["Beta(X) [rad]"]].to_numpy() # euler autour de X prime
    gamma  = liste_points_rep1[["Gamma(Y) [rad]"]].to_numpy() # euler autour de Y prime prime
    
    # Translation dans le repère 1 (GNSS)
    x_0 = liste_points_rep1[["X [m]"]] + deltaX
    y_0 = liste_points_rep1[["Y [m]"]] + deltaY
    z_0 = liste_points_rep1[["Z [m]"]] + deltaZ
    
    # rotation autour de Z (alpha)
    x_1 = x_0 * np.cos(alpha) - y_0.to_numpy() * np.sin(alpha)
    y_1 = x_0.to_numpy() * np.sin(alpha) + y_0 * np.cos(alpha)
    z_1 = z_0
    
    # rotation autour de X (beta)
    x_2 = x_1
    y_2 = y_1 * np.cos(beta) - z_1.to_numpy() * np.sin(beta)
    z_2 = y_1.to_numpy() * np.sin(beta) + z_1 * np.cos(beta)
    
    # rotation autour de Y (gamma)
    x_3 = z_2.to_numpy() * np.sin(gamma) + x_2 * np.cos(gamma)
    y_3 = y_2
    z_3 = z_2 * np.cos(gamma) - x_2.to_numpy() * np.sin(gamma)
    
    return pd.concat([x_3, y_3, z_3], axis=1)

#%% # définir la fonction de changement de repère vers R2 // DataFrame AVEC INCERTITUDE
# ATTENTION : formulation non canonique : la translation est effectuée avant la rotation. Donc, les paramètres de translation
# s'écrivent dans le repère 1 avant transfert dans le repère 2

def TransRotInc (liste_points_rep1, parametres, param_mc) :
    """
    transfère les coordonnées d'une liste de points depuis le repère 1 vers le repère 2
    sans les incertitudes
    
    Parameters
    ----------
    liste_points_rep1 : pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma) 
        exprimées dans le repère 1.
        Coordonnées + incertitude -> float en mètre
    parametres : pd.DataFrame. DF de 6 paramètres + 6 incertitudes (1 sigma).
        entree = deltaX, deltaY, deltaZ, alpha, beta, gamma en mètre et radian
        entree = valeurs du changeur de coord de Repère 1 vers Repère 2 en mètre et en radian
        delta = translation dans le repère 1 (GNSS) AVANT LA ROTATION
        Alpha = rotation autour de Z
        Beta = rotation autour de X
        Gamma = rotation autour de Y
                  DeltaX [m]  DeltaY [m]  DeltaZ [m]  Alpha(Z) [rad]  Beta(X) [rad]  Gamma(Y) [rad]
         Valeurs  float       float       float       float           float          float 
         Sigma    float       float       float       float           float          float

    Returns pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma)
        exprimées dans le répère 2
                  X           Y           Z           sigmaX   sigmaY   sigmaZ
         0        float       float       float       float    float    float           
         1        float       float       float       float    float    float    
         ...

    -------
    
    """
    
    # initialisation des variables de stockage
    liste_points_rep2 = liste_points_rep1[0:0]
        
    for index in liste_points_rep1.index : # choix d'un point / parcous tous les points 1 par 1
        # point_rep2 = liste_points_rep1[0:0]
        point_rep1 = liste_points_rep1[0:0]
        # sigma_1 = liste_points_rep1.loc[index, ["sigmaX", "sigmaY", "sigmaZ"]].max()
        s2 = pd.Series()
        # sigma_1 = liste_points_rep1[["sigma"]].max()
        # boucle MontéCarlo
        for i in range(param_mc) : # génération d'un scénario 
            # ici génération d'un scénario en ajoutant une part aléatoire distribution normale sur la valeur des coord du point
            # à modifier pour que la part aléatoire reflète l'incertitude de chaque coordonnée et non le max de toutes les coordonnées.
            # génération des coordonnées aléatoire autour du point "index"
            s2["X [m]"] = random.gauss(mu=liste_points_rep1.at[index,"X [m]"], sigma=liste_points_rep1.at[index,"sigmaX"])
            s2["Y [m]"] = random.gauss(mu=liste_points_rep1.at[index,"Y [m]"], sigma=liste_points_rep1.at[index,"sigmaY"])
            s2["Z [m]"] = random.gauss(mu=liste_points_rep1.at[index,"Z [m]"], sigma=liste_points_rep1.at[index,"sigmaZ"])
            s2.rename(i)
            #génération des parts aléatoires sur chaque paramètre du changeur.
            s2["DeltaX [m]"] = random.gauss(mu=parametres.at["Valeurs", "DeltaX [m]"], sigma=parametres.at["Sigma", "DeltaX [m]"])
            s2["DeltaY [m]"] = random.gauss(mu=parametres.at["Valeurs", "DeltaY [m]"], sigma=parametres.at["Sigma", "DeltaY [m]"])
            s2["DeltaZ [m]"] = random.gauss(mu=parametres.at["Valeurs", "DeltaZ [m]"], sigma=parametres.at["Sigma", "DeltaZ [m]"])
            s2["Alpha(Z) [rad]"] = random.gauss(mu=parametres.at["Valeurs", "Alpha(Z) [rad]"], sigma=parametres.at["Sigma", "Alpha(Z) [rad]"])
            s2["Beta(X) [rad]"] = random.gauss(mu=parametres.at["Valeurs", "Beta(X) [rad]"], sigma=parametres.at["Sigma", "Beta(X) [rad]"])
            s2["Gamma(Y) [rad]"] = random.gauss(mu=parametres.at["Valeurs", "Gamma(Y) [rad]"], sigma=parametres.at["Sigma", "Gamma(Y) [rad]"])
            point_rep1 = pd.concat([point_rep1, s2.to_frame().T])
        # lancement transfert des coord du point "index"
        point_rep2 = TransRot2(point_rep1)
        # conversion des point_rep2 en moy + std
        moy = point_rep2[["X [m]", "Y [m]", "Z [m]"]].mean(axis=0)
        std = point_rep2[["X [m]", "Y [m]", "Z [m]"]].std(axis=0).rename({"X [m]": "sigmaX", "Y [m]": "sigmaY", "Z [m]": "sigmaZ"})
        new_line = pd.concat([moy, std]).rename(index)
        liste_points_rep2 = pd.concat([liste_points_rep2, new_line.to_frame().T], ignore_index=False)
    
    liste_points_rep2["sigma"] = liste_points_rep2[["sigmaX", "sigmaY", "sigmaZ"]].max(axis=1)
    
    return liste_points_rep2

#%% # définir la fonction de changement de repère de R2 vers R1, fonction inverse
# ATTENTION : formulation non canonique, pour plus d'info, voir def TransRot

def TransRotInv (liste_points_rep2, parametres) :
    """
    transfère les coordonnées d'une liste de points depuis le repère 2 vers le repère 1
    sans les incertitudes
    
    Parameters
    ----------
    liste_points_rep1 : pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma) 
        exprimées dans le repère 1.
        Coordonnées + incertitude -> float en mètre
    parametres : pd.DataFrame. DF de 6 paramètres + 6 incertitudes (1 sigma).
        entree = deltaX, deltaY, deltaZ, alpha, beta, gamma en mètre et radian
        entree = valeurs du changeur de coord de Repère 1 vers Repère 2 en mètre et en radian
        delta = translation dans le repère 1 (GNSS) AVANT LA ROTATION
        Alpha = rotation autour de Z
        Beta = rotation autour de X
        Gamma = rotation autour de Y
                  DeltaX [m]  DeltaY [m]  DeltaZ [m]  Alpha(Z) [rad]  Beta(X) [rad]  Gamma(Y) [rad]
         Valeurs  float       float       float       float           float          float 
         Sigma    float       float       float       float           float          float

    Returns pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma)
        exprimées dans le répère 2
                  X           Y           Z           sigma
         0        float       float       float       float           
         1        float       float       float       float
         ...

    -------
    
    """
    
    deltaX = parametres.loc["Valeurs", "DeltaX [m]"]
    deltaY = parametres.loc["Valeurs", "DeltaY [m]"]
    deltaZ = parametres.loc["Valeurs", "DeltaZ [m]"]
    alpha  = parametres.loc["Valeurs", "Alpha(Z) [rad]"] # euler autour de Z
    beta   = parametres.loc["Valeurs", "Beta(X) [rad]"] # euler autour de X prime
    gamma  = parametres.loc["Valeurs", "Gamma(Y) [rad]"] # euler autour de Y prime prime
    """
    sigma_X = parametres.loc["DeltaX [m]", "Sigma"]
    sigma_Y = parametres.loc["DeltaY [m]", "Sigma"]
    sigma_Z = parametres.loc["DeltaZ [m]", "Sigma"]
    sigma_a  = parametres.loc["Alpha(Z) [rad]", "Sigma"] # euler autour de Z
    sigma_b   = parametres.loc["Beta(X) [rad]", "Sigma"] # euler autour de X prime
    sigma_g  = parametres.loc["Gamma(Y) [rad]", "Sigma"] # euler autour de Y prime prime
    """
    
    x_0 = liste_points_rep2[["X [m]"]]
    y_0 = liste_points_rep2[["Y [m]"]]
    z_0 = liste_points_rep2[["Z [m]"]]
    
    # rotation autour de Y (-gamma)
    x_1 = -1. * z_0.to_numpy() * math.sin(gamma) + x_0 * math.cos(gamma) #corr
    y_1 = y_0
    z_1 = z_0 * math.cos(gamma) + x_0.to_numpy() * math.sin(gamma) #corr
    
    # rotation autour de X (-beta)
    x_2 = x_1
    y_2 = y_1 * math.cos(beta) + z_1.to_numpy() * math.sin(beta) #corr
    z_2 = -1. * y_1.to_numpy() * math.sin(beta) + z_1 * math.cos(beta) #corr
    
    # rotation autour de Z (-alpha)
    x_3 = x_2 * math.cos(alpha) + y_2.to_numpy() * math.sin(alpha) #corr
    y_3 = -1. * x_2.to_numpy() * math.sin(alpha) + y_2 * math.cos(alpha) #corr
    z_3 = z_2

    # Translation dans le repère 1 (GNSS)
    x_4 = x_3 - deltaX
    y_4 = y_3 - deltaY
    z_4 = z_3 - deltaZ
    
    return pd.concat([x_4, y_4, z_4], axis=1)

#%% # définir la fonction de changement de repère vers R2 // DataFrame
# ATTENTION : formulation non canonique : la translation est effectuée avant la rotation. Donc, les paramètres de translation
# s'écrivent dans le repère 1 avant transfert dans l'orientation du repère 2

def TransRotInv2 (liste_points_rep2) :
    """
    transfère les coordonnées d'une liste de points depuis le repère 2 vers le repère 1
    pour les incertitudes
    Pour calcul incertitudes via méthode MontéCarlo
    
    Parameters
    ----------
    liste_points_rep2 : pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 6 paramètres de changeur de coord 
        exprimées dans le repère 2 vers le repère 1
        Coordonnées -> float en mètre
        paramètres -> 6* float en mètre et radians
    
    Returns pd.DataFrame. DF de n vecteurs avec 3 coordonnées en mètre
        exprimées dans le répère 2
                  X           Y           Z    
         0        float       float       float          
         1        float       float       float
         ...

    -------
    
    """
    
    deltaX = liste_points_rep2[["DeltaX [m]"]].to_numpy()
    deltaY = liste_points_rep2[["DeltaY [m]"]].to_numpy()
    deltaZ = liste_points_rep2[["DeltaZ [m]"]].to_numpy()
    alpha  = liste_points_rep2[["Alpha(Z) [rad]"]].to_numpy() # euler autour de Z
    beta   = liste_points_rep2[["Beta(X) [rad]"]].to_numpy() # euler autour de X prime
    gamma  = liste_points_rep2[["Gamma(Y) [rad]"]].to_numpy() # euler autour de Y prime prime
    
    # Initialisation
    x_0 = liste_points_rep2[["X [m]"]]
    y_0 = liste_points_rep2[["Y [m]"]]
    z_0 = liste_points_rep2[["Z [m]"]]
    
    # rotation autour de Y (-gamma)
    x_1 = -1. * z_0.to_numpy() * np.sin(gamma) + x_0 * np.cos(gamma)  #corr
    y_1 = y_0
    z_1 = z_0 * np.cos(gamma) + x_0.to_numpy() * np.sin(gamma) #corr
    
    # rotation autour de X (-beta)
    x_2 = x_1
    y_2 = y_1 * np.cos(beta) + z_1.to_numpy() * np.sin(beta) #corr
    z_2 = -1. * y_1.to_numpy() * np.sin(beta) + z_1 * np.cos(beta) #corr
    
    # rotation autour de Z (-alpha)
    x_3 = x_2 * np.cos(alpha) + y_2.to_numpy() * np.sin(alpha) #corr
    y_3 = -1. * x_2.to_numpy() * np.sin(alpha) + y_2 * np.cos(alpha) #corr
    z_3 = z_2
    
    # Translation dans le repère 1 (GNSS)
    x_4 = x_3 - deltaX
    y_4 = y_3 - deltaY
    z_4 = z_3 - deltaZ
     
    return pd.concat([x_4, y_4, z_4], axis=1)

#%% # définir la fonction de changement de repère vers R2 // DataFrame AVEC INCERTITUDE
# ATTENTION : formulation non canonique : la translation est effectuée avant la rotation. Donc, les paramètres de translation
# s'écrivent dans le repère 1 avant transfert dans le repère 2

def TransRotInvInc (liste_points_rep2, parametres, param_mc) :
    """
    transfère les coordonnées d'une liste de points depuis le repère 2 vers le repère 1
    AVEC les incertitudes
    
    Parameters
    ----------
    liste_points_rep2 : pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma) 
        exprimées dans le repère 1.
        Coordonnées + incertitude -> float en mètre
    parametres : pd.DataFrame. DF de 6 paramètres + 6 incertitudes (1 sigma).
        entree = deltaX, deltaY, deltaZ, alpha, beta, gamma en mètre et radian
        entree = valeurs du changeur de coord de Repère 1 vers Repère 2 en mètre et en radian
        delta = translation dans le repère 1 (GNSS) AVANT LA ROTATION
        Alpha = rotation autour de Z
        Beta = rotation autour de X
        Gamma = rotation autour de Y
                  DeltaX [m]  DeltaY [m]  DeltaZ [m]  Alpha(Z) [rad]  Beta(X) [rad]  Gamma(Y) [rad]
         Valeurs  float       float       float       float           float          float 
         Sigma    float       float       float       float           float          float

    Returns pd.DataFrame. DF de n vecteurs avec 3 coordonnées + 3 incertitudes (1 sigma)
        exprimées dans le répère 2
                  X           Y           Z           sigmaX   sigmaY   sigmaZ
         0        float       float       float       float    float    float           
         1        float       float       float       float    float    float    
         ...

    -------
    
    """
    
    # initialisation des variables de stockage
    liste_points_rep1 = liste_points_rep2[0:0]
        
    for index in liste_points_rep2.index : # choix d'un point / parcous tous les points 1 par 1
        # point_rep2 = liste_points_rep1[0:0]
        point_rep2 = liste_points_rep2[0:0]
        # sigma_1 = liste_points_rep1.loc[index, ["sigmaX", "sigmaY", "sigmaZ"]].max()
        s2 = pd.Series()
        # sigma_1 = liste_points_rep1[["sigma"]].max()
        # boucle MontéCarlo
        for i in range(param_mc) : # génération d'un scénario 
            # ici génération d'un scénario en ajoutant une part aléatoire distribution normale sur la valeur des coord du point
            # à modifier pour que la part aléatoire reflète l'incertitude de chaque coordonnée et non le max de toutes les coordonnées.
            # génération des coordonnées aléatoire autour du point "index"
            s2["X [m]"] = random.gauss(mu=liste_points_rep2.at[index,"X [m]"], sigma=liste_points_rep2.at[index,"sigmaX"])
            s2["Y [m]"] = random.gauss(mu=liste_points_rep2.at[index,"Y [m]"], sigma=liste_points_rep2.at[index,"sigmaY"])
            s2["Z [m]"] = random.gauss(mu=liste_points_rep2.at[index,"Z [m]"], sigma=liste_points_rep2.at[index,"sigmaZ"])
            s2.rename(i)
            #génération des parts aléatoires sur chaque paramètre du changeur.
            s2["DeltaX [m]"] = random.gauss(mu=parametres.at["Valeurs", "DeltaX [m]"], sigma=parametres.at["Sigma", "DeltaX [m]"])
            s2["DeltaY [m]"] = random.gauss(mu=parametres.at["Valeurs", "DeltaY [m]"], sigma=parametres.at["Sigma", "DeltaY [m]"])
            s2["DeltaZ [m]"] = random.gauss(mu=parametres.at["Valeurs", "DeltaZ [m]"], sigma=parametres.at["Sigma", "DeltaZ [m]"])
            s2["Alpha(Z) [rad]"] = random.gauss(mu=parametres.at["Valeurs", "Alpha(Z) [rad]"], sigma=parametres.at["Sigma", "Alpha(Z) [rad]"])
            s2["Beta(X) [rad]"] = random.gauss(mu=parametres.at["Valeurs", "Beta(X) [rad]"], sigma=parametres.at["Sigma", "Beta(X) [rad]"])
            s2["Gamma(Y) [rad]"] = random.gauss(mu=parametres.at["Valeurs", "Gamma(Y) [rad]"], sigma=parametres.at["Sigma", "Gamma(Y) [rad]"])
            point_rep2 = pd.concat([point_rep2, s2.to_frame().T])
        # lancement transfert des coord du point "index"
        point_rep1 = TransRotInv2(point_rep2)
        # conversion des point_rep1 en moy + std
        moy = point_rep1[["X [m]", "Y [m]", "Z [m]"]].mean(axis=0)
        std = point_rep1[["X [m]", "Y [m]", "Z [m]"]].std(axis=0).rename({"X [m]": "sigmaX", "Y [m]": "sigmaY", "Z [m]": "sigmaZ"})
        new_line = pd.concat([moy, std]).rename(index)
        liste_points_rep1 = pd.concat([liste_points_rep1, new_line.to_frame().T], ignore_index=False)
    
    liste_points_rep1["sigma"] = liste_points_rep1[["sigmaX", "sigmaY", "sigmaZ"]].max(axis=1)
    
    return liste_points_rep1

#%% # ouvrir le fichier des paramètres du changeur de coordonnées
# nom_de_fichier = "Changeur de coord_ZXY_" + methode + "_" + str(nombre_it) + "_scenarii_" + str(time.time()) + ".csv"
Solution = pd.read_csv(Fichier_changeur, sep=";", decimal=",", index_col=[0])

#%% # changement de coordonées de R1 vers R2 avec calcul d'incertitude par méthode de MontéCarlo

if GNSS_to_Tracker == True :
    start = time.time()
    # ouverture du fichier GNSS
    liste_points_R1 = pd.read_csv(Fichier_R1, sep=" ", decimal=".", skiprows=[0, 1, 2, 3], header=None, skipinitialspace=1, usecols=[1, 2, 3, 4, 5]) # fichier direct export du Trimble en mètre lambert CC46
    liste_points_R1 = liste_points_R1.rename(columns={1:"X [m]", 2:"Y [m]", 3:"Z [m]", 4:"HzPrec", 5:"VePrec"})
    # renomage et canonisation des valeurs
    liste_points_R1["sigma"] = liste_points_R1[["HzPrec", "VePrec"]].max(axis=1)/2. # formule simplifiée, peut-être on pourra améliorer
    liste_points_R1["sigmaX"] = liste_points_R1["HzPrec"]/2.
    liste_points_R1["sigmaY"] = liste_points_R1["HzPrec"]/2.
    liste_points_R1["sigmaZ"] = liste_points_R1["VePrec"]/2.
    # combinaison des incertitudes si plusieurs mesure des points
    if nombre_rep > 1 :
        # reshaping le tableau de points GPS en supposant qu'on a fait nombre_rep de mesures par point tracker
        ligne = len(liste_points_R1.index)
        nb_pt = ligne//nombre_rep
        liste_points_R1_bis = liste_points_R1[0:0]
        for index in range(nb_pt) : # regrouper les mesures GPS par points tracker
            selection = liste_points_R1[["X [m]", "Y [m]", "Z [m]", "sigmaX", "sigmaY", "sigmaZ"]].iloc[index*nombre_rep:nombre_rep*(index+1)]
            # sel = selection[["X [m]", "Y [m]", "Z [m]"]].mean()
            new_line = selection[["X [m]", "Y [m]", "Z [m]"]].mean().rename(index)
            new_line["sigmaX"] = (selection["X [m]"].std()**2 + pd.DataFrame.sum(selection["sigmaX"]**2))**0.5 # calcul de somme d'écart-type sur X
            new_line["sigmaY"] = (selection["Y [m]"].std()**2 + pd.DataFrame.sum(selection["sigmaY"]**2))**0.5 # calcul de somme d'écart-type sur Y
            new_line["sigmaZ"] = (selection["Z [m]"].std()**2 + pd.DataFrame.sum(selection["sigmaZ"]**2))**0.5 # calcul de somme d'écart-type sur Z
            liste_points_R1_bis = pd.concat([liste_points_R1_bis, new_line.to_frame().T], ignore_index=False)
        liste_points_R1_bis["sigma"] = liste_points_R1_bis[["sigmaX", "sigmaY", "sigmaZ"]].max(axis=1)
        liste_points_R1 = liste_points_R1_bis
    print("\n\nStarting 'transfert des points GNSS vers le repère Tracker...'")
    print("   Nombre de MontéCarlo : ", Nombre_It)
    liste_points_R2 = TransRot(liste_points_R1, Solution) # transfert sans incertitude
    
    # liste_points_R1_inv = TransRotInv(liste_points_R2, Solution) # transfert retour
    liste_points_R2_inc = TransRotInc(liste_points_R1, Solution, Nombre_It) # transfert avec incertitudes
    # enregistrement du fichier
    nom_de_fichier = Fichier_R1.replace(".txt", "_") + "vers_Tracker_" + datetime.now().isoformat(timespec='seconds').replace(":", "-") + ".csv"
    liste_points_R2_inc = liste_points_R2_inc.rename(columns=unite)
    liste_points_R2_inc.to_csv(nom_de_fichier, sep=";", decimal=",", index=True)
    # affichage des graphiques
    plotINC_X = liste_points_R2_inc.plot.scatter(x="X [m]", y="Y [m]", c='sigmaX [m]', colormap='viridis', title="Incertitude sur X", sharex=False)
    plotINC_Y = liste_points_R2_inc.plot.scatter(x="X [m]", y="Y [m]", c='sigmaY [m]', colormap='viridis', title="Incertitude sur Y", sharex=False)
    plotINC_Z = liste_points_R2_inc.plot.scatter(x="X [m]", y="Y [m]", c='sigmaZ [m]', colormap='viridis', title="Incertitude sur Z", sharex=False)
    DIFF = liste_points_R2_inc[["sigmaX [m]", "sigmaY [m]", "sigmaZ [m]"]] - liste_points_R1[["sigmaX", "sigmaY", "sigmaZ"]].rename(columns=unite)
    plotSIGMA = DIFF.plot.bar(title="Ajout d'incertitude par le transfert de coordonnées de GNSS vers Tracker")
    
    end = time.time()
    elapsed = int((end - start))
    print("\n  RESULTATS \n[X Y Z sigmaX sigmaY sigmaZ] exprimés en mètre et dans le repère 2")
    print(liste_points_R2_inc[["X [m]", "Y [m]", "Z [m]", "sigmaX [m]", "sigmaY [m]", "sigmaZ [m]"]])
    print(f"\nTemps d'exécution : {elapsed} s\n")

#%% # changement de coordonées de R2 vers R1 avec calcul d'incertitude par méthode de MontéCarlo

if Tracker_to_GNSS == True :
    start = time.time()
    print("\n\nStarting 'transfert des points Tracker vers le repère GNSS...'")
    print("   Nombre de MontéCarlo : ", Nombre_It)
    # pour chaque fichier trajectoire fille
    for Fichier_R2 in Liste_Fichiers_R2 :
        print("Traitement fichier ", Fichier_R2)
        # ouvrir les fichiers contenant les coordonnées des points dans R2 (souvent Tracker)
        if Spatial_Analyser == True :
            liste_points_R2 = pd.read_csv(Fichier_R2, sep=",", decimal=".", encoding="cp1252", skiprows=[
                                      0, 1], header=None, skipinitialspace=1, usecols=[1, 2, 3, 4, 5, 6])  # fichier direct Spatial Analyser en mm "repère tracker"
            liste_points_R2 = liste_points_R2.rename(columns={1: "X [m]", 2: "Y [m]", 3: "Z [m]", 4: "U95x", 5: "U95y", 6: "U95z"})
            liste_points_R2["DateTime [s]"] = pd.to_datetime("20.12.2023 16:23:00.000", dayfirst = True) # fausse heure ! A remplacer
        elif Tracker_Pilot == True :
            liste_points_R2 = pd.read_csv(Fichier_R2, sep=";", decimal=",", encoding="cp1252", 
                                          skiprows=[0], header=None, skipinitialspace=1, usecols=[1, 2, 3, 5, 6, 7, 12])  # fichier direct Tracker Pilot en mm "repère tracker"
            liste_points_R2 = liste_points_R2.rename(columns={1: "X [m]", 2: "Y [m]", 3: "Z [m]", 5: "U95x", 6: "U95y", 7: "U95z", 12: "DateTime [s]"})
            liste_points_R2["DateTime [s]"] = pd.to_datetime(liste_points_R2["DateTime [s]"], dayfirst = True)
        # filtrage et correction des valeurs
        liste_points_R2[["X [m]", "Y [m]", "Z [m]", "U95x", "U95y", "U95z"]] = liste_points_R2[["X [m]", "Y [m]", "Z [m]", "U95x", "U95y", "U95z"]].div(1000.)
        
        liste_points_R2["sigmaX"] = liste_points_R2["U95x"] / 2.
        liste_points_R2["sigmaY"] = liste_points_R2["U95y"] / 2.
        liste_points_R2["sigmaZ"] = liste_points_R2["U95z"] / 2.
        liste_points_R2["sigma"] = liste_points_R2[[
            "sigmaX", "sigmaY", "sigmaZ"]].max(axis=1)
        
        # calcul principal
        if Incertitude == False :
            liste_points_R1_inc = TransRotInv(liste_points_R2, Solution)
            liste_points_R1_inc[["sigmaX", "sigmaY", "sigmaZ"]] = 0.
        elif Incertitude == True :
            liste_points_R1_inc = TransRotInvInc(liste_points_R2, Solution, Nombre_It)
        # liste_points_R1_inc_inc = TransRotInvInc(liste_points_R2_inc, Solution, Nombre_It)
        
        # Ajout du DateTime [s]
        liste_points_R1_inc["DateTime [s]"] = liste_points_R2["DateTime [s]"]
        
        # enregistrement fichier de points
        nom_de_fichier = Fichier_R2.replace(".csv", "_").replace(".txt", "_") + "vers_GNSS_" + datetime.now(
                                    ).isoformat(timespec='seconds').replace(":", "-") + ".csv"
        liste_points_R1_inc = liste_points_R1_inc.rename(columns=unite)
        liste_points_R1_inc.to_csv(nom_de_fichier, sep=";", decimal=",", index=True)
        # affichage des graphiques
        if Affichage == True :
            plotINC_Z = liste_points_R1_inc.plot.scatter(x="X [m]", y="Y [m]", c='sigmaZ [m]', colormap='viridis', title="Incertitude sur Z", sharex=False)
    
    # end of for Liste_Fichiers_R2
    end = time.time()
    elapsed = int((end - start))
    print(
        "\n  RESULTATS du dernier fichier \n[X Y Z sigmaX sigmaY sigmaZ] exprimés en mètre et dans le repère 2")
    print(liste_points_R1_inc[["X [m]", "Y [m]", "Z [m]", "sigmaX [m]", "sigmaY [m]", "sigmaZ [m]"]])
    print(f"\nTemps d'exécution : {elapsed} s\n")

#%% # analyse "incertitude changeur coord" vs "nbre itération MontéCarlo"
"""
start = time.time()
print("\n\nStarting 'comparatif incertitude vs paramètre MontéCarlo'... \n")

pire_incertitude = pd.Series(data=None, index=None, name="max_sigma")
iteration = 100

for i in range(18) : # pour 18 valeurs de iteration
    print(f"\n\nPour param_mc = {iteration} itérations")
    pire_incertitude.loc[iteration] = TransRotInc(liste_points_R1, Solution, iteration)["sigma"].max()
    iteration = int(iteration * 2**0.5 + 0.5)

end = time.time()
elapsed = int((end - start))
print("\n  RESULTATS \n\nIncertitudes sur chaque paramètre exprimées en mètre et radian en fonction du nombre d'itération dans la méthode MonteCarlo")
print("Incertitudes sur : \n", pire_incertitude)
print(f"\nTemps d'exécution : {elapsed} s\n")
"""