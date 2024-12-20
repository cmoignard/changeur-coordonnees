# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:50:23 2023

@author: cmoignard

maturation algorithme pour fitter au mieux un ensemble de points. Par exemple, une dizaine de points mesurés avec canne d'arpentage
et avec tracker. L'objectif est de trouver la fonction de changement de repère (translation et rotation) 
qui correspond le mieux. Et d'estimer l'incertitude liée à cette fonction de changement de repère, c'est à dire :
    pour une nouvelle coordonnée dans un repère, en calculant les coordonnées dans l'autre repère, quelle serait l'incertitude
    sur ces nouvelles coordonnées

Vigilance : formule de chgmt de base non canonique : translation dans le repère GNSS AVANT la rotation
                                                    rotation selon Euler modifié : Z, X, Y car Z est la rotation principale, GNSS et Tracker ont une bonne verticalité.


Notes de version : Calcul sur de vrai fichiers mesurés grâce aux plots de mesure (export de trimble a changé)
                    Les fichiers GNSS intègrent maintenant l'incertitude
                    Problème : 3% des cas MontéCarlo donnent une mauvaise solution par le solveur (erreur existe même si on change les frontières) -> résolu en réécrivant les formules mathématiques
                    Ajout filtrage des solutions montécarlo qui sont mauvaises. Critère à valider, estimé à Erreur > 1m -> résolu avec les bonnes formules math, et la fonction d'erreur reste au ²
                    Note et retour à l'utilisateur du nombre de solutions éliminées
                    Ajout meilleur initialisation pour les optimizers. + bornage serré des valeurs pour éviter les autres minimum locaux -> OK, ça marche
                    Inversion du calcul : on fait passer les points GPS dans le repère Tracker (au lieu de l'inverse) -> fait
                    Inversion du calcul de changeur : T.R au lieu de R.T -> résolution du problème de mauvaise précision du solveur
                    Affinage de la fonction d'ajout de bruit sur les coordonnées des points (gaussienne ajustée sur chaque coordonnée de chaque point)
                    Amélioration de l'ouverture des fichiers et du filtrage
                    
                    points d'amélioration : utiliser l'incertitude sur X, Y et Z pour les données tracker -> fait

"""

#%% # Imports et paramètres de l'algorithme

# %load_ext cudf.pandas  # pandas operations now use the GPU!

import warnings
warnings.simplefilter("ignore", Warning)



import time
import math
import numpy as np
import pandas as pd
import scipy.optimize as sopt
import random
from datetime import datetime

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 8)

Filtrage = False # Filtrage des optimisations aberrantes (True) ou non filtrage (False)
methode = "SLSQP" # "Powell" semble OK, "L-BFGS-B" part en live avec certains scénarii, "SLSQP" semble OK sur 10 scénarii (pas ok pour des fichiers réels), 4* + rapide que Powell
nombre_it = 10000 # nombre d'itération pour la méthode Montécarlo (calcul d'incertitude pour variation aléatoire gaussienne)
Erreur_GPS = 0.01 # erreur de mesure GPS par défaut [mètre] = 1 sigma
nombre_rep = 5 # nbre de répétition de mesure effectuée avec la canne d'arpentage
# sup = [0, 3, 4, 5, 14, 15, 16, 17, 18] # best
# sup = [0, 3, 4, 14, 17] # best
sup =[]

# format de fichier "tracker" :
Spatial_Analyser = False
Tracker_Pilot = True

# format de fichier "GNSS"
Trimble_GPS_Export = False
Trimble_GPS_Export_with_sigma = True

Fichier_Tracker = "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/Références/REF GNSS EXT.csv"
Fichier_GPS = "C:/Users/cmoignard/Documents/Travail cmoignard/Projets/Protocoles/ARPA PC2/Mesures/20241126_arpa_pc2_hall_fourrage_tscf/GNSS/mesures loc avec TSCF_repère.txt"

#%% # définir la fonction de changement de repère // NDARRAY numpy

def RotTrans (xyz, parametre) :
    """
    transfère les coordonnées d'une liste de points vers le repère 2
    
    Parameters
    ----------
    xyz : np.array. Array de n vecteurs avec 3 coordonnées [[xi, yi, zi],...]. Coordonnées float en mètre
    entree : [float, float, float, float, float, float]
        entree = [deltaX, deltaY, deltaZ, alpha, beta, gamma] en mètre et radian
        entree = valeurs du changeur de coord de Repère 1 vers Repère 2
    deltaX : float
        valeur de la translation selon l'axe X en mètre dans le repère 2 (Tracker).
    deltaY : float
        valeur de la translation selon l'axe Y en mètre dans le repère 2 (Tracker).
    deltaZ : float
        valeur de la translation selong l'axe Z en mètre dans le repère 2 (Tracker).
    alpha : float
        angle de rotation autour de l'axe Z en radian (Euler modifié) = YAW
    beta : float
        angle de rotation autour de l'axe X en radian (Euler modifié) = PITCH
    gamma : float
        angle de rotation autour de l'axe Y en radian (Euler modifié) = ROLL

    Returns
    -------
    np.array. [[float, float, float],...]. coordonnées des point xyz dans le nouveau repère 2

    """
    
    deltaX = parametre[0]
    deltaY = parametre[1]
    deltaZ = parametre[2]
    alpha  = parametre[3] # euler autour de Z
    beta   = parametre[4] # euler autour de X prime
    gamma  = parametre[5] # euler autour de Y prime prime
    
    x_0 = xyz[:,0]
    y_0 = xyz[:,1]
    z_0 = xyz[:,2]
    
    # rotation autour de Z (alpha)
    x_1 = x_0 * math.cos(alpha) - y_0 * math.sin(alpha)
    y_1 = x_0 * math.sin(alpha) + y_0 * math.cos(alpha)
    z_1 = z_0
    
    # rotation autour de X (beta)
    x_2 = x_1
    y_2 = y_1 * math.cos(beta) - z_1 * math.sin(beta)
    z_2 = y_1 * math.sin(beta) + z_1 * math.cos(beta)
    
    # rotation autour de Y (gamma)
    x_3 = z_2 * math.sin(gamma) + x_2 * math.cos(gamma)
    y_3 = y_2
    z_3 = z_2 * math.cos(gamma) - x_2 * math.sin(gamma)
    
    # translation
    x_4 = x_3 + deltaX
    y_4 = y_3 + deltaY
    z_4 = z_3 + deltaZ
    
    return np.transpose(np.array([x_4, y_4, z_4]))

#%% # définir la fonction de changement de repère T.R // NDARRAY numpy
# ATTENTION : formulation non canonique : la translation est effectuée avant la rotation. Donc, les paramètres de translation
# s'écrivent dans le repère 1 avant transfert dans le repère 2

def TransRot (xyz, parametre) :
    """
    transfère les coordonnées d'une liste de points vers le repère 2
    
    Parameters
    ----------
    xyz : np.array. Array de n vecteurs avec 3 coordonnées [[xi, yi, zi],...]. Coordonnées float en mètre
    entree : [float, float, float, float, float, float]
        entree = [deltaX, deltaY, deltaZ, alpha, beta, gamma] en mètre et radian
        entree = valeurs du changeur de coord de Repère 1 vers Repère 2
    deltaX : float
        valeur de la translation selon l'axe X en mètre dans le repère 1 (GNSS).
    deltaY : float
        valeur de la translation selon l'axe Y en mètre dans le repère 1 (GNSS).
    deltaZ : float
        valeur de la translation selong l'axe Z en mètre dans le repère 1 (GNSS).
    alpha : float
        angle de rotation autour de l'axe Z en radian (Euler modifié) = YAW
    beta : float
        angle de rotation autour de l'axe X en radian (Euler modifié) = PITCH
    gamma : float
        angle de rotation autour de l'axe Y en radian (Euler modifié) = ROLL

    Returns
    -------
    np.array. [[float, float, float],...]. coordonnées des point xyz dans le nouveau repère 2

    """
    
    deltaX = parametre[0]
    deltaY = parametre[1]
    deltaZ = parametre[2]
    alpha  = parametre[3] # euler autour de Z
    beta   = parametre[4] # euler autour de X prime
    gamma  = parametre[5] # euler autour de Y prime prime
    
    # Translation dans le repère 1
    x_0 = xyz[:,0] + deltaX
    y_0 = xyz[:,1] + deltaY
    z_0 = xyz[:,2] + deltaZ
    
    # rotation autour de Z (alpha)
    x_1 = x_0 * math.cos(alpha) - y_0 * math.sin(alpha)
    y_1 = x_0 * math.sin(alpha) + y_0 * math.cos(alpha)
    z_1 = z_0
    
    # rotation autour de X (beta)
    x_2 = x_1
    y_2 = y_1 * math.cos(beta) - z_1 * math.sin(beta)
    z_2 = y_1 * math.sin(beta) + z_1 * math.cos(beta)
    
    # rotation autour de Y (gamma)
    x_3 = z_2 * math.sin(gamma) + x_2 * math.cos(gamma)
    y_3 = y_2
    z_3 = z_2 * math.cos(gamma) - x_2 * math.sin(gamma)
    
    
    return np.transpose(np.array([x_3, y_3, z_3]))

#%% # définir la fonction de changement de repère R2 // DataFrame

def RotTransBis (liste_points_rep1, parametres) :
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
    
    x_0 = liste_points_rep1[["X [m]"]]
    y_0 = liste_points_rep1[["Y [m]"]]
    z_0 = liste_points_rep1[["Z [m]"]]
    # sigma_0 = liste_points_rep1[["sigma"]]
    
    # rotation autour de Z (alpha)
    x_1 = x_0 * math.cos(alpha) - y_0.to_numpy() * math.sin(alpha)
    y_1 = x_0.to_numpy() * math.sin(alpha) + y_0 * math.cos(alpha)
    z_1 = z_0
    # sigma_1 = sigma_0 + 1. # exemple calcul => intégrer la vrai formule
    
    # rotation autour de X (beta)
    x_2 = x_1
    y_2 = y_1 * math.cos(beta) - z_1.to_numpy() * math.sin(beta)
    z_2 = y_1.to_numpy() * math.sin(beta) + z_1 * math.cos(beta)
    # sigma_2 = sigma_1 + 1. # exemple calcul => intégrer la vrai formule
    
    # rotation autour de Y (gamma)
    x_3 = z_2.to_numpy() * math.sin(gamma) + x_2 * math.cos(gamma)
    y_3 = y_2
    z_3 = z_2 * math.cos(gamma) - x_2.to_numpy() * math.sin(gamma)
    # sigma_3 = sigma_2 + 1. # exemple calcul => intégrer la vrai formule
    
    # translation
    x_4 = x_3 + deltaX
    y_4 = y_3 + deltaY
    z_4 = z_3 + deltaZ
    # sigma_4 = sigma_3 + 1. # exemple calcul => intégrer la vrai formule
    
    return pd.concat([x_4, y_4, z_4], axis=1)

#%% # définir la fonction de changement de repère R2 // DataFrame
# ATTENTION : formulation non canonique : la translation est effectuée avant la rotation. Donc, les paramètres de translation
# s'écrivent dans le repère 1 avant transfert dans le repère 2

def TransRotBis (liste_points_rep1, parametres) :
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
    # Translation dans le repère 1 (GNSS)
    x_0 = liste_points_rep1[["X [m]"]] + deltaX
    y_0 = liste_points_rep1[["Y [m]"]] + deltaY
    z_0 = liste_points_rep1[["Z [m]"]] + deltaZ
    # sigma_0 = liste_points_rep1[["sigma"]]
    
    # rotation autour de Z (alpha)
    x_1 = x_0 * math.cos(alpha) - y_0.to_numpy() * math.sin(alpha)
    y_1 = x_0.to_numpy() * math.sin(alpha) + y_0 * math.cos(alpha)
    z_1 = z_0
    # sigma_1 = sigma_0 + 1. # exemple calcul => intégrer la vrai formule
    
    # rotation autour de X (beta)
    x_2 = x_1
    y_2 = y_1 * math.cos(beta) - z_1.to_numpy() * math.sin(beta)
    z_2 = y_1.to_numpy() * math.sin(beta) + z_1 * math.cos(beta)
    # sigma_2 = sigma_1 + 1. # exemple calcul => intégrer la vrai formule
    
    # rotation autour de Y (gamma)
    x_3 = z_2.to_numpy() * math.sin(gamma) + x_2 * math.cos(gamma)
    y_3 = y_2
    z_3 = z_2 * math.cos(gamma) - x_2.to_numpy() * math.sin(gamma)
    # sigma_3 = sigma_2 + 1. # exemple calcul => intégrer la vrai formule
    
    return pd.concat([x_3, y_3, z_3], axis=1)


#%% # définir la fonction d'erreur pour une fonction de changement de repère donnée // NDARRAY cupy

def CalculErreur (parametre, table_1, table_2) :
    """
    Calcule la somme des distances euclidiennes (au carré) entre 2 nuages de points exprimés dans le repère 2

    Parameters
    ----------
    entree : [float, float, float, float, float, float]
        [deltaX, deltaY, deltaZ, rotX, rotY, rotZ] exprimé en mètre et en radian
    table_1 : [[float, float, float], ...]
        liste des coordonnées des points dans le repère 1 exprimées en mètre
    table_2 : [[float, float, float], ...]
        liste des coordonnées des points dans le repère 2 exprimées en mètre

    Returns
    -------
    float. Somme des carrés des distances euclidiennes entre les coordonnées de 1 exprimées dans le repère 2
    et les coordonnées de 2 exprimées dans le repère 2

    """
    
    if len(table_1) < len(table_2) :
        print ("fonction calcul erreur : \n", "ATTENTION, il y a plus de points dans le repère 2 que dans le repère 1")
    elif len(table_1) > len(table_2) :
        print ("fonction calcul erreur : \n", "ATTENTION, il y a plus de points dans le repère 1 que dans le repère 2")
    table_1_bis = TransRot(table_1, parametre)
    erreur = ((table_2[:,0] - table_1_bis[:,0]) ** 2 + (table_2[:,1] - table_1_bis[:,1]) ** 2 + (table_2[:,2] - table_1_bis[:,2]) ** 2)
    
    return float (erreur.sum())

#%% # optimisation = minimisation de la fonction erreur ErreurRotTrans
def calcul_1_changeur_coord (liste_points_1, liste_points_2, X0) :
    """
    Solveur d'un changeur de coordonnées pour une paire de nuage de points
    
    Repère 1 vers repère 2
    
    Parameters
    ----------
    liste_points_1 : DataFrame
        Liste des points + incertitude dans le repère 1.
    liste_points_2 : Dataframe
        Liste des points + incertitude dans le repère 2.
    X0 : [float, float, float, float, float, float]
        list of 6 float = initialisation of 6 parameters

    Returns best fit de changeur de coordonnées 
    -------
    sol : OptimizeResult
        cf sopt.minimize()

    """
    # initialisation des frontières
    x0bound = (X0[0]-150., X0[0]+150.) # limites de DeltaX
    x1bound = (X0[1]-50., X0[1]+50.) # limites de DeltaY
    x2bound = (X0[2]-1., X0[2]+1.) # limites de DeltaZ
    # x0bound = (None, None) # limites de DeltaX
    # x1bound = (None, None) # limites de DeltaY
    # x2bound = (None, None) # limites de DeltaZ
    x3bound = (X0[3]-math.pi, X0[3]+math.pi) # limites de alpha
    # x3bound = (0, 2. * math.pi) # limites de alpha
    x4bound = (X0[4]-0.1, X0[4]+0.1) # limites de beta
    x5bound = (X0[5]-0.1, X0[5]+0.1) # limites de gamma
    # x4bound = (-1.5, 1.5)
    # x5bound = (-1.5, 1.5)

    X_bounds = [x0bound, x1bound, x2bound, x3bound, x4bound, x5bound]
    """
    X0_bis = X0
    X0_bis[0] = random.uniform(X0[0]-1., X0[0]+1.)
    X0_bis[1] = random.uniform(X0[1]-1., X0[1]+1.)
    X0_bis[2] = random.uniform(X0[2]-0.2, X0[2]+0.2)
    X0_bis[3] = random.uniform(X0[3]-0.01, X0[3]+0.01)
    X0_bis[4] = random.uniform(X0[4]-0.01, X0[4]+0.01)
    X0_bis[5] = random.uniform(X0[5]-0.01, X0[5]+0.01)
    """
    # initialisation des tableaux de coordonnées
    table_1 = np.array(liste_points_1[["X [m]", "Y [m]", "Z [m]"]].values.tolist())
    table_2 = np.array(liste_points_2[["X [m]", "Y [m]", "Z [m]"]].values.tolist())

    sol = sopt.minimize(CalculErreur, x0=X0, args=(table_1, table_2), method=methode, tol=1e-50, 
                                       bounds=X_bounds, options={'disp': False, 'xtol': 1e-50, 'ftol': 1e-50})
    return sol

#%% # calcul du changeur de coordonnées moyen sur la distribution MontéCarlo avec incertitude

def calcul_changeur_coord_montecarlo (liste_points_1, liste_points_2, param_mc, X0) :
    """
    fonction qui calcule les paramètres du changeur de coordonnées et des incertitudes associées
    en utilisant la méthode de Montécarlo. Génération de param_mc paires de nuages de points avec distribution
    gaussienne dans l'incertitude de leurs coordonnées. Puis solveur sur chaque paire de nuage de point. Sauvegarde
    de chaque solution du solveur. les paramètres du changeur de coord sont les moyennes, les incertitudes sont les écarts
    types des paramètres.
    
    Repère 1 vers Repère 2

    Parameters
    ----------
    liste_points_1 : DataFrame
        Liste des points 1 (GNSS) avec incertitude.
    liste_points_2 : DataFrame
        Liste des points 2 (Tracker) avec incertitude.
    param_mc : int 
        nombre de répétitions dans la méthode Montécarlo.

    Returns
    -------
    table_sol : DataFrame
        index=["Valeurs", "Sigma"], columns=["DeltaX [m]", "DeltaY [m]", "DeltaZ [m]", "Alpha(Z) [rad]","Beta(X) [rad]", "Gamma(Y) [rad]", "Erreur [m]"].

    """
    # initialisation des variables de stockage
    solution_0 = calcul_1_changeur_coord (liste_points_1, liste_points_2, X0)
    table_sol = np.atleast_2d(np.zeros_like(np.append(solution_0.x, solution_0.fun)))
    
    # sigma_1 = liste_points_1["sigma"].max()
    # sigma_2 = liste_points_2["sigma"].max()
        
    # boucle MontéCarlo
    for i in range(param_mc) : # génération d'un scénario et best-fit sur ce scénario
        # ici génération d'un scénario en ajoutant une part aléatoire distribution normale sur la valeur des coord des points
        # à modifier pour que la part aléatoire reflète l'incertitude du point et non le max de tous les points.
        liste_points_1_ = liste_points_1[0:0]
        liste_points_2_ = liste_points_2[0:0]
        for index in liste_points_1.index :
            liste_points_1_.at[index, "X [m]"] = random.gauss(mu=liste_points_1.at[index,"X [m]"], sigma=liste_points_1.at[index,"sigmaX"])
            liste_points_1_.at[index, "Y [m]"] = random.gauss(mu=liste_points_1.at[index,"Y [m]"], sigma=liste_points_1.at[index,"sigmaY"])
            liste_points_1_.at[index, "Z [m]"] = random.gauss(mu=liste_points_1.at[index,"Z [m]"], sigma=liste_points_1.at[index,"sigmaZ"])
            liste_points_2_.at[index, "X [m]"] = random.gauss(mu=liste_points_2.at[index,"X [m]"], sigma=liste_points_2.at[index,"sigmaX"])
            liste_points_2_.at[index, "Y [m]"] = random.gauss(mu=liste_points_2.at[index,"Y [m]"], sigma=liste_points_2.at[index,"sigmaY"])
            liste_points_2_.at[index, "Z [m]"] = random.gauss(mu=liste_points_2.at[index,"Z [m]"], sigma=liste_points_2.at[index,"sigmaZ"])
        """
        liste_points_1_ = liste_points_1[["X", "Y", "Z"]] + np.random.normal(loc=0., scale=sigma_1, size=(len(liste_points_1[["X", "Y", "Z"]].values.tolist()), 3))
        liste_points_2_ = liste_points_2[["X", "Y", "Z"]] + np.random.normal(loc=0., scale=sigma_2, size=(len(liste_points_2[["X", "Y", "Z"]].values.tolist()), 3))
        """
        sol_tmp = calcul_1_changeur_coord (liste_points_1_, liste_points_2_, X0) # ici, on doit entrer des DF et non des numpy
        table_sol = np.append(table_sol, np.atleast_2d(np.append(sol_tmp.x, sol_tmp.fun)), axis=0)
    
    # convertion de la liste des changeurs de coord en DataFrame
    table_sol = pd.DataFrame(table_sol, columns=["DeltaX [m]", "DeltaY [m]", "DeltaZ [m]", "Alpha(Z) [rad]","Beta(X) [rad]", "Gamma(Y) [rad]", "Erreur [m²]"])
    table_sol = table_sol.drop([0])
    
    # fichier = "Liste_des_changeurs_ZXY_" + methode + "_" + str(nombre_it) + "_scenarii_" + str(time.time()) + ".csv"
    # table_sol.to_csv(fichier, sep=";", decimal=",", index=True)
    
    # filtrer les solutions, éliminier les "mauvaises"
    if Filtrage == True :
        table_sol2 = table_sol[table_sol["Erreur [m²]"] < 1.0]
        table_sol_erreur = table_sol[table_sol["Erreur [m²]"] >= 1.0]
        compteur_erreur, col = table_sol_erreur.shape
        compteur2, col = table_sol2.shape
        pourcentage = float(compteur2)/float(param_mc)*100.
        print ("\n   Nombre de calculs aberrants éliminés : ", compteur_erreur)
        print ("   Calcul réalisé sur : ", pourcentage, "% des répétitions")
    else :
        table_sol2 = table_sol
        pourcentage = 100.
    
    # convertion de tous les changeurs de coord en SOLUTION : moyenne + écart-type
    data = {"Valeurs" : table_sol2.mean(), "Sigma" : table_sol2.std()}
    solution = pd.DataFrame(data).transpose()
    solution["Pourcentage [%]"] = pourcentage
    
    while solution.at["Valeurs", "Alpha(Z) [rad]"] < 0. : # canonisation de Alpha(Z)
        solution.at["Valeurs", "Alpha(Z) [rad]"] = solution.at["Valeurs", "Alpha(Z) [rad]"] + 2. * math.pi
    while solution.at["Valeurs", "Alpha(Z) [rad]"] >= 2. * math.pi:
        solution.at["Valeurs", "Alpha(Z) [rad]"] = solution.at["Valeurs", "Alpha(Z) [rad]"] - 2. * math.pi

    return solution

#%% # ouvrir les fichiers contenant les coordonnées des points.

if Spatial_Analyser == True :
    liste_points_Tracker = pd.read_csv(Fichier_Tracker, sep=",", decimal=".", encoding="cp1252", skiprows=[0, 1], header=None, skipinitialspace=1, usecols=[1, 2, 3, 4, 5, 6]) # fichier direct Spatial Analyser ? en mm "repère tracker"
elif Tracker_Pilot == True :
    liste_points_Tracker = pd.read_csv(Fichier_Tracker, sep=";", decimal=",", encoding="cp1252", skiprows=[0], header=None, skipinitialspace=1, usecols=[1, 2, 3, 5, 6, 7]) # fichier direct Tracker Pilot en mm "repère tracker"
else :
    print("ne peut pas lire le fichier tracker, veuillez vérifier le format choisi")

if Trimble_GPS_Export_with_sigma == True :
    # liste_points_GPS = pd.read_csv(Fichier_GPS, sep=" ", decimal=".", skiprows=[0, 1, 2, 3], header=None, skipinitialspace=1, usecols=[1, 2, 3, 4, 5]) # fichier direct export du Trimble en mètre lambert CC46
    liste_points_GPS = pd.read_csv(Fichier_GPS, sep=" ", decimal=".", skiprows=[0, 1, 2, 3], header=None, skipinitialspace=1, usecols=[3, 4, 5, 6, 7]) # spécial à cause des espaces dans les noms de points /!\ nommage points trimble
elif Trimble_GPS_Export == True :
    liste_points_GPS = pd.read_csv(Fichier_GPS, sep=";", decimal=".", skiprows=[0, 1], header=None, skipinitialspace=1, usecols=[1, 2, 3]) # export du Trimble sans précision en mètre lambert CC46

#%% # filtrer et réorganiser les données
if Tracker_Pilot == True :
    liste_points_Tracker = liste_points_Tracker.rename(columns={1:"X [m]", 2:"Y [m]", 3:"Z [m]", 5:"U95x", 6:"U95y", 7:"U95z"})
    liste_points_Tracker = liste_points_Tracker.div(1000.)
elif Spatial_Analyser == True : # filtrage à vérifier
    liste_points_Tracker = liste_points_Tracker.rename(columns={1:"X [m]", 2:"Y [m]", 3:"Z [m]", 4:"U95x", 5:"U95y", 6:"U95z"})
    liste_points_Tracker = liste_points_Tracker.div(1000.)

liste_points_Tracker["sigmaX"] = liste_points_Tracker["U95x"] / 2.
liste_points_Tracker["sigmaY"] = liste_points_Tracker["U95y"] / 2.
liste_points_Tracker["sigmaZ"] = liste_points_Tracker["U95z"] / 2.
liste_points_Tracker["sigma"] = liste_points_Tracker[["sigmaX", "sigmaY", "sigmaZ"]].max(axis=1)

"""
liste_points_GPS = liste_points_GPS.rename(columns={1:"X", 2:"Y", 3:"Z"})

liste_points_GPS["sigma"] = liste_points_GPS[[4, 5]].max(axis=1)/2. # formule simplifiée, peut-être on pourra améliorer

# reshaping le tableau de points GPS en supposant qu'on a fait nombre_rep de mesures par point tracker
ligne, colonne = liste_points_GPS.shape
liste_points_GPS_bis = pd.DataFrame()
for i in range(ligne//nombre_rep) : # regrouper les mesures GPS par points tracker
    selection = liste_points_GPS[["X", "Y", "Z", "sigma"]].iloc[i*nombre_rep:nombre_rep*(i+1)]
    sel = selection[["X", "Y", "Z"]].mean()
    if i==0 : # initialisation du dataframe = première ligne du dataframe
        liste_points_GPS_bis = sel.to_frame().transpose()
        liste_points_GPS_bis["sigma"] = math.sqrt(selection["X"].std()**2 + 
                                                  selection["Y"].std()**2 +
                                                  selection["Z"].std()**2 +
                                                  (selection["sigma"].max())**2) # calcul de somme d'écart-type
    else :
        liste_points_GPS_bis = pd.concat([liste_points_GPS_bis, sel.to_frame().transpose()], ignore_index=True,)
        liste_points_GPS_bis.at[i, "sigma"] = math.sqrt(selection["X"].std()**2 + 
                                                        selection["Y"].std()**2 +
                                                        selection["Z"].std()**2 +
                                                        (selection["sigma"].max())**2) # calcul de somme d'écart-type
liste_points_GPS = liste_points_GPS_bis
"""
if Trimble_GPS_Export == True :
    liste_points_GPS[4] = Erreur_GPS
    liste_points_GPS[5] = Erreur_GPS

# liste_points_GPS = liste_points_GPS.rename(columns={1:"X [m]", 2:"Y [m]", 3:"Z [m]", 4:"HzPrec", 5:"VePrec"}) # original
liste_points_GPS = liste_points_GPS.rename(columns={3:"X [m]", 4:"Y [m]", 5:"Z [m]", 6:"HzPrec", 7:"VePrec"}) # special à cause noms de points trimble avec espaces

liste_points_GPS["sigma"] = liste_points_GPS[["HzPrec", "VePrec"]].max(axis=1)# /2. # Correction, ici le HzPrec et VePrec sont 1 sigma
liste_points_GPS["sigmaX"] = liste_points_GPS["HzPrec"]# /2.
liste_points_GPS["sigmaY"] = liste_points_GPS["HzPrec"]# /2.
liste_points_GPS["sigmaZ"] = liste_points_GPS["VePrec"]# /2.

# reshaping le tableau de points GPS en supposant qu'on a fait nombre_rep de mesures par point tracker
ligne = len(liste_points_GPS.index)
nb_pt = ligne//nombre_rep
liste_points_GPS_bis = liste_points_GPS[0:0]
for index in range(nb_pt) : # regrouper les mesures GPS par points tracker
    selection = liste_points_GPS[["X [m]", "Y [m]", "Z [m]", "sigmaX", "sigmaY", "sigmaZ"]].iloc[index*nombre_rep:nombre_rep*(index+1)]
    # sel = selection[["X", "Y", "Z"]].mean()
    new_line = selection[["X [m]", "Y [m]", "Z [m]"]].mean().rename(index)
    new_line["sigmaX"] = (selection["X [m]"].std()**2 + (selection["sigmaX"]**2).sum())**0.5 # calcul de somme d'écart-type sur X
    new_line["sigmaY"] = (selection["Y [m]"].std()**2 + (selection["sigmaY"]**2).sum())**0.5 # calcul de somme d'écart-type sur Y
    new_line["sigmaZ"] = (selection["Z [m]"].std()**2 + (selection["sigmaZ"]**2).sum())**0.5 # calcul de somme d'écart-type sur Z
    liste_points_GPS_bis = pd.concat([liste_points_GPS_bis, new_line.to_frame().T], ignore_index=False)
    
liste_points_GPS_bis["sigma"] = liste_points_GPS_bis[["sigmaX", "sigmaY", "sigmaZ"]].max(axis=1)
liste_points_GPS = liste_points_GPS_bis

# suppression des points en fonction de la liste en début de fichier
liste_points_Tracker = liste_points_Tracker.drop(index=sup)
liste_points_GPS = liste_points_GPS.drop(index=sup)

#%% # Calcul des valeurs initiales du changeur de coordonnées

start = time.time()
print("\n\nStarting 'Calcul des paramètres initiaux du changeur de coordonnées'...\n")
# print("   Optimizer sur 4 points choisis au hasard en 2D")

lignes, colonnes = liste_points_Tracker.shape

# initialisation des valeurs de rotation (conservation de l'angle d'un vecteur AB) en 2D
beta0 = 0
gamma0 = 0

# definition de la fonction d'erreur en 2D pour R.T
def Erreur2D_RT (parametre, table_1, table_2) :
    alpha = parametre[0]
    x = parametre[1]
    y = parametre[2]
    table_1_bis = table_1.copy()
    table_1_bis[:,0] = table_1[:,0] * math.cos(alpha) - table_1[:,1] * math.sin(alpha) + x
    table_1_bis[:,1] = table_1[:,0] * math.sin(alpha) + table_1[:,1] * math.cos(alpha) + y
    
    erreur = (table_2[:,0] - table_1_bis[:,0]) ** 2 + (table_2[:,1] - table_1_bis[:,1]) ** 2
    
    return float(erreur.sum())

# definition de la fonction d'erreur en 2D pour T.R
def Erreur2D_TR (parametre, table_1, table_2) :
    alpha = parametre[0]
    x = parametre[1]
    y = parametre[2]
    table_1_bis = table_1.copy()
    # translation
    table_1_bis[:,0] = table_1_bis[:,0] + x
    table_1_bis[:,1] = table_1_bis[:,1] + y
    # rotation de alpha
    table_1_bis[:,0] = table_1_bis[:,0] * math.cos(alpha) - table_1_bis[:,1] * math.sin(alpha)
    table_1_bis[:,1] = table_1_bis[:,0] * math.sin(alpha) + table_1_bis[:,1] * math.cos(alpha)
    # calcul de distance carré entre chaque point
    erreur = (table_2[:,0] - table_1_bis[:,0]) ** 2 + (table_2[:,1] - table_1_bis[:,1]) ** 2
    
    return float(erreur.sum())

# intialisation des valeurs de rotation verticale et de translation horizontale à partir d'une moyenne d'optimisation 2D
# sur 50 tirages aléatoires
compteur = 0
table_sol = np.atleast_2d(np.zeros_like([0, 0, 0, 0]))
while compteur < 50 : # initialisation sur 50 tirages aléatoires
    # tirage aléatoire d'un vecteur pour la pré-initialisation
    vecteur = random.sample(range(0, lignes), k=2)
    point_A_Tracker = liste_points_Tracker.iloc[vecteur[0]]
    point_B_Tracker = liste_points_Tracker.iloc[vecteur[1]]
    point_A_GPS = liste_points_GPS.iloc[vecteur[0]]
    point_B_GPS = liste_points_GPS.iloc[vecteur[1]]
    # calcul de alpha0 pré-init de l'angle entre les 2 repères
    alpha0 = (math.atan2(point_B_GPS["Y [m]"]-point_A_GPS["Y [m]"], point_B_GPS["X [m]"]-point_A_GPS["X [m]"]) -
              math.atan2(point_B_Tracker["Y [m]"]-point_A_Tracker["Y [m]"], point_B_Tracker["X [m]"]-point_A_Tracker["X [m]"]))
    
    while alpha0 < 0. : # canonisation alpha0
        alpha0 = alpha0 + 2. * math.pi
    while alpha0 >= 2. * math.pi:
        alpha0 = alpha0 - 2. * math.pi
    
    # initialisation des valeurs de translation (convertion de l'origine du repère GNSS dans le repère Tracker)
    # pour T.R (translation est dans le repère GNSS)
    x0 = (point_A_Tracker["X [m]"]*math.cos(-1.*alpha0)-point_A_Tracker["Y [m]"]*math.sin(-1.*alpha0)) - point_A_GPS["X [m]"]
    y0 = (point_A_Tracker["X [m]"]*math.sin(-1.*alpha0)+point_A_Tracker["Y [m]"]*math.cos(-1.*alpha0)) - point_A_GPS["Y [m]"]
    # tirage de 5 points aléatoires
    liste = random.sample(range(0, lignes), k=5)
    liste_points_GPS_bis = liste_points_GPS.iloc[liste]
    liste_points_Tracker_bis = liste_points_Tracker.iloc[liste]
    table_Tracker = np.array(liste_points_Tracker_bis[["X [m]", "Y [m]"]].values.tolist())
    table_GPS = np.array(liste_points_GPS_bis[["X [m]", "Y [m]"]].values.tolist())
    # solution = minimisation de la fonction d'erreur sur les 5 points aléatoires
    sol = sopt.minimize(Erreur2D_TR, [alpha0 , x0, y0], bounds=[(alpha0-0.5, alpha0+0.5), (x0-1000., x0+1000.), (y0-1000., y0+1000.)], 
                        args=(table_GPS, table_Tracker), method="Powell",
                        options={'disp': False, "ftol":0.000000001})
    compteur += 1
    table_sol = np.append(table_sol, np.atleast_2d(np.append(sol.x, sol.fun)), axis=0)
    """
    print("Liste : ", liste)
    print("E alpha : ", sol.x[0] - alpha0)
    print("E delta X : ", sol.x[1] - x0)
    print("E delta Y : ", sol.x[2] - y0)
    print("Erreur : ", sol.fun, "\n")
    """

table_sol = np.delete(table_sol, 0, 0)
alpha0 = table_sol.mean(axis=0)[0] # moyenne des 50 solutions tirées au hasard
x0 = table_sol.mean(axis=0)[1] # moyenne des 50 solutions tirées au hasard
y0 = table_sol.mean(axis=0)[2] # moyenne des 50 solutions tirées au hasard

# initialisation de deltaZ
z0 = np.mean(liste_points_Tracker[["Z [m]"]]-liste_points_GPS[["Z [m]"]]) # la moyenne des différences d'altitude (hypothèse beta et gamma nul)

# calibrage de alpha0 entre 0 et 2 pi
while alpha0 < 0. :
    alpha0 = alpha0 + 2. * math.pi
while alpha0 >= 2. * math.pi:
    alpha0 = alpha0 - 2. * math.pi

# affichage des résultats de l'initialisation
print("Initialisation de alpha(Z) [rad] : ", alpha0, "     std : ", table_sol.std(axis=0)[0])
print("Initialisation de beta(X) [rad]  : ", 0., "                     std : ", 0.)
print("Initialisation de gamma(Y) [rad] : ", 0., "                     std : ", 0.)
print("Initialisation de DeltaX [m]     : ", x0, "     std : ", table_sol.std(axis=0)[1])
print("Initialisation de DeltaY [m]     : ", y0, "     std : ", table_sol.std(axis=0)[2])
print("Initialisation de DeltaZ [m]     : ", z0, "     std : ", np.std(liste_points_Tracker[["Z [m]"]]-liste_points_GPS[["Z [m]"]]).iloc[0])
print("Erreur moyenne initialisation    : ", table_sol.mean(axis=0)[3], "     std : ", table_sol.std(axis=0)[3])
end = time.time()
elapsed = int((end - start))
print(f"\nTemps d'exécution : {elapsed} s\n")

X0 = [x0, y0, z0, alpha0, beta0, gamma0] # paramètres initiaux de la fonction de changement de repère.
# estimation de l'erreur sur la translation H : 50 m
# estimation de l'erreur sur la translation V : <1m
# estimation de l'erreur sur la rotation alpha autour de Z : < 0.1
# estimation de l'erreur sur les rotations beta et gamma : < 0.01

# Transfert des points en 2D pour voir l'erreur
# pd.DataFrame(table_sol, columns=["DeltaX [m]", "DeltaY [m]", "DeltaZ [m]", "Alpha(Z) [rad]","Beta(X) [rad]", "Gamma(Y) [rad]", "Erreur [m²]"])
table_sol = pd.DataFrame(table_sol, columns=["Alpha(Z) [rad]", "DeltaX [m]", "DeltaY [m]", "Erreur [m²]"])
data = {"Valeurs" : table_sol.mean(), "Sigma" : table_sol.std()}
Param_2D = pd.DataFrame(data).transpose()
Param_2D.at["Valeurs", "DeltaZ [m]"] = z0
Param_2D.at["Sigma", "DeltaZ [m]"] = 0.
Param_2D.at["Valeurs", "Beta(X) [rad]"] = beta0
Param_2D.at["Sigma", "Beta(X) [rad]"] = 0.
Param_2D.at["Valeurs", "Gamma(Y) [rad]"] = gamma0
Param_2D.at["Sigma", "Gamma(Y) [rad]"] = 0.

# affichage des points avec transfert en 2D
liste_points_R1_dans_R2_2D = TransRotBis(liste_points_GPS, Param_2D)
liste_points_R1_dans_R2_2D["ErreurX [m]"] = liste_points_Tracker["X [m]"] - liste_points_R1_dans_R2_2D["X [m]"]
liste_points_R1_dans_R2_2D["ErreurY [m]"] = liste_points_Tracker["Y [m]"] - liste_points_R1_dans_R2_2D["Y [m]"]
liste_points_R1_dans_R2_2D["ErreurZ [m]"] = liste_points_Tracker["Z [m]"] - liste_points_R1_dans_R2_2D["Z [m]"]
liste_points_R1_dans_R2_2D["Erreur [m]"] = (liste_points_R1_dans_R2_2D["ErreurX [m]"]**2 + liste_points_R1_dans_R2_2D["ErreurY [m]"]**2 + liste_points_R1_dans_R2_2D["ErreurZ [m]"]**2)**0.5
#print("Points GPS dans repère Tracker :\n", liste_points_R1_dans_R2_2D)
plot2D = liste_points_R1_dans_R2_2D.plot.scatter(x="X [m]", y="Y [m]", color="DarkBlue", label="GNSS dans Tracker / 2D", title="Points de ref - 2D - projection sur X-Y")

liste_points_Tracker.plot.scatter(x="X [m]", y="Y [m]", color="DarkRed", label="Tracker dans Tracker", ax=plot2D)


#%% # Calcul du changeur de coord selon méthode MontéCarlo

start = time.time()
print("\n\nStarting 'calcul des paramètres du changeur de coordonnées avec méthode MontéCarlo'...")
print("\nMéthode : ", methode)
print("Euler : Z,X,Y")
print("Nombre d'iteration (Montécarlo) : ", nombre_it)
lignes, cols = liste_points_Tracker.shape
print("Nombre de points : ", lignes, "     sup : ", sup)
print("Sigma tracker : ", liste_points_Tracker["sigma"].max(), "      Sigma GNSS : ", liste_points_GPS["sigma"].max())

Solution = calcul_changeur_coord_montecarlo (liste_points_GPS, liste_points_Tracker, nombre_it, X0)
nom_de_fichier = "TR_ZXY_GNSSS_Tracker_" + methode + "_" + str(nombre_it) + "mc_" + str(sup) + "sup_" + datetime.now().isoformat(timespec='seconds').replace(":", "-") + ".csv"
Solution.to_csv(nom_de_fichier, sep=";", decimal=",", index=True)

#%% # affichage des résultats

end = time.time()
elapsed = int((end - start))
print("\n  RESULTATS \n\n[deltaX  deltaY  deltaZ  alpha(Z)=Yaw  beta(X)=Pitch  gamma(Y)=Roll] exprimés en mètre et radian")
print("Solution avec les point(s) ", sup, " supprimé(s) : \n", Solution[["DeltaX [m]", "DeltaY [m]", "DeltaZ [m]"]], "\n\n", Solution[["Alpha(Z) [rad]","Beta(X) [rad]", "Gamma(Y) [rad]", "Erreur [m²]"]])
print(f"\nTemps d'exécution : {elapsed} s\n")

#%% # analyse "incertitude changeur coord" vs "nombre de points"
"""
start = time.time()
nombre_it = 500
print("\n\nStarting comparatif incertitude vs nbre de points GNSS \n")
print("Méthode : ", methode)
print("Euler : Z,X,Y")
print("Nombre d'iteration (Montécarlo) : ", nombre_it)
liste_sol = Solution.drop(["Valeurs"])

for i in range(4, lignes+1) : # pour 4 à 19 points
    liste_points_Tracker_bis = liste_points_Tracker.iloc[:i]
    liste_points_GPS_bis = liste_points_GPS.iloc[:i]
    # i_bis = 19-i
    print(f"\n\nPour {i} points")
    liste_sol.loc[i] = calcul_changeur_coord_montecarlo (liste_points_GPS_bis, liste_points_Tracker_bis, nombre_it, X0).loc["Sigma"]

liste_sol = liste_sol.drop(["Sigma"])
end = time.time()
elapsed = int((end - start))
print("\n  RESULTATS \n\nIncertitudes sur chaque paramètre exprimées en mètre et radian en fonction du nombre de points utilisés pour le calcul")
print("Incertitudes sur : \n", liste_sol[["DeltaX [m]", "DeltaY [m]", "DeltaZ [m]", "Alpha(Z) [rad]","Beta(X) [rad]", "Gamma(Y) [rad]"]])
print(f"\nTemps d'exécution : {elapsed} s\n")
"""
#%% # analyse "incertitude changeur coord" vs "nbre itération MontéCarlo"
"""
start = time.time()
print("\n\nStarting comparatif incertitude vs paramètre MontéCarlo \n")
print("Méthode : ", methode)
print("Euler : Z,X,Y")
# print("Nombre d'iteration (Montécarlo) : ", 1000)
print("Nombre de points : ", ligne//nombre_rep)
liste_sol2 = Solution.drop(["Valeurs"])
iteration = 100

for i in range(18) : # pour 18 valeurs de iteration
    print(f"\n\nPour param_mc = {iteration} itérations")
    liste_sol2.loc[iteration] = calcul_changeur_coord_montecarlo (liste_points_GPS, liste_points_Tracker, iteration, X0).loc["Sigma"]
    iteration = int(iteration * 2**0.5 + 0.5)

liste_sol2 = liste_sol2.drop(["Sigma"])
end = time.time()
elapsed = int((end - start))
print("\n  RESULTATS \n\nIncertitudes sur chaque paramètre exprimées en mètre et radian en fonction du nombre d'itération dans la méthode MonteCarlo")
print("Incertitudes sur : \n", liste_sol2[["DeltaX [m]", "DeltaY [m]", "DeltaZ [m]", "Alpha(Z) [rad]","Beta(X) [rad]", "Gamma(Y) [rad]"]])
print(f"\nTemps d'exécution : {elapsed} s\n")
"""
#%% # Recherche d'un point faux dans la liste de points
"""
nombre_it = 100 # nombre d'itération pour la méthode Montécarlo (calcul d'incertitude pour variation aléatoire gaussienne)
start = time.time()
liste_sol3 = Solution.drop(["Valeurs"])
for item in liste_points_Tracker.index :
    liste_points_R1 = liste_points_GPS.drop(index=item)
    liste_points_R2 = liste_points_Tracker.drop(index=item)
    print("\n\n Nombre de points : ", ligne//nombre_rep-1, "   Point supprimé : ", int(item))
    print("Sigma tracker : ", liste_points_R1["sigma"].max(), "      Sigma GPS : ", liste_points_R2["sigma"].max())
    liste_sol3.loc[item] = calcul_changeur_coord_montecarlo (liste_points_R1, liste_points_R2, nombre_it).loc["Sigma"]
    print("\n  RESULTATS \n\n[deltaX  deltaY  deltaZ  alpha(Z)=Yaw  beta(X)=Pitch  gamma(Y)=Roll] exprimés en mètre et radian")
    print("Solution : \n", liste_sol2[["DeltaX [m]", "DeltaY [m]", "DeltaZ [m]", "Alpha(Z) [rad]","Beta(X) [rad]", "Gamma(Y) [rad]"]])

end = time.time()
print(f"\nTemps d'exécution : {elapsed} s\n")
"""

#%% # calcul de transfert des points de R1 vers R2 et affichage des erreurs
# Solution.at["Valeurs", "Alpha(Z) [rad]"] = Solution.at["Valeurs", "Alpha(Z) [rad]"] + math.pi

liste_points_R1_dans_R2 = TransRotBis(liste_points_GPS, Solution)
liste_points_R1_dans_R2["ErreurX [m]"] = liste_points_Tracker["X [m]"] - liste_points_R1_dans_R2["X [m]"]
liste_points_R1_dans_R2["ErreurY [m]"] = liste_points_Tracker["Y [m]"] - liste_points_R1_dans_R2["Y [m]"]
liste_points_R1_dans_R2["ErreurZ [m]"] = liste_points_Tracker["Z [m]"] - liste_points_R1_dans_R2["Z [m]"]
liste_points_R1_dans_R2["Erreur [m]"] = (liste_points_R1_dans_R2["ErreurX [m]"]**2 + liste_points_R1_dans_R2["ErreurY [m]"]**2 + liste_points_R1_dans_R2["ErreurZ [m]"]**2)**0.5
print("Points GPS dans repère Tracker :\n", liste_points_R1_dans_R2)

"""
# Plot en projection sur X-Y tracker
plotXY = liste_points_R1_dans_R2.plot.scatter(x="X [m]", y="Y [m]", color="DarkBlue", label="GPS dans Tracker", title="H - 3D - TR - projection sur X-Y")
liste_points_Tracker.plot.scatter(x="X [m]", y="Y [m]", color="DarkRed", label="Tracker dans Tracker", ax=plotXY)
"""

# Plot en projection sur X-Y tracker bis avec colormap pour le décalage vertical
plotXYbis = liste_points_R1_dans_R2.plot.scatter(x="X [m]", y="Y [m]", c="Erreur [m]", cmap="viridis", s=50, label="GNSS dans Tracker", title="Points de ref - 3D / Erreur max en couleur")
liste_points_Tracker.plot.scatter(x="X [m]", y="Y [m]", color="Red", s=10, label="Tracker dans Tracker", ax=plotXYbis)

"""
# Plot en projection sur X-Z tracker
plotXZ = liste_points_R1_dans_R2.plot.scatter(x="X", y="Z", color="DarkBlue", label="GPS dans Tracker", title="V - 3D - TR - projection sur X-Z")
liste_points_Tracker.plot.scatter(x="X", y="Z", color="DarkRed", label="Tracker dans Tracker", ax=plotXZ)

# Plot en projection sur Y-Z tracker
plotYZ = liste_points_R1_dans_R2.plot.scatter(x="Y", y="Z", color="DarkBlue", label="GPS dans Tracker", title="V - 3D - TR - projection sur Y-Z")
liste_points_Tracker.plot.scatter(x="Y", y="Z", color="DarkRed", label="Tracker dans Tracker", ax=plotYZ)
"""