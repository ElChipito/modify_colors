import pydicom
import os
import numpy as np
import pandas as pd
from pydicom.dataset import Dataset

def spacing(min_cd8 = 0.8, max_cd8 = 2.8) :                    # On crée dans cette fonction un intervalle de 16 valeurs                         
    cd8_values = np.linspace(min_cd8, max_cd8, 11)             # Ca varie de min à max   
    spacings = [[cd8_values[i], cd8_values[i+1]] for i in range(0, len(cd8_values)-1)]
    spacing_dict = {i + 1: spacing for i, spacing in enumerate(spacings)}  # On return un dictionnaire qui contient dans les values les intervales
    return spacing_dict # Chaque value est une liste de deux éléments du type [0, 0.2]

def colors_gradient() :      # On crée à présent le gradient de couleur poru modifier la couleur des tumeurs
    gradient =  [(0, 0, 171), (56, 56, 255), (113, 113, 255), (170, 170, 255), (205, 205, 255), 
                (255, 205, 205), (255, 170, 170), (255, 113, 113), (255, 56, 56), (171, 0, 0)]
    
    colors_dict = {i + 1: color for i, color in enumerate(gradient)} # On return un dictionnaire de la même taille que la fct précédente
    return colors_dict       # Chaque value contient une couleur

def get_roi_names_and_numbers_rtstruct(rtstruct_path) : # On récupère ici les noms et les numéros de toutes les tumeurs d'un RTSTRUCT
    try :                                               # Petite gestion d'erreur au as où ce n'est pas un bon fichier
        ds = pydicom.dcmread(rtstruct_path, force=True)
    except pydicom.errors.InvalidDicomError as e :
        print("Ce n'est pas un RTSTRUCT, attention ! ", e)

    roi_names = [roi.ROIName for roi in ds.StructureSetROISequence]     # On extrait la liste des noms des tumeurs
    roi_numbers = [roi.ROINumber for roi in ds.StructureSetROISequence] # On extrait les numéros
     
    return roi_names, roi_numbers 

def get_color(cd8, spacing_dict, color_dict) :     # Cette fonction permet d'associer une couleur à un score cd8
    for key, value in spacing_dict.items() :       # On parcourt les intervalles
        if cd8 >= value[0] and cd8 < value[1] :    # Et on regarde si le score cd8 est compris dedans 
            cd8_number = key                       # Si c'est le cas, on récupère la clef de l'intervalle
    color_number = color_dict[cd8_number]          # Les clefs étant les mêmes dans les deux dictionnaires, on récupère la couleur
    return color_number 

def check_and_split(name) :                 # On crée ici la fonction qui nosu permet de check le nom des roi name et de les modifier
    name_list = name.split('_')             # On split le nom du roi
    if len(name_list) > 4 :                 # La taille du roi original est de 4, donc on vérifie s'il est plus long ou pas 
        name = '_'.join(name_list[:-2])     # Si c'est plus long, on tronque le nom
    return name                             # Tronqué ou non, on renvoie le nom, soit original, soit reconstitué  

def clean_path(path):                       # C'est une toute petite fonction mais qui évitr trop de syntaxe plus tard 
    return path.replace(" ", "")            # Dans le fichier csv, des espaces se sont glissés dans les paths, on les enlèves

def find_min(df):
    min_scores = df.groupby('path RTSTRUCT').apply(lambda x: x.loc[x['Score CD8'].idxmin()])
    return min_scores

def create_box_around_tumor(ds, tumor_contour_data, box_name) :          # Cette fonction permet de créer une "box" autour d'une tumeur

    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')       # On commence par introduire les 6 varables qui nous intéressent
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf') 

    for contour in tumor_contour_data:         # On parcourt le contour à la recherche des min et des max de la tumeur
        points = [float(p) for p in contour]   # Et sur les trois axes
        xs = points[0::3]
        ys = points[1::3]
        zs = points[2::3]
        min_x = min(min_x, *xs)
        max_x = max(max_x, *xs)
        min_y = min(min_y, *ys)
        max_y = max(max_y, *ys)
        min_z = min(min_z, *zs)
        max_z = max(max_z, *zs)

    center_x = (min_x + max_x) / 2     # On calcule le centre de la tumeur selon chaque axe pour centrer plus tard la box
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2

    min_x = center_x - (max_x - min_x) # On calcule les mins et les max de la box
    max_x = center_x + (max_x - min_x) # En prenant la longueur de la tumeur à partir du centre
    min_y = center_y - (max_y - min_y) # Ca permet de délimiter !
    max_y = center_y + (max_y - min_y)
    min_z = center_z - (max_z - min_z)
    max_z = center_z + (max_z - min_z)

    box_contour_data = [               # Et puis on "dessine" la box de sorte à obtenir un contour convenable
        min_x, min_y, min_z,           
        max_x, min_y, min_z,          
        max_x, max_y, min_z,           
        min_x, max_y, min_z,          
        min_x, min_y, min_z,          
        min_x, min_y, max_z,
        max_x, min_y, max_z,
        max_x, max_y, max_z,
        min_x, max_y, max_z,
        min_x, min_y, max_z           
    ]

    new_roi_contour = Dataset()        # On crée ensuite une nouvelle entrée ROIContourSequence pour la box
    new_roi_contour.ReferencedROINumber = len(ds.StructureSetROISequence) + 1 # on rajoute un nombre qui sera systématiquement le dernier de la liste
    new_roi_contour.ROIDisplayColor = [255, 255, 255]  # Et puis une couleur (le blanc ici)
    new_roi_contour.ContourSequence = []

    contour_seq = Dataset()            # On crée une nouvelle entrée pour la géométrie de la boite
    contour_seq.ContourGeometricType = 'CLOSED_PLANAR'    
    contour_seq.NumberOfContourPoints = len(box_contour_data) // 3  # On divise par trois pour avec les coordonées dans l'espace
    contour_seq.ContourData = [str(x) for x in box_contour_data]    # On remplie le contour  
    new_roi_contour.ContourSequence.append(contour_seq)             # Et on remplie le roi contour crée juste avant  
    ds.ROIContourSequence.append(new_roi_contour)                   # Et on envoie !

    new_struct_set_roi = Dataset()                # On crée enfin une nouvelle entrée StructureSetROISequence 
    new_struct_set_roi.ROINumber = new_roi_contour.ReferencedROINumber # On prend le même numéro que le roi contour
    new_struct_set_roi.ROIName = box_name                    # On prend le nom de la box (le nom de la tumeur)
    new_struct_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'  # ligne importante : il faut un algorithme de génération !
    ds.StructureSetROISequence.append(new_struct_set_roi)    # Et on envoie !


def box(min_scores) :                                 # On crée une nouvelle fonction qui implémente les boites sur les rtstructs
    for idx, min_score in min_scores.iterrows() :     # On parcours le csv d'entrée (celui qui contient les tumeurs concernées par les box)
        rtstruct_path = min_score['path RTSTRUCT']    # On prend les path RTSTRUCT
        rtstruct_path = clean_path(rtstruct_path)     # On les clean
        roi_name = min_score['ROIname']               # Et on prend le roi name (qui nous servira aussi comme nom de la box)
        
        ds = pydicom.dcmread(rtstruct_path, force=True)  # On prend le fichier rtstruct
        roi_names, roi_numbers = get_roi_names_and_numbers_rtstruct(rtstruct_path) # On récupère les noms des roi et leur nuémros
        for name in roi_names :                  # Et on fait parile que dans la fonction modify, on check si les noms correspondent
            name1 = check_and_split(name)
            if name1 in roi_name :               # Et si ca match
                index_tum = roi_names.index(name)        # On récupère l'index
                number_tum = int(roi_numbers[index_tum]) # Et le numéro

                tumor_contour_data = [contour_seq.ContourData for roi_contour in ds.ROIContourSequence # Cette ligne n'est certe pas très belle
                                      if roi_contour.ReferencedROINumber == number_tum  # Mais elle est opti ! On va chercher le contour de la tumeur
                                      for contour_seq in roi_contour.ContourSequence]    
                if tumor_contour_data :  
                    create_box_around_tumor(ds, tumor_contour_data, f"Box_{roi_name}")  # On rajoute la box
  
        new_file_path = os.path.splitext(rtstruct_path)[0] + ".dcm"    # Et puis on save dans un fichier dicom
        ds.save_as(new_file_path)
        print(f"Fichier avec boîte enregistré sous : {new_file_path}")
        
        
def modify_rtstruct(cd8, rtstruct_path, roi_name, path, cut = False, supr_ring = True, box = False, spacing=spacing(), colors_dict=colors_gradient()):     # Fonction principale
    # Dans cette fonction, on teste une seule ligne du fichier csv, c'est à dire qu'on a seulement un score cd8, un roi_name, et un ficheir rtstruct
    # On crée ensutie une autre fonction dans laquelle on évolue dans le csv
    try :                                          # On vérifie que le fichier déposé est un RTSTRUCT
        ds = pydicom.dcmread(rtstruct_path, force=True)
    except pydicom.errors.InvalidDicomError as e :
        print("Tu t'es planté, ce n'est pas un RTSTRUCT :", e)  
    
    color = list(get_color(cd8, spacing, colors_dict)) # On récupère la couleur lié au score cd8 donné en entrée

    # On récupère ensuite les noms et les numéros des tumeurs dans le RTSTRUCT pour pouvoir trouver notre tumeur et la modifier
    name_in_rtstruct, number_in_rtstruct = get_roi_names_and_numbers_rtstruct(rtstruct_path) 
    name_in_excel = roi_name    # On prend le roi_name donné en entré (celui dans le csv)
    for name in name_in_rtstruct :                  # On parcourt l'ensemble des noms de tumeurs présent dans le fichier RTSTRUCT
        name1 = check_and_split(name)
        if name1 in name_in_excel :                 # On compare avec le nom présent dans l'excel et dès qu'on tombe sur le bon...   
            index_tum = name_in_rtstruct.index(name)        # On récupère l'index du nom dans le RTSTRUCT
            number_tum = int(number_in_rtstruct[index_tum]) # C'est le même que l'index de son numéro, donc on récupère le numéro
            number_ring = number_tum + 1                    # Et celui du ring, qui suit le numéro tout le temps
            for j in ds.ROIContourSequence :                # On parcours tous les ROI
                if j.ReferencedROINumber == number_tum :    # Et si on tombe sur numéro de la tumeur 
                    j.ROIDisplayColor = color               # On modifie la couleur de cette dernière (ring compris), enfin !
                elif j.ReferencedROINumber == number_ring and supr_ring == True : # On supprime les rings si ils n'ont pas encore été enlevés
                    ds.ROIContourSequence.remove(j)         # Il faut bien vérifier qu'ils n'ont pas été supr, sinon on risque de supprimer des tumeurs
            
            if cut == True :
                for roi in ds.StructureSetROISequence :        # On parcours les différents roi
                    if roi.ROINumber == number_tum :           # On récupère la tumeur qui nous intéresse   
                        if cd8 >= 1.9 :                        # Si le cd8 est supérieur ou égal à 1.9
                            roi.ROIName = f"{name1}_{cd8:.2f}_hot" # Alors la tumeur est considérée comme chaude
                        if cd8 < 1.9 :                         # Sinon elle sera considérée comme froide
                            roi.ROIName = f"{name1}_{cd8:.2f}_cold" # On modifie alors le nom de la tumeur pour avoir le score cd8 
                                      
    new_file_path = os.path.splitext(rtstruct_path)[0] + ".dcm"  # On finit par sauvegarder le fichier, couleurs modifiées 
    ds.save_as(new_file_path)
    print(f"Fichier modifié enregistré sous : {new_file_path}")  # On fait un petit print pour valider la modif
    
    if box == True :
        df = pd.read_csv(path)    # On commence par lire le fichier 
        min_scores = find_min(df) # On utilise min_scores pour trouver les scores CD8 les plus bas pour chaque fichier RTSTRUCT
        box(min_scores)           # On ajoute des boîtes blanches autour des tumeurs avec les scores CD8 les plus bas poru chaque patient et chaque rendez_vous
        
def modify_all_rtstructs(path) :        # C'est la fonction qui modifie tous les RTSTRUCT présent dans un csv
    df = pd.read_csv(path)                                     # On récupère la dataframe
    df = df[["Score CD8",'ROIname', "path RTSTRUCT"]]          # On ne garde que les trois colonnes qui nous intéresse
    i = 0                               # on prend un compteur qui nous permettra de jouer avec les indices
    for index, row in df.iterrows():    # On parcours toutes les lignes de la dataframe
        cd8 = row[0]                    # On prend le score cd8
        roi_name = row[1]               # Le roi_name
        rtstruct_path = row[2]          # Et puis le chemin vers le rtstruct 
        rtstruct_path = clean_path(rtstruct_path)   # Il y avait beaucoup d'espace dans le nom des chemin, alors on les enlève en espérant que ca focntionne
        
        modify_rtstruct(cd8, rtstruct_path, roi_name, path)  # On injecte dans la fonction du dessus pour changer la couleur de la tumeur
        print(i, '/{}'.format(len(df)-1))              # On print les étapes (On peut mettre le nombre de ligne total de la dataframe)
        i+=1                  

#%% ================= CHANGEMENT GLOBAL =======================================================================================================================

path = "C:/Users/mateo/OneDrive/Desktop/Gustave Roussy/3 - Tumeur_couleur/CD8_scores.csv"
modify_all_rtstructs(path)

#%% ================ TEST LOCAL ===============================================================================================================================
path = "C:/Users/mateo/OneDrive/Desktop/Gustave Roussy/3 - Tumeur_couleur/CD8_scores.csv"
rtstruct_path = "C:/Users/mateo/OneDrive/Desktop/Gustave Roussy/3 - Tumeur_couleur/patient_2/2021_14104_SZ/E0/RTSTRUCT/RS1.2.752.243.1.1.20231206100302519.5000.12404.dcm"
df = pd.read_csv(path)
cd8 = df['Score CD8'].iloc[9]
roi_name = df['ROIname'].iloc[9]
modify_rtstruct(cd8, rtstruct_path, roi_name)