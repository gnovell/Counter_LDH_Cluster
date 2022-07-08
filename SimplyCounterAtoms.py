import sys
import numpy as np
from scipy.spatial import ConvexHull,Delaunay
from sklearn.cluster import DBSCAN

class Atom:
    def __init__(self,linea):
        self.resid = linea[:5].strip()
        self.resname = linea[5:10].strip()
        self.name = linea[10:15].strip()
        self.id = linea[15:20].strip()
        self.XYZ = np.array(linea[20:45].split(),float)

def LecturaGRO(archivo):
    archiu=open(archivo,'rt')
    counter=0
    final=3
    Sistema=[]
    for linea in archiu :
        if counter == 1 :
            final=int(linea)
        if counter > 1 and counter < final+1 :
            atomo=Atom(linea[:45])
            Sistema.append(atomo)
        if counter == final+2 :
            Cell=linea.split()
        counter+=1
    Mcell=np.zeros((3,3),float)
    Mcell[0,0]=float(Cell[0])
    Mcell[1, 1] = float(Cell[1])
    Mcell[2, 2] = float(Cell[2])
    Mcell[0, 1] = float(Cell[3])
    Mcell[0, 2] = float(Cell[4])
    Mcell[1, 0] = float(Cell[5])
    Mcell[1, 2] = float(Cell[6])
    Mcell[2, 0] = float(Cell[7])
    Mcell[2, 1] = float(Cell[8])
    archiu.close()
    return(Sistema,Mcell)

def Filtro(datos,filtro):
    datos_filtrados=[]
    for i in range(len(datos)):
        existe=[elemento for elemento in filtro if(elemento in datos[i].name)]
        if bool(existe):
            datos_filtrados.append(datos[i])
    return(datos_filtrados)

########################################################################################################################
########################################################################################################################

#Read de gro file
archivo = sys.argv[1]
#Extraction of data
Sistema,Mcell=LecturaGRO(archivo)

#List and Matrix of metals Al and Zn
Metales = Filtro(Sistema,['AL','ZN'])
MetalesXYZ = np.array([Metales[i].XYZ for i in range(len(Metales))])

#Locate the layers of metals from LDH
sc=DBSCAN(eps=0.35,min_samples=5).fit(MetalesXYZ)
Capas=[]
for i in range(5):
    capa=np.array(MetalesXYZ[sc.labels_ == i],float)
    if len(capa) > 250:
        Capas.append(capa)
MetalesCapas=np.concatenate(Capas[:])

# Extraction of Cl atoms, Nitrogen of nitrate, and Oxigen of water.
Atomos_Cloro = Filtro(Sistema,['CL'])
Atomos_Nitrato = Filtro(Sistema,['N1'])
Atomos_Aguas = Filtro(Sistema,['OW'])
# Matrix of coordenates from respective atoms
ClorosXYZ = np.array([Atomos_Cloro[i].XYZ for i in range(len(Atomos_Cloro))])
NitratosXYZ = np.array([Atomos_Nitrato[i].XYZ for i in range(len(Atomos_Nitrato))])
AguasXYZ = np.array([Atomos_Aguas[i].XYZ for i in range(len(Atomos_Aguas))])

#The hull points that describe the layer metals of LDH
hull=ConvexHull(MetalesCapas,qhull_options='QJ')
#Calculate the volume of hull from layer metals of LDH
MaxVol=hull.volume

#Description of points of hull to search the atoms inside the hull
hull=Delaunay(hull.points)

# Search the atoms of Cl, N, and O inside the hull of LDH metals
Inside_Cloro=[]
Inside_Nitrato=[]
Inside_Aguas=[]

for i in range(len(ClorosXYZ)):
    if hull.find_simplex(ClorosXYZ[i]) >= 0:
        Inside_Cloro.append(Atomos_Cloro[i])
for i in range(len(NitratosXYZ)):
    if hull.find_simplex(NitratosXYZ[i]) >= 0:
        Inside_Nitrato.append(Atomos_Nitrato[i])
for i in range(len(AguasXYZ)):
    if hull.find_simplex(AguasXYZ[i]) >= 0:
        Inside_Aguas.append(Atomos_Aguas[i])

# Print the final results
print(archivo,'  Volumen LDH : ',MaxVol,'  Clor inside: ',len(Inside_Cloro),'  Nitrato inside: ',len(Inside_Nitrato),'  Aguas Inside: ',len(Inside_Aguas))
