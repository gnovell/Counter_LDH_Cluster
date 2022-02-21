import sys
import numpy as np
from scipy.spatial import ConvexHull,Delaunay
from sklearn.cluster import DBSCAN

#number of Cell replication in XYZ directions
Replica= (3, 3, 3)

#function to read the gro file and take The metal atoms(Al,Zn), Cl, N1 of Nitrate, and OW of water and the Cell matrix.  
def LecturaGRO(archivo):
    archiu=open(archivo,'rt')
    counter=0
    final=3
    Metales=[]
    Cloros=[]
    Nitratos=[]
    Aguas=[]
    for linea in archiu :
        if counter == 1 :
            final=int(linea)
        if counter > 1 and counter < final+1 :
            atomo=linea[12:15].strip()
            if 'AL' in atomo or 'ZN' in atomo:
                Metales.append(linea[20:45].split())
            if 'CL' in atomo :
                Cloros.append(linea[20:45].split())
            if 'N1' in atomo:
                Nitratos.append(linea[20:45].split())
            if 'OW' in atomo:
                Aguas.append(linea[20:45].split())
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
    return(np.array(Metales,float),np.array(Cloros,float),np.array(Nitratos,float),np.array(Aguas,float),Mcell)

#function to translate a matrix by a vector 
def MoveUnit(Matriz_xyz,Vector):
    Traslacion=np.eye(4,4)
    Traslacion[3,0:-1]=Vector
    Matriz_xyz2=np.c_[Matriz_xyz,np.ones(len(Matriz_xyz))]
    NuevasCordenadas=np.dot(Matriz_xyz2,Traslacion)[:,0:-1]
    return(NuevasCordenadas)

#function to calculate a distance of a matrix from an atom
def distancia(Atomo,Matriz):
    M_distancias = np.linalg.norm(Matriz - Atomo,axis=1)
    return(M_distancias)

#function to replicate a matrix with a cell units by a Replica vector
def Multiplo(Mxyz,Mcell):
    MultiploXYZ = []
    counter = 1
    for i in range(Replica[0]):
        for j in range(Replica[1]):
            for k in range(Replica[2]):
                VectorDesplazamiento = np.dot(np.array([i, j, k], float), Mcell)
                MatrizCoordenadas = MoveUnit(Mxyz, VectorDesplazamiento)
                MultiploXYZ.append(MatrizCoordenadas)
                counter += 1
    return(np.array(MultiploXYZ, float))

########################################################################################################################
########################################################################################################################

#select the gro file
archivo = sys.argv[1]
#extract the information from gro file, Metals, Clorhides, Nitrates, Water, and the unit cell
Metales,Cloros,Nitratos,Aguas,Mcell=LecturaGRO(archivo)

#replicate the metals by 3x3x3 unit cells
MultiploMetales=np.concatenate(Multiplo(Metales,Mcell))
#locate the center of metals
centro=np.array(MultiploMetales.mean(axis=0),float)
#Select the metals around 10nm of center
MetalesCentro=[]
distanciaCentro = distancia(centro, MultiploMetales)
while distanciaCentro.min() < 10.0:
    MetalesCentro.append(MultiploMetales[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
MetalesCentro=np.array(MetalesCentro[:])
#locate the layers of cluster by DBSCAN with a distance of 0.35nm
sc=DBSCAN(eps=0.35,min_samples=5).fit(MetalesCentro)
#select the layers bigger than 250 metals
Capas=[]
for i in range(5):
    capa=np.array(MetalesCentro[sc.labels_ == i],float)
    if len(capa) > 250:
        Capas.append(capa)
MetalesCentroXYZ=np.concatenate(Capas[:])

#locate the center of metal layers
centro=MetalesCentroXYZ.mean(axis=0)
#maximum distance of metal from the center
distanciasMaxMetalesCentro=distancia(centro,MetalesCentroXYZ).max()
#clean the memory
del MultiploMetales,MetalesCentro
#select the clorhide atoms from the center to 1nm outside the cluster
AtomosCentro=[]
MultiploXYZ = np.concatenate(Multiplo(Cloros, Mcell))
distanciaCentro = distancia(centro, MultiploXYZ)
while distanciaCentro.min() < distanciasMaxMetalesCentro + 1.0:
    AtomosCentro.append(MultiploXYZ[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
Cloros=np.array(AtomosCentro[:])
#select the N1 atoms (N atom of nitrite) from the center to 1nm outside the cluster
AtomosCentro=[]
MultiploXYZ = np.concatenate(Multiplo(Nitratos, Mcell))
distanciaCentro = distancia(centro, MultiploXYZ)
while distanciaCentro.min() < distanciasMaxMetalesCentro + 1.0:
    AtomosCentro.append(MultiploXYZ[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
Nitratos=np.array(AtomosCentro[:])
#select the OW atoms (O atom of water molecules) from the center to 1nm outside the cluster
AtomosCentro=[]
MultiploXYZ = np.concatenate(Multiplo(Aguas, Mcell))
distanciaCentro = distancia(centro, MultiploXYZ)
while distanciaCentro.min() < distanciasMaxMetalesCentro + 1.0:
    AtomosCentro.append(MultiploXYZ[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
Aguas=np.array(AtomosCentro[:])
#clean the memory
del AtomosCentro,MultiploXYZ

#location of the hull for the Metals cluster
hull=ConvexHull(MetalesCentroXYZ,qhull_options='QJ')
#calculate the volume of cluster
MaxVol=hull.volume
#locate the Clorhide, Nitrates, and Waters inside the cluster
hull=Delaunay(hull.points)
counter_CL=len(list(filter(lambda x: x > 0,hull.find_simplex(Cloros))))
counter_N1=len(list(filter(lambda x: x > 0,hull.find_simplex(Nitratos))))
counter_OW=len(list(filter(lambda x: x > 0,hull.find_simplex(Aguas))))
#print the Volume, the number of Chloride, Nitrates, and Waters inside the cluster
print(archivo,'  ','Volume max.: ',MaxVol,'  # Chloride inside: ',counter_CL,'  # Nitrates inside: ',counter_N1,'  # Waters inside; ',counter_OW)
