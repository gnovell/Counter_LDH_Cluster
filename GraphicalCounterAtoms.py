import sys
import numpy as np
from scipy.spatial import ConvexHull,Delaunay
from sklearn.cluster import DBSCAN

# Parameters of search and replicatipon vectors
counter_atoms = ('AL','ZN','N1','S1','OW','CL')
replica=(3,3,3)
resname_count=['AL','N1','OW','CL','NA','OW','ALc','N1i','S1i','OWi','CLi']

# Atom object, resname, name, and coordenates
class Atomo:
    def __init__(self,residuo,nombre,XYZ):
        self.resname = residuo
        self.name = nombre
        self.XYZ = XYZ

# read function of gro file
def LecturaGRO(archivo):
    archiu=open(archivo,'rt')
    counter=0
    final=3
    sistema=[]
    for linea in archiu :
        if counter == 1 :
            final=int(linea)
        if counter > 1 and counter < final+1 :
            residuo = linea[5:10].strip()
            nombre = linea[10:15].strip()
            XYZ = np.array(linea[20:45].split(), float)
            atomo=Atomo(residuo,nombre,XYZ)
            sistema.append(atomo)
        if counter == final+2 :
            celda=linea.split()
        counter+=1
    Mcell=np.zeros((3,3),float)
    Mcell[0,0]=float(celda[0])
    Mcell[1, 1] = float(celda[1])
    Mcell[2, 2] = float(celda[2])
    Mcell[0, 1] = float(celda[3])
    Mcell[0, 2] = float(celda[4])
    Mcell[1, 0] = float(celda[5])
    Mcell[1, 2] = float(celda[6])
    Mcell[2, 0] = float(celda[7])
    Mcell[2, 1] = float(celda[8])
    archiu.close()
    return(sistema,Mcell)

# function to write the gro format
def WriteGro(datos,cell):
    print('python code of gnovell for multiplication of gro format')
    print(len(datos))
    counter = 1
    countres = 0
    for i in range(len(datos)):
        residuo = str(datos[i].resname)
        atom = str(datos[i].name)
        XYZ = datos[i].XYZ        
        if atom in resname_count:
            countres += 1
        if countres > 99999:
            residuo = '*****' + residuo
        else:
         residuo = str(countres) + residuo
        if counter > 99999:
            numero_atomo = '*****'
        else:
            numero_atomo = counter
        print('{:>8}{:>7}{:>5}{:8.3f}{:8.3f}{:8.3f}'.format(residuo, atom, numero_atomo, *XYZ))
        counter += 1    
    print('{:12.5f}{:12.5f}{:12.5f}{:12.5f}{:12.5f}{:12.5f}{:12.5f}{:12.5f}{:12.5f}'.format(cell[0, 0],
                                                                                            cell[1, 1],
                                                                                            cell[2, 2],
                                                                                            cell[0, 1],
                                                                                            cell[0, 2],
                                                                                            cell[1, 0],
                                                                                            cell[1, 2],
                                                                                            cell[2, 0],
                                                                                            cell[2, 1]))

#filter function to reduce the atom filter
def Filtro(datos,filtro):
    datos_filtrados=[]
    for i in range(len(datos)):
        existe=[elemento for elemento in filtro if(elemento in datos[i].name)]
        if bool(existe):
            datos_filtrados.append(datos[i])
    return(datos_filtrados)

#function to move a coordinate by a vector
def MoveUnit(Matriz_xyz,Vector):
    Traslacion=np.eye(4,4)
    Traslacion[3,0:-1]=Vector
    Matriz_xyz2=np.c_[Matriz_xyz,np.ones(len(Matriz_xyz))]
    NuevasCordenadas=np.dot(Matriz_xyz2,Traslacion)[:,0:-1]
    return(NuevasCordenadas)

#function to replicate the gro files by the replica vector in the first lines of code   
def Multiplo(Mxyz,Mcell,replica):
    MultiploXYZ = []
    counter = 1
    for i in range(replica[0]):
        for j in range(replica[1]):
            for k in range(replica[2]):
                VectorDesplazamiento = np.dot(np.array([i, j, k], float), Mcell)
                MatrizCoordenadas = MoveUnit(Mxyz, VectorDesplazamiento)
                MultiploXYZ.append(MatrizCoordenadas)
                counter += 1
    return(np.array(MultiploXYZ, float))

#function of distances of atom from a matrix of coordinates
def distancia(atomo,matriz):
    M_distancias = np.linalg.norm(matriz - atomo,axis=1)
    return(M_distancias)


#####################################################################################################3

#read the gro file from command line execution of python
archivo = sys.argv[1]
sistema,Mcell=LecturaGRO(archivo)

#Estraction of metal atoms
metales = Filtro(sistema,['AL','ZN'])
MetalesXYZ = np.array([metales[i].XYZ for i in range(len(metales))])

#Replicacation of metal matrix of coordinates
Matriz_replica_MetalesXYZ=np.concatenate(Multiplo(MetalesXYZ,Mcell,replica))

#search of center of all metals
centro=np.array(Matriz_replica_MetalesXYZ.mean(axis=0),float)

#Cutting the metals coordinates from de center with a 10nm of radius
MetalesCentro=[]
distanciaCentro = distancia(centro, Matriz_replica_MetalesXYZ)
while distanciaCentro.min() < 10.0:
    MetalesCentro.append(Matriz_replica_MetalesXYZ[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
MetalesCentro=np.array(MetalesCentro[:])
#Localization of clusters layers with the DBSAN algoritm from scikit-learn module
sc=DBSCAN(eps=0.35,min_samples=5).fit(MetalesCentro)
Capas=[]
for i in range(5):
    capa=np.array(MetalesCentro[sc.labels_ == i],float)
    if len(capa) > 250:
        Capas.append(capa)
MetalesCentroXYZ=np.concatenate(Capas[:])

#Recalculate the center of cluster layers
centro=MetalesCentroXYZ.mean(axis=0)
distanciasMaxMetalesCentro=distancia(centro,MetalesCentroXYZ).max()
#eliminate variables to reduce the use of memory
del MetalesCentro,Matriz_replica_MetalesXYZ

#selection of atoms around the center of the metal cluster
#for each atom, we replicate and get the coordinates near the center of the cluster
sistema_centro=[]
for i in range(len(sistema)):
    residuo=sistema[i].resname
    XYZ=sistema[i].XYZ
    replicadoXYZ=np.concatenate(Multiplo([XYZ],Mcell,replica))
    distanciaCentro=distancia(centro,replicadoXYZ)
    while distanciaCentro.min() < distanciasMaxMetalesCentro + 1.0:
        nuevo_atomo=Atomo(residuo,sistema[i].name,replicadoXYZ[distanciaCentro.argmin()])
        sistema_centro.append(nuevo_atomo)
        distanciaCentro[distanciaCentro.argmin()] = 1000

# Filtramos atomos de Cloro, nitrato y agua para buscar los que hay dentro del cluster metalico
#Description of edges of metal cluster of LDH layers
hull=ConvexHull(MetalesCentroXYZ,qhull_options='QJ')
#Coordenates of ConvexHull formation and marked as CAP and AU.
hull_XYZ=list(hull.points)
atomo_hull=list(map(lambda x: sistema_centro.append(Atomo('CAP','AU',x)),hull_XYZ))
#Search the Cl, NO3, MBT and waters inside the ComnvexHull
hull=Delaunay(hull.points)
AtomosFiltrados=Filtro(sistema_centro,['CL','N1','S1','OW'])
AtomosInside=[]
for atomo in AtomosFiltrados:
    x = hull.find_simplex(atomo.XYZ)
    if x > 0 :
        AtomosInside.append(atomo)

for atomo in AtomosInside:
    if atomo.name == 'CL':
        atomo.name='CLi'
    if atomo.name == 'N1':
        atomo.name='N1i'
    if atomo.name == 'S1':
        atomo.name='S1i'
    if atomo.name == 'OW':
        atomo.name='OWi'
#Write the GRO file format
WriteGro(sistema_centro,Mcell)
