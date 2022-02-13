import sys
import numpy as np
from scipy.spatial import ConvexHull,Delaunay
from sklearn.cluster import DBSCAN

#Number of Cell replication in XYZ directions
Replica= (3, 3, 3)

#Function to read the gro file and take The metal atoms(Al,Zn), Cl, N1 of Nitrate, and OW of water and the Cell matrix.  
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

def MoveUnit(Matriz_xyz,Vector):
    Traslacion=np.eye(4,4)
    Traslacion[3,0:-1]=Vector
    Matriz_xyz2=np.c_[Matriz_xyz,np.ones(len(Matriz_xyz))]
    NuevasCordenadas=np.dot(Matriz_xyz2,Traslacion)[:,0:-1]
    return(NuevasCordenadas)

def Filtro(datos,filtro):
    datos_Metal=[]
    for i in range(len(datos)):
        existe=[elemento for elemento in filtro if(elemento in datos[i][1].strip())]
        if bool(existe):
            datos_Metal.append(datos[i])
    return(datos_Metal)

def layers(matriz_conexiones,lineas_layer,inicio):
    lista_id = []
    fin = 0
    lista_id.append(inicio)
    counter = 0
    while fin == 0:
        lista = np.hstack(np.where(matriz_conexiones[inicio] == 1)).tolist()
        for x in lista:
            if x not in lista_id:
                lista_id.append(x)
        counter += 1
        if counter == len(lista_id):
            fin = 1
        else:
            inicio = lista_id[counter]
#    print(counter)
    lista = []
    for i in range(len(lista_id)):
        #    lista.append(lineas_layer[i])
        lista.append(lineas_layer[lista_id[i]]-1)
#    print(*lista)
    return(lista_id,lista)

def distancia(Atomo,Matriz):
    M_distancias = np.linalg.norm(Matriz - Atomo,axis=1)
    return(M_distancias)

def Centro(Matriz):
    centrado=filter(lambda x: Matriz[x] < 7.0, range(len(Matriz)))
    return(list(centrado))

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

def Reduccion(Matriz,lista):
    matriz=map(lambda x: Matriz[x],lista)
    return list(matriz)

########################################################################################################################
########################################################################################################################

#leer archivo gro
#archivo='md_2.gro'
archivo = sys.argv[1]
Metales,Cloros,Nitratos,Aguas,Mcell=LecturaGRO(archivo)

# duplicar celdas por 3
MultiploMetales=np.concatenate(Multiplo(Metales,Mcell))
# Buscar centro de los metales
centro=np.array(MultiploMetales.mean(axis=0),float)
# Cortar metales a un radio del 10nm del centro
MetalesCentro=[]
distanciaCentro = distancia(centro, MultiploMetales)
while distanciaCentro.min() < 10.0:
    MetalesCentro.append(MultiploMetales[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
MetalesCentro=np.array(MetalesCentro[:])
# Localizar Capas y en centro del cluster
sc=DBSCAN(eps=0.35,min_samples=5).fit(MetalesCentro)
Capas=[]
for i in range(5):
    capa=np.array(MetalesCentro[sc.labels_ == i],float)
    if len(capa) > 250:
        Capas.append(capa)
MetalesCentroXYZ=np.concatenate(Capas[:])

#Centro de las capas de Metales
centro=MetalesCentroXYZ.mean(axis=0)
distanciasMaxMetalesCentro=distancia(centro,MetalesCentroXYZ).max()

del MultiploMetales,MetalesCentro
# Clasificar atomos de Cloro
AtomosCentro=[]
MultiploXYZ = np.concatenate(Multiplo(Cloros, Mcell))
distanciaCentro = distancia(centro, MultiploXYZ)
while distanciaCentro.min() < distanciasMaxMetalesCentro + 1.0:
    AtomosCentro.append(MultiploXYZ[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
Cloros=np.array(AtomosCentro[:])
# Clasificar atomos de N1, centro del NOO
AtomosCentro=[]
MultiploXYZ = np.concatenate(Multiplo(Nitratos, Mcell))
distanciaCentro = distancia(centro, MultiploXYZ)
while distanciaCentro.min() < distanciasMaxMetalesCentro + 1.0:
    AtomosCentro.append(MultiploXYZ[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
Nitratos=np.array(AtomosCentro[:])
# Clasificar atomos de OW, centro de la molecula de agua
AtomosCentro=[]
MultiploXYZ = np.concatenate(Multiplo(Aguas, Mcell))
distanciaCentro = distancia(centro, MultiploXYZ)
while distanciaCentro.min() < distanciasMaxMetalesCentro + 1.0:
    AtomosCentro.append(MultiploXYZ[distanciaCentro.argmin()])
    distanciaCentro[distanciaCentro.argmin()] = 1000
Aguas=np.array(AtomosCentro[:])

del AtomosCentro,MultiploXYZ

#Descripcion de los bordes de las capas de metales
hull=ConvexHull(MetalesCentroXYZ,qhull_options='QJ')
MaxVol=hull.volume
#print('Volumen maximo : ',MaxVol)
#
#Busqueda entre capas de los Cloros, Nitratos y Aguas
hull=Delaunay(hull.points)
counter_CL=len(list(filter(lambda x: x > 0,hull.find_simplex(Cloros))))
counter_N1=len(list(filter(lambda x: x > 0,hull.find_simplex(Nitratos))))
counter_OW=len(list(filter(lambda x: x > 0,hull.find_simplex(Aguas))))

print(archivo,'  ','Volumen maximo: ',MaxVol,'  Total Numbers of Clorine inside: ',counter_CL,'  Total Nitrates inside: ',counter_N1,'  Total Aguas inside; ',counter_OW)
