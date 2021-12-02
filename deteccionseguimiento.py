

# importar librerias 
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import jetson.inference
import jetson.utils
#----------------------------------------------------- DETECCION -------------------------------------------------------

class detectormobile:

    def iniciar(frame,DNN):
        
        Cuda = jetson.utils.cudaFromNumpy(frame)
        #Transformar a Cuda desde numpy        

        # Detector
        detector = DNN.Detect(Cuda)
		
        return detector

    def detectarpersonas(detector,mobilenet,x_fin,x_ini,y_ini,y_fin,listacentros):
		# For para cada Objeto
        lpc_count=0
        rects =[]
        objeto = []
        centrox = 2500
        centroy = 2500

        for d in detector:

            # confiabilidad de la deteccion
            Clase = mobilenet.GetClassDesc(d.ClassID)
            confiabilidad = d.Confidence

            # si la probabilidad de que sea un objeto es mayor a 50%
            if confiabilidad > 0.25:
                
                # si la clase "no es una persona":
                if Clase != "person":
                    # inicie de nuevo el ciclo for
                    continue

                # caja del detector
                person_box = [int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)]
                
                # tamaño de la caja de la persona*por el frame = caja en el frame
                (startX, startY, endX, endY) = person_box
                
                #calcular centro del recuadro
                centrox = (startX + endX) / 2
                centroy = endY
                
                #contar solo si esta dentro del Sector
                if centrox <= x_fin[0] and centrox>= x_ini[0]:

                    if centroy >= y_ini[1] and centroy <= y_fin[1]:
                        rects.append(person_box)
                        listacentros.append((centrox,centroy))
		
        return rects, centrox, centroy



# ---------------------------------------------------- FILTRO-----------------------------------------------------------
class non_max_suppression_fast:
    def __init__(self,cajas, thresh):
        self.ini = 0
    def input(cajas, thresh):
    
   		# si las cajas estan vacias retornar vacio
        if len(cajas) == 0:
            return []

    	# si las cajas son de tipo string transformar a float
        if cajas.dtype.kind == "i":
            cajas = cajas.astype("float")

        pick = []
		
		# Obtener valores de las cajas
        x1 = cajas[:, 0]
        y1 = cajas[:, 1]
        x2 = cajas[:, 2]
        y2 = cajas[:, 3]

		# Calcular el area y ordenar segun el "y" mas bajo
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

		#mientras idx sea mayor a 0
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
			
			# se obtienes los valores de las cajas mas grandes 
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

			# obtener la altura y el ancho
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

			#obtener el overlap
            overlap = (w * h) / area[idxs[:last]]

            # eliminar las otras cajas
            idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > thresh)[0])))

        return cajas[pick].astype("int")

# ---------------------------------- SEGUIMIENTO DE PERSONAS ---------------------------------------------


class CentroidTracker:
    def __init__(self, maxDisappeared, maxDistance):
        # Inicializar Variables
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # Guardar en self el numero maximo de "frames"con el 
		# Cual "desaparecera" el seguimiento del centroide
        self.maxDisappeared = maxDisappeared

        # Guardar en self la maxima distancia entre centroides 
		# para que "desaparezca" el seguimiento
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        # Registrar un nuevo objeto en otro
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # El desregistro se basa a partir de los dos diccionarios eliminar el objectID
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  

    def update(self, rects):
        # Mirar si bbox del detector esta vacio
        if len(rects) == 0:
            # Hacer un ciclo con todos los objetos que estan siendo seguidos
	    	# y marcarlos como desaparecidos 
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # Si se supero el numero maximo de frames antes de desaparecer 
				# Se desrigistra
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Retornar bbox sin informacion de seguimiento del objeto
            return self.bbox

        # Si bbox no esta vacio
		# inicializa el arreglo de centroides de entrada 
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
        # se hace el loop sobre las bbox que entran
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # obtener el centro de los bbox y guardarlo en la lista de entrada
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i]) 

        # si actualmente no se estan siguiendo objetos registrarlos
		# para el seguimiento
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE

        # si si se estan siguiendo objetos hay que hacer un match 
		# Entre los inputs de entrada con los que ya se estan siguiendo
        else:
            # guardar en lista los centroides con sus respectivos objectsID
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # calcular la distancia entre cada par de centroides (actuales, nuevos)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # primero se toma e l valos mas peque;o en cada fila 
	    	# y luego se ordena el index segun el valor mas pequeno, 
	    	# mandandolo al frente de la lista
            rows = D.min(axis=1).argsort()

            # se realiza el mismo procedimiento en columnas
            cols = D.argmin(axis=1)[rows]

            # se inicializan variables de filas usadas y columnas usadas 
	    	# las cuales diran si es necesario hacer registro, desregistro o actualizar
	   	 	# el seguimiento
            usedRows = set()
            usedCols = set()

            # hacer loop sobre la filas y columnas
            for (row, col) in zip(rows, cols):
                # si ya se examinaron ignorar
                if row in usedRows or col in usedCols:
                    continue

                # si seperan la maxima distancia ignorar 
                if D[row, col] > self.maxDistance:
                    continue

                # si no es ninguno de los casos anteriores guarda el ObjectID 
				# para la fila actual, este va a ser el nuevo centroide 
				# y reinicial el contador de objectID desaparecidos
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0

                # indicar las columnas que ya fueron calculadas
                usedRows.add(row)
                usedCols.add(col)

            # calcular las filas y columnas que no fueron usadas
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # dado el caso en el que el numero de centroides sea igual o 
	    	# mayor al numero de numero de centroides de entrada 
	    	# se necesita ver si algunos de estos objetos deben desaparecer
		
            if D.shape[0] >= D.shape[1]:
                # hacer loop sobre las filas no usadas
                for row in unusedRows:
                    # guardar el object ID en la fila correspondiente 
		    		# y sumar el contador de objectID desaparecidos
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # verificar si el numero de objects ha pasado si si hacer desregistro
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # si el numero de centroides de entrada es mayor que el numero de centroides actuales
	    	# registrar los nuevos centroides
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        # retornar bbox
        return self.bbox




