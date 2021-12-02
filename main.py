from tkinter import *
from PIL import Image, ImageTk

from deteccionseguimiento import *

from collections import defaultdict
import cv2
import imutils
import datetime
import numpy as np
import pandas as pd

import jetson.inference
import jetson.utils
import threading

from flask import Flask, render_template, Response, request
app = Flask(__name__)


def iniciar():
    # TRAER VARIABLES GLOBALES
    global cap
    global frame
    global personasactuales
    global lblactualnum
    global lblporcentajenum
    global Iniciar
    global lblwarning

    # Inicializacion del conteo de personas
    personasactuales = 0

    # Creacion de la ventana de iniciar conteo con textos y botones
    Inicio.destroy()
    Iniciar = Tk()
    Iniciar.geometry("1600x900")
    Iniciar.configure(bg="#bfcde6")

    # Mostrar el texto del número maximo de personas
    lblmax = Label(Iniciar, text="Este es el numero maximo de personas que pueden estar en el área seleccionada", font=("Bookman Old Style", 20),wraplength=400,bg="#bfcde6")
    lblmax.place(x=500, y=500)

    #Mostrar el número maximo de personas
    lblmaxnum = Label(Iniciar, text=maximo, font=("Bookman Old Style", 200),bg="#bfcde6")
    lblmaxnum.place(x=560, y=100)

    # Mostrar el texto del porcentaje de personas
    lblporcentaje = Label(Iniciar, text="Este es el porcentaje de aforo en el área seleccionada", font=("Bookman Old Style", 20),wraplength=400,bg="#bfcde6")
    lblporcentaje.place(x=1030, y=500)

    # Mostrar el número del porcentaje de personas
    lblporcentajenum = Label(Iniciar, text=porcentaje, font=("Bookman Old Style", 200),bg="#bfcde6")
    lblporcentajenum.place(x=1100, y=100)

    # Mostrar el texto del número actual de personas
    lblactual = Label(Iniciar, text="Actualmente en el área seleccionada hay este número de personas", font=("Bookman Old Style", 20),wraplength=400,bg="#bfcde6")
    lblactual.place(x=0, y=500)

    # Mostrar el número actual de personas
    lblactualnum = Label(Iniciar, text=personasactuales, font=("Bookman Old Style", 200),bg="#bfcde6")
    lblactualnum.place(x=75, y=100)

    # Mostrar el caracter "%" en la interfaz
    lblperc=Label(Iniciar, text="%", font=("Bookman Old Style", 15),bg="#bfcde6")
    lblperc.place(x=1400, y=320)

    # Creacion de la alerta cuando se supera el 70% de aforo máximo
    lblwarning = Label(Iniciar, text="",fg = 'red',font=("Bookman Old Style", 18), bg="#bfcde6")
    lblwarning.place(x=400,y=700)

    # Llamado a la funcion para actualizar las variables en pantalla de la interfaz
    actualizardatos()
    

def actualizardatos():

    #Actuializacion del número actual de personas
    lblactualnum.configure(text=personasactuales)

    #Llamado a la funcion que se encarga del procesamiento de la imagen y toma de datos
    lblactualnum.after(1,callback)
    

def callback():
	# TRAER VARIABLES GLOBALES
    global listacentros
    global centrox
    global centroy
    global heatmap
    global lpc_count
    global rects 
    global mapa
    global maximo
    global info
    global grayinfo
    global flaskini
    global frame
    global datos
    global data
    global conteo
    global count
    global num_personas
    
    fontScale = 1
    colortext = (0,0,0)
    thick = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
	# LEER FRAME
    ret, frame = cap.read()
    info = np.ones(shape=(720,1280,3), dtype=np.uint8)*255

    if ret:

# ---------------------------------------------- DEFINICION DEL ESPACIO -------------------------------------
        color = (0, 255, 255)
        color2 = (0, 0, 255)
        x_ini = [matrizpuntos[0,0],matrizpuntos[0,1]]
        y_ini = [matrizpuntos[1,0],matrizpuntos[1,1]]
        x_fin = [matrizpuntos[2,0],matrizpuntos[2,1]]
        y_fin = [matrizpuntos[3,0],matrizpuntos[3,1]]

        sectorizacion = np.array([x_ini, y_ini, x_fin, y_fin])

# ------------------------------------------------- DETECTOR-----------------------------------------------------
	
		#INICIAR EL DETECTOR
        mobilenetssd = detectormobile.iniciar(frame,mobilenet)

        # Obtener los recuadros de las personas detectadas
        rects, centrox, centroy = detectormobile.detectarpersonas(mobilenetssd,mobilenet,x_fin,x_ini,y_ini,y_fin,listacentros)
        
        # juntar los recuadros en una lista
        listacentros.append((centrox,centroy))
        box = np.array(rects)
        box = box.astype(int)
          
#----------------------------------------------- HEATMAP--------------------------------------------------------------
        # si centro en x y centro en y estan en la lista 
        if (centrox,centroy) in listacentros:

			# obtener cuantas veces esta centrox y centroy en listacentros 
            cont_map = listacentros.count((centrox,centroy))
            
            # segun cuantas veces este el detector se le cambia el color al punto en el mapa de calor 
            variacion = 20
            if (cont_map*variacion)>=250 :
                cont_map = 240
            cv2.circle(heat_map, (int(centrox), int(centroy)), 1, (20+(cont_map*variacion), 20+(cont_map*variacion), 20+(cont_map*variacion)), -1)
            
        
# -------------------------------------------------- TRACKER -----------------------------------------------------------
        # aplicar el filtro non_max_supression
        rects = non_max_suppression_fast.input(box,thresh =0.27)
        objeto = tracker.update(rects)
        # Centroid Tracker
        # Obtener nueva posicion (manteniendo el ID)
        for (objectId, bbox) in objeto.items():

            x1, y1, x2, y2 = bbox
            # Convertir a enteros
            x1 = int(x1)
            y1 = int(y1)
            y2 = int(y2)
            x2 = int(x2)

            # calculando el piso de la persona
            centro_x = int((x1 + x2) / 2)
            centro_y = int(y2)  #para terrizar al piso solo y2
                
            # introducir todos los centros en un diccionario para poder realizar lineas
            dic_centro[objectId].append((centro_x, centro_y))

            # dibujando la linea
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            lpc_count = len(objeto)

	

		# Del conteo actual muestree 10 y obtenga el mayor 
        count.append(lpc_count)
        conteo = conteo +1

        if conteo == 10:
            num_personas = np.max(count)
            conteo = 0
            count = []
	
		#obtener una imagen 
        actuales = "Personas actuales: {}".format(num_personas)
        info = cv2.putText(info,actuales,(450,100),font,fontScale,colortext ,thick,cv2.LINE_AA)

# -------------------------------------------------- MOSTRAR INFORMACION ------------------------------------------------------



        # Dibujar sector
        cv2.drawContours(frame, [sectorizacion], -1, color, 2)
        cv2.drawContours(heat_map, [sectorizacion], -1, color, 2)

        # MOSTRAR FRAME
        cv2.imshow('camara', frame)
        

        
#--------------------------------------------- TRANSFORMACION MAPA DE CALOR ----------------------------------------------------
        # Transformar a escala de grises
        graymap = cv2.cvtColor(heat_map,cv2.COLOR_BGR2GRAY)
		
		#Aplicar ruido gausiano
        graymap = cv2.GaussianBlur(graymap,(11,11),2,2)
		
		#Transformar a espacio de color JET
        mapa = cv2.applyColorMap(graymap,cv2.COLORMAP_JET)

        #Llamado a la funcion que toma el dato de las personas actuales
        lblactualnum.after(1,conteopersonas)
        

#-------------------------------------------------INFORMACION DE AFORO----------------------------------------------------------
        maximop = "Maximo numero de personas: {}".format(maximo)
        info = cv2.putText(info,maximop,(450,200),font,fontScale,colortext ,thick,cv2.LINE_AA)
        porcentajepersonas = int((num_personas/maximo)*100)
        porc= "Porcentaje actual de personas: %{}".format(porcentajepersonas)

        # Si el porcentaje es mayor a 70% mostrar alerta
        if porcentajepersonas > 70:
            alerta= "!ALERTA SE ESTA ACERCANDO AL AFORO MAXIMO!"
            info = cv2.putText(info,alerta,(450,300),font,fontScale,(0,0,255) ,thick,cv2.LINE_AA)

        info = cv2.putText(info,porc,(450,300),font,fontScale,colortext ,thick,cv2.LINE_AA) 
        
        grayinfo = info

        # si se presiona q (parar deteccion)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cam.release()
        
        
    
#---------------------------------------ENVIO FLASK ------------------------------------- 
def obtener_mapa():
    global mapa
    while True:
        imgmapa = imutils.resize(mapa, width=500)
        _, bufer = cv2.imencode(".jpg", imgmapa)
        imagen = bufer.tobytes()
        
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + imagen + b"\r\n"



def obtener_info():
    global grayinfo
    while True:
        imginfo = imutils.resize(grayinfo, width=500)
        _, bufer = cv2.imencode(".jpg", imginfo)
        informacion = bufer.tobytes()
        
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + informacion + b"\r\n"



def run_flask():
    app.run(host="0.0.0.0")


   
@app.route("/streaming_camara")
def streaming_camara():
    return Response(obtener_mapa(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/streaming_info")
def streaming_info():

    return Response(obtener_info(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/aforo',methods=['POST'])
def aforo():
    global maximo
    maximo = request.form['aforo']
    maximo = int(maximo)

#------------------------------CONTEO Y CALCULO DE PORCENTAJE EN LA INTERFAZ------------------------


def conteopersonas():
    global personasactuales
    global porcentaje
    
    personasactuales = num_personas

    #Calcula el porcentaje y configura el texto con dicho porcentaje
    porcentaje = int((personasactuales / maximo) * 100)
    lblporcentajenum.configure(text=porcentaje)
    if(porcentaje>70):
        lblwarning.configure(text="!CUIDADO, SE ESTA ALCANZANDO EL AFORO MAXIMO PARA ESTA AREA!",fg = 'red')
    else:
        lblwarning.configure(text="")
    lblporcentajenum.after(1)

    #Se cumple el bucle, el programa envia datos nuevamente y los actualiza al llamar esta funcion de nuevo
    actualizardatos()



#----------------------------------CREACION DE LA VENTANA DE AFORO MAXIMO---------------------------------#

def Ventanapersonas():
    global ventanaaforo
    global aforomax
    ventanaaforo = Tk()
    ventanaaforo.title("Aforo máximo")
    ventanaaforo.geometry("600x400")
    textoaforo = Label(ventanaaforo, text="Escriba el máximo de personas que pueden estar en el área seleccionada", font=("Bookman Old Style",15), anchor = "center", wraplength = 400)

    #Toma del valor que ingresa el usuario
    aforo = StringVar()
    aforomax = Entry(ventanaaforo, textvariable=aforo, justify=CENTER, font=("Bookman Old Style",15))
    aforomax.place(x=175, y=160)
    lblaforomaximo = Label(ventanaaforo, text="personas", font=("Bookman Old Style",15), anchor = "center", wraplength = 400)
    lblaforomaximo.place(x=250, y=200)

    #Boton de finalizar para guardar la variable
    btnOkaforo = Button(ventanaaforo, text="Finalizar", width=20, command=retrocederaforo, borderwidth=3, relief="solid",  height=2, font=("Bookman Old Style",15), bg="#dfe6f2")
    btnOkaforo.place(x=160, y=300)
    textoaforo.pack()
    

#-------------------------------------CREACION DE LA VENTANA DE SECTORIZACION------------------------------------------------------#
def Ventanasectorizacion():

    #TRAER VARIABLES GLOBALES
    global lblimagen
    global ventanasectorizacion
    global imagen
    global canvas
    global matrizpuntos
    global cont_puntos

    #Creacion de la ventana de sectorizacion
    ventanasectorizacion = Toplevel()
    ventanasectorizacion.geometry("1600x900")
    cont_puntos = 0
    matrizpuntos = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    ret, imagen = cap.read()
    if ret == True:

        #Se toma la foto desde la camara y se muestra al usuario
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imagen)
        img = ImageTk.PhotoImage(image=im)


        canvas = Canvas(ventanasectorizacion, width=1280, height=720)
        canvas.create_image(0, 0, anchor="nw", image=img)
        canvas.place(x=80,y=50)

        #Llamado de la funcion para ver las acciones del raton en la imagen
        canvas.bind("<Button-1>", presion_mouse)



    #Textos y botones para la ventana de sectorizacion
    textosectorizacion = Label(ventanasectorizacion, text="Indique el área que quiere sectorizar dentro de la imagen colocando los 4 puntos que enmarcan dicha área en sentido horario y a continuación haga clic en Finalizar", font=("Bookman Old Style",15), wraplength= 1200)
    btnOksectorizacion = Button(ventanasectorizacion, text="Finalizar", width=45, command=retrocedersectorizacion, height=2, borderwidth=3, relief="solid", bg="#dfe6f2", font=("Bookman Old Style",15))
    btnOksectorizacion.place(x=400, y=800)

    textosectorizacion.pack()
    
    

    # El programa retorna a la ventana principal
    ventanasectorizacion.mainloop()


#----------------------------------------------------DETECCION DE EVENTOS DEL MOUSE----------------------------------#
def presion_mouse(evento):
    global cont_puntos
    canvas.create_oval(evento.x - 5, evento.y - 5, evento.x + 5, evento.y + 5, fill="red")
    matrizpuntos[cont_puntos,:] = [evento.x , evento.y]
    cont_puntos = cont_puntos + 1


#----------------------------------------------------LOGICA DE CIERRE DE VENTANA DE AFORO Y TOMA DE DATO---------------------------------
def retrocederaforo():
    global maximo
    global porcentaje
    global personasactuales
    maximo = int(aforomax.get())
    personasactuales = 0
    porcentaje = 0
    #print(porcentaje)
    ventanaaforo.destroy()


#----------------------------------------------------CIERRE DE VENTANA SECTORIZACION---------------------------------
def retrocedersectorizacion():
    ventanasectorizacion.destroy()
   

#Creacion de la ventana inicial de la interfaz
Inicio = Tk()
Inicio.title("Control de aforo en espacios publicos")
Inicio.geometry("1600x900")
bg = ImageTk.PhotoImage(Image.open("/home/erickb/Desktop/Tesis-control-de-aforo-en-espacios-publicos-main/fondo.jpg"))
cap = cv2.VideoCapture(0)
canvas1 = Canvas(Inicio, width=400, height=400)
canvas1.pack(fill="both", expand=True)
canvas1.create_image(0, 0, image=bg, anchor="nw")
TextoInicio = Label(Inicio, text="Control de aforo", font=("Bookman Old Style",60), anchor = "center",bg="#bfcde6")
TextoInicio.place(x=450, y=20)
btnIniciar = Button(Inicio, text="Iniciar conteo", borderwidth=3, relief="solid", width=30,  height=2, command=iniciar, bg="#dfe6f2", font=("Bookman Old Style",15))
btnIniciar.place(x=60, y=600)
btnpersonas = Button(Inicio, text="Aforo máximo", width=30, height=2, borderwidth=3, relief="solid", command=Ventanapersonas, bg="#dfe6f2", font=("Bookman Old Style",15))
btnpersonas.place(x=510, y=600)                   
btnsectorizacion = Button(Inicio, text="Selección de área", borderwidth=3, relief="solid", width=30,  height=2, command=Ventanasectorizacion, bg="#dfe6f2", font=("Bookman Old Style",15))
btnsectorizacion.place(x=950, y=600)


#------------------------------------------------------------ INICILIZAR TRACKER Y DETECTOR --------------------------------------------------------------------------
mobilenet = jetson.inference.detectNet("ssd-mobilenet-v2")
tracker = CentroidTracker(maxDisappeared=10, maxDistance=100)


#Configurar la resolucion
cap.set(3,720)
cap.set(4,480)

# iniciar variables, deteccion y seguimiento
global tframes
global disc_centro
global fps
global listacentros
global centrox
global centroy
global heatmap
global lpc_count
global cont_ini
global info
global flaskini
global datos
global data
global conteo
global count
global num_personas

count = []
data= []
datos =0
conteo = 0
fps = 0
tframes = 0
lpc_count = 0
num_personas =0 

# Variables para Tracking
dic_centro = defaultdict(list)          
lista_obid = []
listacentros = []
centrox = 2500
centroy = 2500

# Variables para mapa de calor
heat_map = np.zeros(shape=(480,720,3), dtype=np.uint8)


#Inicializar Flask
hilo1 = threading.Thread(target=run_flask)
hilo1.start()
Inicio.mainloop()

