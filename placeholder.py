#Un placeholder en tensorflow, es una variable a la cual se le asignan datos para un uso en una fecha posterior.
#Permite crear operaciones y gráficos de computo, sin la necesidad de los datos.
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "/home/rey/Imágenes/original_shack.jpg"    #imagen a extraer los datos altura: 413 y anchura: 622
raw_image_data = mpimg.imread(filename)            #

image = tf.placeholder("uint8",[None,None,3])
#print(raw_image_data)
slice = tf.slice(image, [0,0,0],[300,400,-1])   

#función slice: image= imagen de entrada, begin=tiene que ser menor que la altura por ejemplo la altura máxima es 413 
#entonces begin solo puede iniciar con un valor menor a 413 tambien dependiendo del tamaño a cortar, ya que el corte no 
#puede ser mayor a la imagen. size = el tamaño resultante del corte, especificado por el usuario.

with tf.Session()as session:
    result = session.run(slice, feed_dict={image: raw_image_data })
    print(result.shape);
    
plt.imshow(result)
plt.show()
