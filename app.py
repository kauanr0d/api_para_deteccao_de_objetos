from flask import Flask, request, render_template, send_from_directory
from detector import *
import cv2

app = Flask(__name__)

class_name = carregar_classes("coco.names")

caminho_cfg = "yolov4.cfg"
caminho_weights = "yolov4.weights"

modelo = iniciar_modelo(caminho_cfg,caminho_weights)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    class_names = carregar_classes("coco.names")

    file = request.files['file']
    file.save("static/imagem.jpg")

    imagem = abrir_imagem("static/imagem.jpg")
    objetos, imagem_detectada = detectar(model=modelo, imagem=imagem, classe_objeto='book', class_names=class_names)
   
    cv2.imwrite("static/imagem_detectada.jpg", imagem_detectada)
    saida_tts(objetos)
    
    return render_template('resultado.html', objetos=objetos, imagem='imagem_detectada.jpg')

if __name__ == '__main__':
    app.run(debug=True)