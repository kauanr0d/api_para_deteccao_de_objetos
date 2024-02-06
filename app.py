from flask import Flask, request, render_template, send_from_directory, jsonify, abort, make_response
from werkzeug.utils import secure_filename
from detector import *
from classes.DetectorDeObjetos import DetectorDeObjetos
import cv2
import json
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1000 * 1000
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', 'jpeg']
 
detector = DetectorDeObjetos("yolov4.cfg","yolov4.weights","coco.names")
class_name = detector.carregar_classes("coco.names")
modelo = detector.iniciar_modelo()
 
#caminho_cfg = "yolov4.cfg"
#caminho_weights = "yolov4.weights"

#modelo = iniciar_modelo(caminho_cfg, caminho_weights)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/detect', methods=['POST'])
def upload():

    file = request.files['file']
    filename = secure_filename(file.filename)

    if filename != '':
        file_extension = os.path.splitext(filename)[1]
        if file_extension not in app.config['UPLOAD_EXTENSIONS']:
            abort(415)

    file.save("static/imagem.jpg")

    objeto = request.form['objeto']  # recebe a descrição do input objeto

    if objeto not in class_name:
        raise ValueError("Descrição do objeto inválida!")

    imagem = abrir_imagem("static/imagem.jpg")
    objetos, imagem_detectada, box, score, posicao = detectar( model=modelo,
    imagem=imagem, classe_objeto=objeto, class_names=class_name)



    box_dict = {'x': float(box[0]), 'y': float(
        box[1]), 'width': float(box[2]), 'height': float(box[3])}

    box_json = json.dumps(box_dict, indent=2)

    cv2.imwrite("static/imagem_detectada.jpg", imagem_detectada)
    response = {
        'object_class':  list(objetos.keys())[0],
        'box': json.loads(box_json),
        'posicao': posicao,
        'taxa de confianca': round(float(score[0]), 2),
    }
    # render_template('resultado.html', objetos=objetos, imagem='imagem_detectada.jpg')#, retorne isto
    return make_response(jsonify(response)), renderizar(objetos)


def renderizar(objetos):
    return render_template('resultado.html', objetos=objetos, imagem='imagem_detectada.jpg')
    

# rota para gerar saída TTS
@app.route('/play_audio')
def play_audio():
    return playsound("saídaTTS.mp3")


if __name__ == '__main__':
    app.run(debug=True)
