from flask import Flask, request, render_template, send_from_directory, jsonify, abort
from werkzeug.utils import secure_filename
from detector import *
import cv2
import json
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1000 * 1000
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', 'jpeg']


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
    filename = secure_filename(file.filename)

    if filename != '':
        file_extension = os.path.splitext(filename)[1]
        if file_extension not in app.config['UPLOAD_EXTENSIONS']:
            abort


    file.save("static/imagem.jpg")
  
    objeto = request.form['objeto']  # recebe a descrição do input objeto
    
    if objeto == None or objeto not in class_names:
        raise ValueError("Descrição do objeto inválida!")
    

    imagem = abrir_imagem("static/imagem.jpg")
    objetos, imagem_detectada, box, score, posicao = detectar(model=modelo, imagem=imagem, classe_objeto=objeto, class_names=class_names)
     
    box_dict = {'x': float(box[0]), 'y': float(box[1]), 'width': float(box[2]), 'height': float(box[3])} 
 
    box_json = json.dumps(box_dict,indent=2)

    cv2.imwrite("static/imagem_detectada.jpg", imagem_detectada)
    response = {
        'object_class' :  list(objetos.keys())[0],
        'box' : json.loads(box_json),
        'posicao': posicao,
        'taxa de confianca' : round(float(score[0]),2),
    }
    #render_template('resultado.html', objetos=objetos, imagem='imagem_detectada.jpg')#, retorne isto
    return jsonify(response),render_template('resultado.html', objetos=objetos, imagem='imagem_detectada.jpg')

#rota para gerar saída TTS
@app.route('/play_audio')
def play_audio():
    return playsound("saídaTTS.mp3")
    

if __name__ == '__main__':
    app.run(debug=True)
     