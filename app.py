from flask import Flask, request, render_template, send_from_directory, jsonify, abort, make_response, send_file
from werkzeug.utils import secure_filename
from detector import *
from classes.DetectorDeObjetos import DetectorDeObjetos
import cv2
import json
import os
import threading


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1000 * 1000
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
 
detector = DetectorDeObjetos("yolov4.cfg","yolov4.weights","coco.names")
class_name = detector.carregar_classes("coco.names")
modelo = detector.iniciar_modelo()
 
#caminho_cfg = "yolov4.cfg"
#caminho_weights = "yolov4.weights"

#modelo = iniciar_modelo(caminho_cfg, caminho_weights)
detector_tiny = DetectorDeObjetos("yolov4-tiny.cfg","yolov4-tiny.weights","coco.names")
modelo_tiny = detector_tiny.iniciar_modelo()

@app.route('/')
def index():
    return render_template("index.html")


 
#curl -X POST -F "file=@<caminho_da_imagem>" -F "objeto=<nome_do_objeto>" http://127.0.0.1:5000/detect

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
        make_response(jsonify({'error': 'Descrição do objeto inválida!'}), 400)

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
    return make_response(jsonify(response))#, renderizar(objetos)

  
@app.route('/display', methods= ['GET'] )
def renderizar():
    return send_file("static/imagem_detectada.jpg")


@app.route('/video_capture', methods=['POST'])
def upload_video():
    url_para_rtsp = request.form['endereco-rtsp']
    nome_objeto = request.form['objeto']

    captura = cv2.VideoCapture(url_para_rtsp)

    if not captura.isOpened():
        print('Erro ao abrir transmissão')
        return

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    box_loc = []

    while True: 
        ret,frame = captura.read()

        conf_threshold = 0.04
        nms_threshold = 0.1

        classes, scores, boxes = modelo_tiny.detect(frame,0.01,0.02)
        for class_id, score, box in zip(classes, scores, boxes):
            if class_name[int(class_id)] == nome_objeto:
                color = COLORS[int(class_id) % len(COLORS)]
                label = f"{class_name[int(class_id)]} : {score}"
                class_name1 = class_name[int(class_id)]
                posicao = localizacao(frame, box)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, f"{class_name1} : {score} {posicao}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        cv2.imshow('Video', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    captura.release()
    cv2.destroyAllWindows()

    return 'Captura de vídeo encerrada'



# rota para gerar saída TTS
@app.route('/play_audio')
def play_audio():
    return playsound("saídaTTS.mp3")


if __name__ == '__main__':
    app.run(debug=True)
 