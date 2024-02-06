import cv2
import numpy as np
import gtts
from playsound import playsound

class DetectorDeObjetos:
    
    def __init__(self, cfg_path, weights_path, classes_file):
        self.caminho_cfg = cfg_path
        self.caminho_weights = weights_path
        self.classes_file = classes_file

    @property
    def caminho_cfg(self):
        return str(self._caminho_cfg)
    
    @caminho_cfg.setter
    def caminho_cfg(self, value):
        self._caminho_cfg = value

    @property
    def caminho_weights(self):
        return str(self._caminho_weights)
    
    @caminho_weights.setter
    def caminho_weights(self, value):
        self._caminho_weights = value

    @property
    def classes_file(self):
        return str(self._classes_file)
    
    @classes_file.setter
    def classes_file(self, value):
        self._classes_file = value

    @staticmethod
    def carregar_classes(classes_file):
        with open(classes_file, 'r') as f:
            class_names = f.read().splitlines()
        return class_names

    def iniciar_modelo(self):
        net = cv2.dnn.readNet(str(self._caminho_weights), str(self._caminho_cfg))  # Corrigindo o acesso aos caminhos
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(608, 608), scale=1 / 255)
        return model

    def detectar_objeto(image_path, object_class):
        class_names = DetectorDeObjetos.load_classes(DetectorDeObjetos.classes_file)
        model = DetectorDeObjetos.iniciar_modelo()

        image = cv2.imread(image_path)
        conf_threshold = 0.06
        nms_threshold = 0.1
        classes, scores, boxes = model.detect(image, conf_threshold, nms_threshold)
        objects = []

        for (classid, score, box) in zip(classes, scores, boxes):
            if class_names[int(classid)] == object_class:
                objects.append({
                    'class': class_names[int(classid)],
                    'score': float(score),
                    'box': box.tolist()
                })

        if objects:
            object_info = objects[0]  # Consider only the first object found
            posicao = DetectorDeObjetos.obter_posicao(image, object_info['box'])
            DetectorDeObjetos.gerar_audio(object_class, posicao)

        return objects

    def obter_posicao(image, box):
        altura, largura, _ = image.shape
        centro_x = largura // 2
        centro_y = altura // 2

        x, y, w, h = box
        posicao = ""

        centro_objetox = x + w // 2
        centro_objetoy = y + h // 2

        if centro_objetox < centro_x and centro_objetoy > centro_y:
            posicao = "Inferior esquerdo"
        elif centro_objetox < centro_x and centro_objetoy < centro_y:
            posicao = "Superior esquerdo"
        elif centro_objetox > centro_x and centro_objetoy > centro_y:
            posicao = "Inferior direito"
        elif centro_objetox > centro_x and centro_objetoy < centro_y:
            posicao = "Superior direito"
        else:
            posicao = "Centro"

        return posicao

    def gerar_audio(object_class, position):
        phrase = f"Objeto {object_class} localizado no canto {position}"
        audio = gtts.gTTS(phrase, lang="pt-br")
        audio.save("saidaTTS.mp3")
        #playsound("saidaTTS.mp3")
