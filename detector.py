import cv2
import numpy as np
from playsound import playsound
import gtts


COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 255), (255, 0, 0)]

  
def carregar_classes(nome_arquivo):
    with open(nome_arquivo, 'r') as arquivo:
        class_names = [cname.strip() for cname in arquivo.readlines()]
    return class_names


def iniciar_modelo( arquivo_weights,  arquivo_cfg):
    net = cv2.dnn.readNet(arquivo_weights,arquivo_cfg)
    model = cv2.dnn.DetectionModel(net)
    model.setInputParams(size=(608,608),scale = 1/255) #tamanho definido na linha 8 e 9 do arquivo .cfg
    return model
 
def abrir_imagem(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)
    ##imagem_rgb = cv2.cvtColor(imagem,cv2.COLOR_BGR2RGB)
    cv2.imwrite("saida.jpg", cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR))
    return imagem

def detectar(model, imagem, classe_objeto, class_names):
    conf_threshold = 0.06
    nms_threshold = 0.1
    classes, scores, boxes = model.detect(imagem, conf_threshold, nms_threshold)
    tabela_objetos = {}
    box_loc = []

    for(classid, score, box) in zip(classes, scores, boxes):
        if class_names[int(classid)] == classe_objeto:
            box_loc = box
            color = COLORS[int(classid) % len(COLORS)]
            label = f"{class_names[int(classid)]} : {score}"
            class_name = class_names[int(classid)]
            tabela_objetos[class_name] = class_name
            posicao = localizacao(imagem, box)
            cv2.rectangle(imagem, box, color, 2)
            cv2.putText(imagem, label + " " + posicao, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
    saida_tts(tabela_objetos, posicao)
    return tabela_objetos, imagem, box_loc,scores,posicao

def saida_tts(tabela_objetos, posicao):
    objeto = list(tabela_objetos.keys())[0]
    frase = f"Objeto {objeto} localizado no canto {posicao}"
    audio = gtts.gTTS(frase, lang="pt-br")
    audio.save("saídaTTS.mp3")
    #playsound("frase.mp3")
 

def localizacao(imagem, box) -> str:
    altura, largura, _ = imagem.shape
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


