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

def detectar(model, imagem,classe_objeto,class_names):
    conf_threshold = 0.06
    nms_thrreshold = 0.1
    classes, scores, boxes = model.detect(imagem,conf_threshold,nms_thrreshold)
    lista_objetos = []
    tabela_objetos = {}

    for(classid, score, box) in zip(classes,scores,boxes):
        if class_names[int(classid)] == classe_objeto:
            color = COLORS[int (classid)%len(COLORS)]
            label = f"{class_names[int(classid)]} : {score}"
            class_name = class_names[int(classid)]    
            tabela_objetos[class_name] = class_name
            cv2.rectangle(imagem,box,color,2)
            cv2.putText(imagem, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
    return tabela_objetos, imagem

def saida_tts(tabela_objetos):
    frase = "Objetos detectados"
    fala = gtts.gTTS(frase, lang = "pt-br")
    fala.save("frase.mp3")
    playsound("frase.mp3")

    for objetos in tabela_objetos:
        fala = gtts.gTTS(objetos, lang ='pt-br')
        fala.save("objetos.mp3")
        playsound("objetos.mp3")


#def main():
 #   class_names = carregar_classes("coco.names")

 #   descricao_objeto = "book"
    
  #  modelo = iniciar_modelo("yolov4.weights","yolov4.cfg")

   # imagem = abrir_imagem("/home/kauan/Downloads/mesinha2.jpeg")

    #objetos = {}
    #objetos, imagem = detectar(model= modelo,imagem= imagem,classe_objeto= descricao_objeto,class_names=class_names)

#    saida_tts(objetos)

 #   cv2.imshow("deteccoes",imagem)
  #  cv2.waitKey(0)
   # cv2.destroyAllWindows()

#if __name__ == "__main__":
 #   main()
    

