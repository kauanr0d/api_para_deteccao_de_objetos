
Para rodar o projeto em sua máquina siga os seguintes passos:

SEM AMBIENTE VIRTUAL

1 - Baixe o arquivo yolov4.weights disponível em: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

2 - Copie ou insira o arquivo yolov4.weights dentro do diretório deste projeto

3 - Abra seu terminal dentro da pasta do projeto e execute o segundo comando:

    pip install -r requirements.txt
    
    Caso haja algum problema com a instalação do módulo playsound, utilize o seguinte comando:
    
    pip install --upgrade setuptools wheel
    
4 - Com todos os módulos instalados, basta abrir seu terminal dentro do diretório do projeto e executar o seguinte comando:

    python3 app.py
    
    ou
    
    python app.py
    
COM AMBIENTE VIRTUAL

Siga o passo 1 e 2

3 - Abra seu terminal dentro da pasta do projeto e crie seu ambiente virtual com o seguinte comando:

    python3 -m env nomedoseuenv

4 - Ative seu ambiente virtual:

    Se estiver usando windowns: .\nomedoseuenv\Scripts\activate
    
    Se estiver usando linux ou mac: source nomedoseuenv/bin/activate
    
5 - Com seu ambiente virtual ativado, use o comando:

    pip install -r requirements.txt  e, em seguida, execute o arquivo app.py com python3 app.py
  
