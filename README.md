# Atividade Prática Supervisionada - 3º Semestre CC

![Badge em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge)
![Badge Linguagem](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

## 🐍 Escopo

O grupo deverá fazer um estudo de técnicas de reconhecimento de imagens
contendo letras do alfabeto escritas à mão. Para este trabalho será avaliado
somente o reconhecimento da letra “i” maiúsculo e minúsculo, entre outras vogais do
alfabeto. O objetivo do trabalho é fazer um programa que responda se uma imagem
de uma letra vogal escrita à mão é a letra “i” ou não.

Será disponibilizado um conjunto de dados retirado da página
https://www.nist.gov/srd/nist-special-database-19. O grupo deve usar apenas as
letras vogais maiúsculo e minúsculo. É aconselhável que o grupo use apenas uma
parte do conjunto de dados, visto que o conjunto todo é muito grande e pode fazer o
treinamento demorar muito.

O grupo deverá aplicar técnicas de Aprendizado de Máquina para resolver o
problema. De preferência deverão ser aplicadas mais de uma técnica diferentes para
resolver o problema, que deverão ser avaliadas segundo os critérios de acurácia,
precisão, recall, e pontuação f1.
As técnicas e avaliação de cada técnica deverão ser apresentadas no trabalho.

## 📁 Acesso ao projeto

Você pode [acessar o código fonte do projeto](https://github.com/gfreitasrosa/APS-PI-6SEM/tree/main), [baixá-lo](https://github.com/gfreitasrosa/APS-PI-6SEM/archive/refs/heads/main.zip) ou clonar o repositório.

## ❓ Como rodar o programa?

### Passo 1:
  -  Clonar o repositório ou baixa-lo
### Passo 2:
  -  Criar e ativar um ambiente virtual python 3.9.5:
 
  ```bash
  1 - python -m venv nome_do_seu_ambiente
  
  2 - cd nome_do_seu_ambiente\Scripts\activate
  ```
### Passo 3
  - Já com o ambiente virtual ativado, instalar as libs usando:

  ```bash
  pip install -r requirements.txt
  ```
### Passo 4
  - Após isso já será possível rodar o programa, executando o arquivo python main.py

>[!NOTE]
   >
   >É importando que as hierarquias de repositórios estejam corretas e o caminho para o pathlib no programa classificadores.py esteja apontando para o dataset correto.

### *Desenvolvedores*:
