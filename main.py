#from classificadores.classificadores import *
from classificadores.classificador_mlp import *
from classificadores.classificador_sgd import *
import tkinter as tk
from tkinter import filedialog

def executar_mlp():
    filepath = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Arquivos de imagem", (".png", ".jpg", ".jpeg", ".gif"))]
    )
    if filepath:
        classificar_imagem_mlp(mlp_clf, filepath)


def executar_sgd():
    filepath = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Arquivos de imagem", (".png", ".jpg", ".jpeg", ".gif"))]
    )
    if filepath:
        classificar_imagem(sgd_clf, filepath)

    
# Criar janela principal
janela = tk.Tk()
janela.title("Escolher Imagem")

# Definir o tamanho padrão da janela
largura = 200
altura = 200
janela.geometry(f"{largura}x{altura}")

# Botão para abrir o seletor de arquivos
btn_executar_sgd = tk.Button(janela, text="Executar teste Classificador SGD", command=executar_sgd)
btn_exceutar_mlp = tk.Button(janela, text="Executar teste Classificador MLP", command=executar_mlp)
btn_executar_sgd.pack(pady=10)
btn_exceutar_mlp.pack(pady=10)

# Rótulo para exibir a imagem selecionada
label_imagem = tk.Label(janela)
label_imagem.pack()

# Iniciar o loop principal da janela
janela.mainloop()