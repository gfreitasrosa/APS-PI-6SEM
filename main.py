import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

def abrir_seletor_arquivo():
    filepath = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Arquivos de imagem", (".png", ".jpg", ".jpeg", ".gif"))]
    )
    if filepath:
        exibir_imagem(filepath)
        converter_para_matriz(filepath)

def exibir_imagem(filepath):
    imagem = Image.open(filepath)
    imagem = imagem.resize((300, 300), Image.ANTIALIAS)
    imagem = ImageTk.PhotoImage(imagem)

    label_imagem.config(image=imagem)
    label_imagem.imagem = imagem

def converter_para_matriz(filepath):
    imagem = Image.open(filepath)
    matriz = np.array(imagem)
    print("Matriz NumPy:")
    print(matriz)

#def realiza_teste():
    
# Criar janela principal
janela = tk.Tk()
janela.title("Escolher Imagem")

# Definir o tamanho padrão da janela
largura = 200
altura = 200
janela.geometry(f"{largura}x{altura}")

# Botão para abrir o seletor de arquivos
btn_selecionar = tk.Button(janela, text="Selecionar Imagem", command=abrir_seletor_arquivo)
btn_teste = tk.Button(janela, text="Executar teste", command="")
btn_teste.pack(pady=10)
btn_selecionar.pack(pady=10)

# Rótulo para exibir a imagem selecionada
label_imagem = tk.Label(janela)
label_imagem.pack()

# Iniciar o loop principal da janela
janela.mainloop()