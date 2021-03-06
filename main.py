# bibliotecas utilizadas
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
import imutils

OUTPUT_WIDTH = 500


def main():
    st.sidebar.header("Segmentador de Placas Automotivas por Convolução")
    st.sidebar.info("100% em Python")

    # menu com oções de páginas
    opcoes_menu = ["Segmentador de Placas", "Sobre o projeto"]
    escolha = st.sidebar.selectbox("Escolha uma opção", opcoes_menu)

    our_image = Image.open("000.jpg")

    if escolha == "Segmentador de Placas":

        st.title("Segmentador de Placas Automotivas por Convolução")
        st.text("Vladimir Simões da Luz Junior")
        st.markdown(
            "Projeto da Masterclass Introdução à Visão Computacional, do Carlos Melo (Sigmoidal). Para saber mais sobre a Masterclass, acesse [esta página](https://pay.hotmart.com/K44730436X?checkoutMode=10&bid=1608039415553).")

        st.markdown(
            " Web App para segmentar, através da convolução de diversos tipos de kernel, a região da placa em um carro")
        st.markdown(
            " Ressalto que este projeto tem a finalidade de mostrar o que é possível realizar, utilizando apenas python e a biblioteca Open-CV para o processamento de imagens. Além de realizar a prova de conceito(POC) do projeto em questão")
        st.markdown(
            " Utilizando Deep Learning, os resultados com certeza seriam melhores, logo é possível que em algumas fotos este pré-processamento de imagens não seja preciso...")
        st.markdown(
            " Como premissa, considero que as imagens dos Automovéis são de boa resolução e com uma distância, relativa ao carro, curta.")

        # carregar e exibir imagem
        # our_image = cv2.imread(file_name)  ---> Não vai dar certo
        st.subheader("Carregar arquivo de imagem para segmentar a placa automotiva")
        image_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.sidebar.text("Imagem Original")
            st.sidebar.image(our_image, width=150)
        else:
            st.sidebar.text("Imagem Original")
            st.sidebar.image(our_image, width=150)


        st.markdown("Verifique o processo com a imagem disponível ou com qualquer outra escolhida:")
        st.image(our_image)

        st.markdown(" Primeiramente vamos converter a imagem para tons de cinza:")
        converted_image = np.array(our_image.convert('RGB'))
        gray_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2GRAY)
        st.image(gray_image)

        st.markdown(" Logo em seguida aplicamos um filtro bilateral para reduzir o nível de ruídos da foto:")
        gray = cv2.bilateralFilter(gray_image, 13, 15, 15)
        st.image(gray)

        st.markdown(
            " Agora vamos aplicar um transformação morfológica, denominada Black Hat Transform(para saber mais [acesse](https://www.geeksforgeeks.org/top-hat-and-black-hat-transform-using-python-opencv/)).")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (
        40, 13))  # tamanho do kernel é proporcional ao tamanho da placa que queremos encontrar na imagem
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        st.image(black_hat)

        st.markdown(" Em seguida, utilizando o filtro Sobel e o numpy.absolute, vamos detectar contornos na vertical:")
        gradient_x = cv2.Sobel(black_hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradient_x = np.absolute(gradient_x)
        # valore acima de 255(ddepth=cv2.CV32F)
        # extrair valores minimos e máximos
        (minimo, maximo) = (np.min(gradient_x), np.max(gradient_x))
        # normalizar (valor - min) / (max - min)
        gradient_x = 255 * ((gradient_x - minimo) / (maximo - minimo))
        # vamos para UINT8
        gradient_x = gradient_x.astype("uint8")
        st.image(gradient_x)

        st.markdown(
            " A fim de ressaltar a placa, na imagem, aplicaremos um Blur através de uma função Gaussiana. Então fazemos uma operação de Close para filtrar ruídos e logo transformamos a imagem em binária:")
        gradient_x = cv2.GaussianBlur(gradient_x, (5, 5), 0)
        # cv2_imshow(gradient_x)
        gradient_x = cv2.morphologyEx(gradient_x, cv2.MORPH_CLOSE, kernel)
        # cv2_imshow(gradient_x)
        thres = cv2.threshold(gradient_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        st.image(thres)

        st.markdown(
            " Realizamos uma operação de opening na imagem, para retirar ruídos brancos no fundo preto(para entender [acesse](https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html))")
        thres = cv2.erode(thres, None, iterations=2)  # 2
        thres = cv2.dilate(thres, None, iterations=1)  # 1
        st.image(thres)

        st.markdown(
            "Por fim, com a função cv2.findContours, vamos segmentar a região da placa no carro e imprimir a imagem respectiva: ")
        contornos = cv2.findContours(thres.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contornos = imutils.grab_contours(contornos)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]

        for c in contornos:
            (x, y, w, h) = cv2.boundingRect(c)
            proporcao = w / h

            # dimensoes da placa: 40x13cm
            if proporcao >= 2.5 and proporcao <= 4:
                area_placa_identificada = gray[y: y + h, x: x + w]
                placa_recortada = cv2.threshold(area_placa_identificada, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                st.image(placa_recortada)
                st.markdown("É possível também imprimir a placa colorida:")
                placa_colorida = converted_image[y: y + h, x: x + w]
                st.image(placa_colorida)


    elif escolha == 'Sobre o projeto':
        st.subheader("Este é um projeto da Masterclass Introdução à Visão Computacional.")
        st.markdown("Para saber mais informações, acesse [este site.](https://sigmoidal.ai)")
        st.text("Carlos Melo")
        st.success("Instagram @carlos_melo.py")
        st.text("Vladimir Simões da Luz Junior ")
        st.success("Instagram @vladiluzjr")
        st.info("[Linkedin](https://www.linkedin.com/in/vladimir-simoes-da-luz-junior/)")
        st.video("https://www.youtube.com/watch?v=JhkhbTTxlQg")


if __name__ == '__main__':
    main()
