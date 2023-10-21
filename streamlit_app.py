import cv2
import streamlit as st

from src.model.predictor import model

hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''


def main():
    st.set_page_config(page_title="Face Detection")
    st.title("Face Object Detection")

    st.markdown(hide_img_fs, unsafe_allow_html=True)

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

        predicted = model.model.predict(frame)
        predict_plot = predicted[0].plot()

        frame_placeholder.image(predict_plot, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
