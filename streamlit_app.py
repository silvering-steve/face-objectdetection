import streamlit as st


def main():
    st.set_page_config(page_title="Face Detection")
    st.title("Face Object Detection")
    st.caption("Detect face, nose, mouth, and eyes using your camera")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.write("Video Capture Ended")
            break

        frame_placeholder.image(frame, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
