import streamlit as st
from ai.main import slow_model, fast_model
import os
import tempfile


def the_page():
    model_selection=""

    st.header("Speech Recognition & Speaker Diarization")

    st.info("Beta: Only English audio is supported. Optimized for speed and stability, the system currently requires English-only files for accurate processing.")

    mp3_file = st.file_uploader(label="Please Upload Your .MP3 File", type=["mp3"])
    if mp3_file is not None:
        with tempfile. NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(mp3_file.read())
            temp_audio_path = tmp_file.name
        
        st.success(f"File Uploaded: {mp3_file.name}")
        st.audio(temp_audio_path)

    col1, col2 = st.columns([2,2])

    if mp3_file is not None:
        with col1:
            st.subheader("Get Standard Results")
            st.info("""
                **Standard Model**
                * **Speed:** Fast
                * **Accuracy:** High ASR | Low Diarization
                * **Best For:** Quick transcriptions where speaker separation is less critical.
                """)
            if st.button("Fast Model"):
                model_selection="fast_model"

        with col2:
            st.subheader("Get Pro Results")
            st.info("""
                **Pro Model**
                * **Speed:** Slow
                * **Accuracy:** Perfect for Both
                * **Best For:** Professional grade ASR and precise Speaker Diarization.
                """)
            if st.button("Pro Model"):
                model_selection="slow_model"
        if model_selection !="":
            with st.spinner():
                if model_selection=="fast_model":
                    with st.spinner("Analysing..."):
                        output = fast_model(temp_audio_path)
                    if output is not None:
                        st.success("Fast Model output generated!")
                        if st.download_button(label="Download Transcript", data=output, file_name="transcript_fast.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                            pass
                else:
                    with st.spinner("Analysing..."):
                        output = slow_model(temp_audio_path)
                    
                    if output is not None:
                        st.success("Pro Model output generated!")
                        if st.download_button(label="Download Transcript", data=output, file_name="transcript_fast.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                            pass