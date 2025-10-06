import streamlit as st
from PIL import Image
import tempfile
from neural_style_transfer import load_image, im_convert, run_style_transfer

st.title("Neural Style Transfer App")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
steps = st.slider("Number of optimization steps", 100, 1000, 300, 50)

if content_file and style_file:
    content_img = Image.open(content_file).convert('RGB')
    style_img = Image.open(style_file).convert('RGB')
    st.image([content_img, style_img], caption=["Content", "Style"], width=256)
    if st.button("Run Style Transfer"):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as cfile, tempfile.NamedTemporaryFile(suffix=".jpg") as sfile:
            content_img.save(cfile.name)
            style_img.save(sfile.name)
            content = load_image(cfile.name)
            style = load_image(sfile.name)
            output = run_style_transfer(content, style, num_steps=steps)
            out_img = im_convert(output)
            st.image(out_img, caption="Stylized Output")
            st.success("Done!")
