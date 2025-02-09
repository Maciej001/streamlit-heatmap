import streamlit as st
from PIL import Image
import io
from ai_utils import attention_analysis
from prompts import neuro_prompt
from typing import Optional
from data_schemas import AttentionAnalysis
from ai_utils import get_bounding_boxes, add_bboxes_to_aois
from image_utils import overlay_bounding_boxes, create_heatmap

st.set_page_config(layout="wide")
left_column, right_column = st.columns(2)

if 'aois' not in st.session_state:
    st.session_state.aois = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'image_with_boxes' not in st.session_state:
    st.session_state.image_with_boxes = None
if 'heatmap' not in st.session_state:
    st.session_state.heatmap = None
if 'previous_upload' not in st.session_state:
    st.session_state.previous_upload = None
    
def handle_left_button():
    st.session_state.aois = attention_analysis(neuro_prompt, st.session_state.image)
    bboxes = get_bounding_boxes(st.session_state.aois, st.session_state.image)
    st.session_state.aois = add_bboxes_to_aois(st.session_state.aois, bboxes, st.session_state.image)
    st.session_state.image_with_boxes = overlay_bounding_boxes(st.session_state.image, st.session_state.aois)
    
def handle_right_button():
    st.session_state.heatmap = create_heatmap(st.session_state.image, st.session_state.aois)
    
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None and uploaded_file != st.session_state.previous_upload:
    st.session_state.aois = None
    st.session_state.image = None
    st.session_state.image_with_boxes = None
    st.session_state.heatmap = None
    st.session_state.previous_upload = uploaded_file

with left_column:
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        image.thumbnail([512,512], Image.Resampling.LANCZOS)
        st.session_state.image = image
        st.image(image, use_container_width=True)
        
        # Buttons
        left, right = st.columns(2)
        
        # Left Button – Find AOIs
        if left.button("Find AOIs", use_container_width=True):
            handle_left_button()
            
        # Right Button – Create Heatmap
        if right.button(
            "Create heatmap", 
            use_container_width=True,
            disabled=st.session_state.aois is None
        ):
            handle_right_button()


with right_column:
    if st.session_state.image_with_boxes is not None:
        st.image(st.session_state.image_with_boxes, use_container_width=True)
    if st.session_state.heatmap is not None:
        st.image(st.session_state.heatmap, use_container_width=True)
    else:
        st.write("Upload an image and find AOIs to see the analysis here.")
