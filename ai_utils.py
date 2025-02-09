import streamlit as st

from typing import Optional, List
import json
import PIL.Image
# import google.generativeai as genai
from google import genai
from google.genai import types
from pydantic import BaseModel
from data_schemas import AttentionAnalysis, AttentionElement, AOIS, BoundingBox
from prompt_utils import create_bbox_prompt
from image_utils import normalize_bbox

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
model_name = 'gemini-2.0-pro-exp-02-05'


def attention_analysis(prompt: str, image: PIL.Image.Image) -> Optional[AttentionAnalysis]:
    """
    Analyze attention points in an image using Gemini API.

    Args:
        prompt (str): The prompt for the analysis
        image (PIL.Image.Image): The image to analyze

    Returns:
        Optional[AttentionAnalysis]: The analyzed attention data or None if analysis fails
    """

    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        print("Generating content...")
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type = 'application/json',
                response_schema = AttentionAnalysis
            )
        )

        print("Parsed", response.parsed)
        return response.parsed
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None
    
def parse_response_json(json_output: str):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def get_bounding_boxes(aois: AttentionAnalysis, image: PIL.Image.Image) -> List[BoundingBox]:
    """
    Structured output doesn't work for bounding boxes, 
    so we need to parse the JSON output to extract the bounding boxes.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        bboxes_prompt = create_bbox_prompt(aois)
        print("bboxes_prompt: ", bboxes_prompt)
        response = client.models.generate_content(
            model=model_name,
            contents=[bboxes_prompt, image],
            config=types.GenerateContentConfig(
                temperature=0
            )
        )
        print("Response from get_bounding_boxes:", response.text)

        return parse_response_json(response.text)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None
    
def find_aoi_by_label(aois: AttentionAnalysis, label: str) -> AttentionElement:
    for element in aois.elements:
        if element.label.lower() == label.lower():
            return element
    return None


def add_bboxes_to_aois(aois: AttentionAnalysis, bboxes: List[BoundingBox], image: PIL.Image.Image) -> List[AOIS]:
    with_bboxes = []
    print('aois: ', aois)
    print('bboxes: ', bboxes)
    for i, bounding_box in enumerate(json.loads(bboxes)):
        aoi_element = find_aoi_by_label(aois, bounding_box["label"])
        if aoi_element:
            norm_bbox = normalize_bbox(bounding_box["box_2d"], image.width, image.height)
            with_bboxes.append({
                "label": bounding_box["label"],
                "attention_score": aoi_element.attention_score,
                "bounding_box": norm_bbox
            })
    return with_bboxes
    