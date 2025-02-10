# Image Attention Heatmap Generator

## 1. Introduction

This project uses AI (Google's Gemini Pro Vision) to predict how humans will visually engage with an image, particularly advertisements.  It generates two main outputs:

1.  **Areas of Interest (AOIs) with Bounding Boxes:**  Identifies key elements in the image (like faces, text, specific design features) and draws green bounding boxes around them.  Each box is labeled with the object's description and an "attention score."
2.  **Attention Heatmap:**  Creates a heatmap overlay on the original image, visualizing the predicted distribution of visual attention.  The heatmap uses a color gradient (green-yellow-orange-red) to show areas of increasing predicted attention, with red indicating the highest predicted attention.

The tool is built using Streamlit, making it easy to interact with through a web browser.  It's designed to help analyze the effectiveness of visual content, particularly for marketing and advertising.

## 2. Installation

This section provides step-by-step instructions to get the application running, even if you're not a Python expert.

**Prerequisites:**

*   **Python 3.8 or higher:**  You'll need Python installed.  You can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).  During installation, make sure to check the box that says "Add Python X.X to PATH" (where X.X is the version number).
*   **A Google Cloud Account and API Key:**  This project uses the Google Gemini Pro Vision API.  You'll need a Google Cloud account and an API key with access to this API. Create it here [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

**Steps:**

1.  **Clone the Repository**
    Open a terminal (or command prompt) and run:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    Replace `<repository_url>` with the actual URL of this GitHub repository, and `<repository_name>` with the name of the folder it creates.

2.  **Create a Virtual Environment**  
    This isolates the project's dependencies.  In your terminal, within the project directory, run:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**
    *   **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    You should see `(venv)` at the beginning of your terminal prompt, indicating the environment is active.

4.  **Install Dependencies**  
    Install the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    This command reads the `requirements.txt` file and installs all the necessary libraries.


5.  **Set up Streamlit Secrets**
    *   Create a folder `.streamlit` in root directory.
    *   Create a file called `secrets.toml` inside `.streamlit` folder.
    *   Add your Google Gemini API key to `.streamlit/secrets.toml` file:
        ```toml
        GEMINI_API_KEY = "YOUR_API_KEY_HERE"
        ```
        Replace `YOUR_API_KEY_HERE` with your actual Gemini API key. **Do not commit this file to your git repository.**

6.  **Run the Application**  Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    This will open a new tab in your web browser with the application running.

7. **Deactivate Virtual Environment**
   ```bash
    deactivate
   ```
## 3. How It Works

The application's workflow is as follows:
1. **Image Upload:** The user uploads an image (.jpg, .png, .jpeg, or .webp) using the file uploader in the Streamlit interface. The image is resized, while keeping the ratio, to a maximum of 512x512 pixels.

2. **"Find AOIs" Button (Left Button):** When the user clicks this button, the `handle_left_button` function is triggered.
3. **Attention Analysis (attention_analysis in ai_utils.py):** The uploaded image and a detailed prompt (neuro_prompt in prompts.py) are sent to the Gemini Pro Vision API. The prompt instructs the model to predict areas of visual attention based on various factors (faces, text, contrast, etc.) and heuristics derived from eye-tracking research. The API returns a JSON response containing a list of "Attention Elements," each with a label (description of the element) and an attention_score (normalized 0-1). The AttentionAnalysis Pydantic model in data_schemas.py is used to enforce the structure of the API response, ensuring type safety.
4. **Bounding Box Generation (get_bounding_boxes in ai_utils.py)**: Because getting structured bounding box data directly from Gemini wasn't reliable, a second call to the API is made (See [2D spatial understanding with Gemini 2.0](https://github.com/google-gemini/cookbook/blob/4437c15aa0bcb8f397b49f5b2e549f64e3a0985f/quickstarts/Spatial_understanding.ipynb)). This time, a different prompt (bounding_boxes_prompt in prompts.py), dynamically generated using identified labels from first call with create_bbox_prompt function. This prompt asks for the bounding box coordinates (y1, x1, y2, x2, and the label) of each detected AOI. The results comes as text, and the function parse_response_json parse the text and creates the json, and result is coerced to match BoundingBox model.
5. **Combining AOIs and Bounding Boxes (add_bboxes_to_aois in ai_utils.py):** The bounding box data is merged with the initial attention analysis results that has attention score. The normalize_bbox function in image_utils.py converts the coordinates from a 1000x1000 reference frame to the actual image dimensions, and switches the y1,x1,y2,x2 format to x1,y1,x2,y2.
6. **Overlay Bounding Boxes (overlay_bounding_boxes in image_utils.py):** Green bounding boxes, labels, and attention scores are drawn on a copy of the original image. This image with the overlays is displayed in the right column of the Streamlit app.
7. **"Create Heatmap" Button (Right Button):** When the user clicks this button (enabled only after AOIs are found), the `handle_right_button` function is triggered.
8. **Heatmap Generation (create_heatmap in image_utils.py):** This function takes the original image and the AOI data (including bounding boxes and attention scores) as input.
9. **Gaussian Distribution:** For each AOI, an oval-shaped Gaussian distribution is calculated. The standard deviation of the Gaussian is determined by the size of the bounding box (smaller boxes get smaller, more focused heat spots, and larger boxes get broader heat spots). 
10. **Clamping (SIGMA_MIN, SIGMA_MAX)** prevents excessively small or large Gaussians. There is an inverse scaling factor to ensure that larger areas do not get more attention.
11. **Intensity Scaling:** The attention score from the AOI is used to scale the intensity of the Gaussian. A power function increases contrast of the gaussian.
12. **Accumulation:** The individual Gaussian distributions for all AOIs are added together to create a combined heatmap.
13. **Normalization:** The combined heatmap is normalized to the range 0-1.
14. **Color Mapping (create_color_mapping in image_utils.py):** A custom color mapping function converts the normalized heatmap values to RGBA colors. The mapping transitions from transparent to green, then to yellow, orange, and finally red, with varying transparency levels.
15. **Blur:** The heatmap is blurred with gaussian blur to create smoother visualisation.
16. **Overlay:** The colored heatmap is blended with the original image using alpha compositing, creating the final heatmap overlay. This blended image is displayed in the right column.
17. **Session State:** Streamlit's `st.session_state` is used to store intermediate results (aois, image, image_with_boxes, heatmap, previous_upload). This ensures that the application's state is maintained across interactions and re-renders, and handles cases where the input image changed.

## 4. Key Files and Their Roles

*  **app.py:** The main Streamlit application file. Handles user interface, button clicks, and overall workflow.
*  **ai_utils.py:** Contains functions for interacting with the Gemini API (attention_analysis, get_bounding_boxes) and processing the API responses (parse_response_json, add_bboxes_to_aois).
*  **image_utils.py:** Contains functions for image processing, including normalizing bounding boxes (normalize_bbox), drawing bounding boxes (overlay_bounding_boxes), and creating the heatmap (create_heatmap).
*  **data_schemas.py:** Defines Pydantic models (AttentionElement, AttentionAnalysis, BoundingBox) and TypedDicts (AOIS) for data validation and type hinting.
*  **prompts.py:** Stores the text prompts used for interacting with the Gemini API.
*  **prompt_utils.py:** Functions to generate prompts.
*  **requirements.txt:** Lists the required Python packages.

## 5. Further Reading
*  [2D spatial understanding with Gemini 2.0](https://github.com/google-gemini/cookbook/blob/4437c15aa0bcb8f397b49f5b2e549f64e3a0985f/quickstarts/Spatial_understanding.ipynb)
*  [Utilizing Eye-Tracking in Advertising: Preliminary Findings](https://ibimapublishing.com/articles/JMRCS/2024/404100/404100.pdf)
*  [INTRODUCTION TO EYE TRACKING: A HANDS-ON TUTORIAL FOR STUDENTS AND PRACTITIONERS](https://arxiv.org/pdf/2404.15435)
*  [An Eye Gaze Heatmap Analysis of Uncertainty Head-Up Display Designs for Conditional Automated Driving](https://arxiv.org/pdf/2402.17751)
*  [How Heatmaps and Eye Tracking Help to Optimize Sales](https://blog.saleslayer.com/how-heat-maps-eye-tracking-optimize-sales)
*  [An ultimate heatmap guide for product & marketing teams](https://www.zipy.ai/guide/heatmap)
*  [Behavioral Research in Advertising](https://www.realeye.io/use-cases/use-cases-testing-advertisements)
