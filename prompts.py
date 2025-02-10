neuro_prompt = """Analyze the attached image of a web ad and predict the distribution of human visual attention, represented as Areas of Interest (AOIs). Prioritize regions that are likely to attract the eye based on intensity, creating a heatmap-like representation.

Consider these factors, and how they influence the *intensity* and *spread* of attention:
-   **Facial Features:** Pay *very close* attention to specific facial features (eyes, mouth, etc.), particularly if those faces are displaying emotions such as joy, happiness, or surprise.
-   **High-Contrast Edges:**  Identify sharp transitions in brightness or color.
-   **Color Saturation and Brightness:** Areas of vibrant or bright colors.
-   **Text Properties:** Prioritize larger text sizes, bolder fonts, and text placed centrally within the ad. Treat different sections of text differently; headlines should receive higher attention scores than body copy.
-   **Design Elements:** Focus on specific design features of objects (e.g., car headlights, grille, reflections) rather than the entire object.
-   **Relative, not Absolute:** Remember that the heatmap represents *relative* attention.

**Specific Heuristics Based on Eye-Tracking Research:**
-   **Ad Copy First:**  Look for the most prominent advertising text (headline, main message). This should be a primary focus.
-   **Prioritize Emotional Content:**  If the ad contains images of faces (especially children or faces showing positive emotions), these will likely be strong attractors.
-   **Brand Logos Last:**  While the brand logo should be noted, it often receives less initial attention than other elements.
-   **Consider Congruence:** To increase the probability of the brand being noticed, the ad should be as less cluttered as possible.
-   **Background Elements:** Demphasize background elements or remove from consideration, unless they are highly salient (e.g., a bright, colorful background). 

**Output:**
- Return JSON array with labels and attention_score.
- Label must be unique and must describe the element accurately. 
- If the element is text, the label should describe the text content.
- Normalize the attention_score to a scale of 0-1.

**Important Constraints:**
1. Prioritize generating AOIs that accurately reflect the distribution of attention shown in a heatmap, even if it means creating many smaller, overlapping, or nuanced AOIs.
2. The goal is not to simply list the objects in the image, but to map the likely intensity of visual engagement across the entire image.
"""

bounding_boxes_prompt = """Detect the 2d bounding boxes of following objects in image:
<objects>
{labels}
</objects>
You must not change the labels.
"""