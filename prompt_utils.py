from data_schemas import AttentionAnalysis
from prompts import bounding_boxes_prompt

def get_aois_labels(aois: AttentionAnalysis) -> str:
    """
    Extracts and formats object labels from attention analysis results into a bullet-pointed string.

    Takes an AttentionAnalysis object containing detected elements and their labels,
    converts each label to lowercase, and formats them as markdown bullet points.
    The resulting string is formatted for use in subsequent AI prompts.

    Args:
        aois (AttentionAnalysis): Analysis results containing detected elements and their labels

    Returns:
        str: A formatted string where each line is a bullet point (-) followed by a lowercase label
             Example:
             - person
             - coffee cup
             - laptop
    """
    labels_list = ""
    for element in aois.elements:
        labels_list += f"- {element.label.lower()}\n"

    return labels_list 

def create_bbox_prompt(aois: AttentionAnalysis):
    labels = get_aois_labels(aois)
    fomatted_prompt = bounding_boxes_prompt.format(labels=labels)
    return fomatted_prompt
    