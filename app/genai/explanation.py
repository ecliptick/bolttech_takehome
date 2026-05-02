import json
import logging
from typing import Literal

from app.genai.llm import generate_content

logger = logging.getLogger(__name__)

PersonaType = Literal["customer", "claims_adjuster"]

def generate_claim_explanation(
    claim_details: dict,
    model_prediction: int,
    model_probability: float,
    feature_importances: dict,
    target_persona: PersonaType,
) -> str:
    """
    Generate a tailored explanation for a claim decision based on the persona.
    """
    
    # Prompt engineering strategy:
    # We use a single prompt template that changes its tone and structure based on the requested persona.
    # 
    # For a CUSTOMER:
    # - Tone: Empathetic, clear, and non-technical.
    # - Focus: Explain *why* their specific claim was approved or denied based on their policy and damage.
    # - Actionable Insights: What they need to do next (e.g., pay the excess fee, provide more evidence, or options for repair).
    #
    # For a CLAIMS ADJUSTER:
    # - Tone: Analytical, objective, and data-driven.
    # - Focus: Highlight the ML model's confidence, the key features driving the decision (e.g., specific symptom flags, excess/RRP ratio), and potential anomalies.
    # - Actionable Insights: Recommendations for manual review, potential fraud signals, or policy edge cases to investigate.

    prompt = f"""
You are an expert insurance claims AI assistant.
Your task is to explain a machine learning model's claim decision for a mobile device insurance claim.

Claim Details:
{json.dumps(claim_details, indent=2)}

Model Decision: {"Approved" if model_prediction == 1 else "Declined"}
Model Confidence (Probability of Approval): {model_probability:.1%}
Key Contributing Features: {json.dumps(feature_importances, indent=2)}

Target Persona: {target_persona.upper()}

Please write a highly tailored explanation for the target persona.

If the Persona is CUSTOMER:
1. Use an empathetic, clear, non-technical tone.
2. Address the customer directly ("Dear Customer" or "Hi").
3. Explain the decision clearly based on the facts of the damage and their policy. Do not mention "machine learning models" or "probabilities".
4. Provide actionable next steps (e.g., "Next, please arrange to pay the excess fee" or "Unfortunately, this falls outside your coverage, but you can appeal by...").

If the Persona is CLAIMS_ADJUSTER:
1. Use a professional, analytical, and data-driven tone.
2. Focus on why the model made this prediction, highlighting the key contributing features and the confidence score.
3. Call out any specific anomalies in the claim details (e.g., high excess fee relative to device value, unusual damage combinations like both water and screen damage).
4. Provide actionable insights for the adjuster (e.g., "Recommend fast-tracking approval" or "Recommend manual investigation due to borderline probability and contradictory symptoms").

Output ONLY the explanation text.
"""

    return generate_content(prompt)
