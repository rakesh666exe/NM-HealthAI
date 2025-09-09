import gradio as gr 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================
# Load Model & Tokenizer
# ============================
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================
# Response Generator
# ============================
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# ============================
# Features
# ============================
def disease_prediction(symptoms):
    prompt = f"""
Based on the following symptoms, provide possible medical conditions and general medication suggestions. 
Always emphasize the importance of consulting a doctor for proper diagnosis.

Symptoms: {symptoms}

Possible conditions and recommendations:

**IMPORTANT: This is for informational purposes only. Please consult a healthcare professional.**
Analysis:
"""
    return generate_response(prompt, max_length=1200)

def treatment_plan(condition, age, gender, medical_history):
    prompt = f"""
Generate personalized treatment suggestions for the following patient:

Medical Condition: {condition}
Age: {age}
Gender: {gender}
Medical History: {medical_history}

Personalized treatment plan including home remedies and general medication guidelines:

**IMPORTANT: This is for informational purposes only. Please consult a healthcare professional.**
Treatment Plan:
"""
    return generate_response(prompt, max_length=1200)

# ============================
# Gradio UI
# ============================
with gr.Blocks(css="""
body {background: linear-gradient(135deg, #ece9e6, #ffffff);}
h1 {text-align: center; font-size: 2.5em; color: #2c3e50;}
.gradio-container {max-width: 1100px; margin: auto;}
.card {background: white; border-radius: 15px; padding: 20px; box-shadow: 0 6px 15px rgba(0,0,0,0.1);}
button {border-radius: 12px !important; font-size: 1.1em;}
""") as app:

    gr.Markdown("<h1>🩺 Medical AI Assistant</h1>")
    gr.Markdown("<p style='text-align:center; color: red;'><b>⚠ Disclaimer:</b> This tool is for informational purposes only. Always consult healthcare professionals.</p>")

    with gr.Tabs():
        with gr.TabItem("🔍 Disease Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        symptoms_input = gr.Textbox(
                            label="Enter Symptoms",
                            placeholder="e.g., fever, headache, cough, fatigue...",
                            lines=5
                        )
                        predict_btn = gr.Button("🔎 Analyze Symptoms", variant="primary")
                with gr.Column(scale=2):
                    with gr.Group(elem_classes="card"):
                        prediction_output = gr.Textbox(
                            label="Possible Conditions & Recommendations",
                            lines=20
                        )

            predict_btn.click(disease_prediction, inputs=symptoms_input, outputs=prediction_output)

        with gr.TabItem("💊 Treatment Plans"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        condition_input = gr.Textbox(
                            label="Medical Condition",
                            placeholder="e.g., diabetes, hypertension, migraine...",
                            lines=2
                        )
                        age_input = gr.Number(label="Age", value=30)
                        gender_input = gr.Dropdown(
                            choices=["Male", "Female", "Other"],
                            label="Gender",
                            value="Male"
                        )
                        history_input = gr.Textbox(
                            label="Medical History",
                            placeholder="Previous conditions, allergies, medications or None",
                            lines=3
                        )
                        plan_btn = gr.Button("📋 Generate Treatment Plan", variant="primary")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes="card"):
                        plan_output = gr.Textbox(
                            label="Personalized Treatment Plan",
                            lines=20
                        )

            plan_btn.click(
                treatment_plan,
                inputs=[condition_input, age_input, gender_input, history_input],
                outputs=plan_output
            )

# ============================
# Launch App
# ============================
app.launch(share=True)
