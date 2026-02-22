"""
Medical RAG Backend
Handles PDF processing, embeddings, vector storage, and structured comparison output
"""

import os
import re
from typing import List, Tuple

import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class MedicalRAGBackend:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", api_key: str = None):
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.vector_store = None
        self.chunks = []
        self.api_key = api_key  # For future use 

    # =========================
    # STEP 1: Extract Text
    # =========================
    def extract_text_from_pdf(self, pdf_file) -> str:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n\n+', '\n', text)
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    # =========================
    # STEP 2: Chunking
    # =========================
    def chunk_text(self, text: str, chunk_size=800, overlap=100) -> List[str]:
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        self.chunks = chunks
        return chunks

    # =========================
    # STEP 3: Embeddings
    # =========================
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        return self.embedding_model.encode(chunks, convert_to_numpy=True)

    # =========================
    # STEP 4: Build FAISS
    # =========================
    def build_vector_store(self, embeddings: np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)
        self.vector_store = faiss.IndexFlatL2(self.embedding_dim)
        self.vector_store.add(embeddings)

    # =========================
    # STEP 5: Retrieval
    # =========================
    def retrieve_relevant_chunks(self, query: str, top_k=5) -> List[str]:
        if not self.vector_store:
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
        _, indices = self.vector_store.search(query_embedding, min(top_k, len(self.chunks)))

        return [self.chunks[i] for i in indices[0]]

    # =========================
    # PARAMETER EXTRACTION
    # =========================
    def _extract_medical_parameters(self, context: str) -> List[dict]:

        parameters = []

        known_params = {
            'Hemoglobin': 'g/dL',
            'RBC Count': 'million/µL',
            'WBC Count': '/µL',
            'Platelet Count': '/µL',
            'Hematocrit': '%',
            'Total Bilirubin': 'mg/dL',
            'ALT (SGPT)': 'U/L',
            'AST (SGOT)': 'U/L',
            'Alkaline Phosphatase': 'U/L',
            'Total Cholesterol': 'mg/dL',
            'LDL Cholesterol': 'mg/dL',
            'HDL Cholesterol': 'mg/dL',
            'Triglycerides': 'mg/dL',
            'Fasting Blood Sugar': 'mg/dL',
            'HbA1c': '%'
        }

        for name, unit in known_params.items():

            # Extract value
            value_pattern = rf"{name}\s*[:\s]+([\d.,]+)\s*{re.escape(unit)}"
            value_match = re.search(value_pattern, context, re.IGNORECASE)

            if not value_match:
                continue

            value_str = value_match.group(1).replace(",", "")
            value = f"{value_str} {unit}"

            remaining_text = context[value_match.end():value_match.end() + 300]

            reference = None
            status = "Normal"

            # Case 1: min–max
            range_match = re.search(r'([\d.,]+)\s*[–-]\s*([\d.,]+)', remaining_text)
            if range_match:
                min_val = float(range_match.group(1).replace(",", ""))
                max_val = float(range_match.group(2).replace(",", ""))
                reference = f"{range_match.group(1)} – {range_match.group(2)} {unit}"

                val_num = float(value_str)

                if val_num < min_val:
                    status = "Low"
                elif val_num > max_val:
                    status = "High"

            # Case 2: < value
            elif "<" in remaining_text:
                lt_match = re.search(r'<\s*([\d.,]+)', remaining_text)
                if lt_match:
                    ref_val = float(lt_match.group(1).replace(",", ""))
                    reference = f"< {lt_match.group(1)} {unit}"

                    val_num = float(value_str)
                    if val_num >= ref_val:
                        status = "High"

            # Case 3: > value
            elif ">" in remaining_text:
                gt_match = re.search(r'>\s*([\d.,]+)', remaining_text)
                if gt_match:
                    ref_val = float(gt_match.group(1).replace(",", ""))
                    reference = f"> {gt_match.group(1)} {unit}"

                    val_num = float(value_str)
                    if val_num <= ref_val:
                        status = "Low"

            if reference:
                parameters.append({
                    "name": name,
                    "value": value,
                    "range": reference,
                    "status": status
                })

        return parameters

    # =========================
    # GENERATE STRUCTURED OUTPUT
    # =========================
    def _generate_output(self, context: str) -> str:
        """Generate detailed medical report breakdown with consequences and recommendations"""
        explanation = "# 📋 MEDICAL REPORT ANALYSIS\n\n"
        explanation += "Below is a detailed breakdown of each test and what it means.\n\n"

        parameters = self._extract_medical_parameters(context)

        # Define detailed explanations for each parameter
        param_info = {
            "Hemoglobin": {
                "what_is": "protein in red blood cells that carries oxygen throughout your body",
                "low_consequence": "Low hemoglobin means your blood cannot carry enough oxygen, leading to fatigue, weakness, shortness of breath, and dizziness",
                "high_consequence": "High hemoglobin can thicken blood and increase risk of clots",
                "low_action": "Eat iron-rich foods (spinach, red meat, beans), increase vitamin C intake, and consult your doctor",
                "high_action": "Stay hydrated, reduce iron intake, and consult your doctor"
            },
            "RBC Count": {
                "what_is": "number of red blood cells that carry oxygen",
                "low_consequence": "Low RBC count (anemia) causes fatigue, weakness, pale skin, and reduced oxygen delivery to organs",
                "high_consequence": "High RBC count can increase blood viscosity, raising risk of blood clots and stroke",
                "low_action": "Increase iron intake, eat vitamin B12 foods, rest more, and see your doctor",
                "high_action": "Stay well-hydrated, avoid dehydration, and consult your doctor"
            },
            "WBC Count": {
                "what_is": "number of white blood cells that fight infections and protect your immune system",
                "low_consequence": "Low WBC count weakens your immune system, making you susceptible to infections",
                "high_consequence": "High WBC count indicates your body may be fighting an infection, inflammation, or a more serious condition",
                "low_action": "Avoid crowds/sick people, maintain hygiene, rest well, and see your doctor",
                "high_action": "Rest, stay hydrated, monitor for signs of infection, and consult your doctor"
            },
            "Platelet Count": {
                "what_is": "number of platelets that help your blood clot and stop bleeding",
                "low_consequence": "Low platelets increase risk of heavy bleeding, bruising, and bleeding gums",
                "high_consequence": "High platelets can increase risk of blood clots, stroke, or heart attack",
                "low_action": "Avoid injuries, limit strenuous activities, avoid blood thinners, and see your doctor",
                "high_action": "Stay hydrated, monitor for thrombus symptoms, and consult your doctor"
            },
            "Hematocrit": {
                "what_is": "percentage of red blood cells in your total blood volume",
                "low_consequence": "Low hematocrit (anemia) means less oxygen delivery, causing fatigue and weakness",
                "high_consequence": "High hematocrit increases blood thickness, raising stroke and clot risk",
                "low_action": "Increase iron, B12, and folate intake; get adequate rest; see your doctor",
                "high_action": "Increase water intake, avoid dehydration, limit iron, and consult your doctor"
            },
            "Total Bilirubin": {
                "what_is": "yellow pigment from breakdown of old red blood cells, processed by the liver",
                "low_consequence": "Low bilirubin is rarely a concern but may indicate certain conditions",
                "high_consequence": "High bilirubin indicates liver problems, jaundice (yellowing of skin/eyes), and potential liver damage",
                "low_action": "No specific action needed; monitor",
                "high_action": "Avoid alcohol, reduce fatty foods, eat liver-healthy foods, and see your doctor immediately"
            },
            "ALT (SGPT)": {
                "what_is": "liver enzyme that indicates liver function and cell damage",
                "low_consequence": "Low ALT is generally normal",
                "high_consequence": "High ALT indicates liver inflammation, damage, or disease (hepatitis, fatty liver, cirrhosis)",
                "low_action": "No action needed",
                "high_action": "Avoid alcohol completely, reduce fatty/fried foods, limit medications, and see a hepatologist"
            },
            "AST (SGOT)": {
                "what_is": "enzyme found in liver, heart, and muscle; indicates tissue damage",
                "low_consequence": "Low AST is normal",
                "high_consequence": "High AST indicates liver damage, heart disease, or muscle issues",
                "low_action": "No action needed",
                "high_action": "Avoid alcohol, eat healthy foods, manage stress, and consult your doctor"
            },
            "Alkaline Phosphatase": {
                "what_is": "enzyme in bone and liver; indicates bone/liver health",
                "low_consequence": "Low levels are rare but may indicate nutrient deficiency",
                "high_consequence": "High levels may indicate bone disease, liver disease, or healing fractures",
                "low_action": "Ensure adequate nutrition; see your doctor if persistent",
                "high_action": "Get calcium and vitamin D; avoid bone-damaging activities; see your doctor"
            },
            "Total Cholesterol": {
                "what_is": "total amount of cholesterol in blood; high levels increase heart disease risk",
                "low_consequence": "Low cholesterol is rarely harmful unless extremely low",
                "high_consequence": "High cholesterol leads to artery blockage, heart attack, and stroke",
                "low_action": "Monitor; no major action usually needed",
                "high_action": "Exercise 30+ mins daily, reduce saturated fats, eat soluble fiber, stop smoking, and see your doctor"
            },
            "LDL Cholesterol": {
                "what_is": "\"bad\" cholesterol that builds up in artery walls, causing blockages",
                "low_consequence": "Low LDL is protective for heart health",
                "high_consequence": "High LDL greatly increases risk of heart attack and stroke from arterial blockage",
                "low_action": "Good; maintain current lifestyle",
                "high_action": "🚨 CRITICAL: Reduce red meat/dairy, eat heart-healthy fats (olive oil, fish), exercise daily, consider medication, and see your doctor immediately"
            },
            "HDL Cholesterol": {
                "what_is": "\"good\" cholesterol that removes bad cholesterol and protects your heart",
                "low_consequence": "Low HDL increases heart disease and stroke risk; poor heart protection",
                "high_consequence": "High HDL protects your heart and reduces disease risk",
                "low_action": "Increase aerobic exercise, eat omega-3 fish, reduce refined carbs, stop smoking, and see your doctor",
                "high_action": "Excellent; maintain current healthy lifestyle"
            },
            "Triglycerides": {
                "what_is": "type of fat in blood; high levels increase heart disease risk",
                "low_consequence": "Low triglycerides are healthy",
                "high_consequence": "High triglycerides increase risk of heart disease, stroke, and pancreatitis",
                "low_action": "Good; maintain current lifestyle",
                "high_action": "Eliminate sugar, reduce carbs, lose weight, increase aerobic exercise, reduce alcohol, and see your doctor"
            },
            "Fasting Blood Sugar": {
                "what_is": "blood sugar level after 8+ hours of fasting; indicates diabetes risk",
                "low_consequence": "Very low blood sugar can cause hypoglycemia (dizziness, shakiness, confusion)",
                "high_consequence": "High fasting sugar is pre-diabetic/diabetic, increasing diabetes, heart disease, and kidney damage risk",
                "low_action": "Eat regular meals with carbs and protein; monitor for hypoglycemia symptoms",
                "high_action": "Reduce sugar and refined carbs, eat fiber-rich foods, exercise, lose weight, and see your doctor"
            },
            "HbA1c": {
                "what_is": "average blood sugar over 3 months; indicates long-term diabetes control",
                "low_consequence": "Low HbA1c is healthy",
                "high_consequence": "High HbA1c indicates pre-diabetes or diabetes with risk of complications (kidney disease, nerve damage, blindness)",
                "low_action": "Good; maintain current lifestyle",
                "high_action": "⚠️ IMPORTANT: Lifestyle changes now can prevent diabetes. Reduce sugar/carbs, exercise 30+ mins daily, lose weight if needed, and see your doctor"
            }
        }

        categories = {
            "🩸 BLOOD COUNT FINDINGS (CBC)": [
                "Hemoglobin", "RBC Count", "WBC Count",
                "Platelet Count", "Hematocrit"
            ],
            "🟡 LIVER FUNCTION TESTS (LFT)": [
                "Total Bilirubin", "ALT (SGPT)",
                "AST (SGOT)", "Alkaline Phosphatase"
            ],
            "💔 CHOLESTEROL & LIPID PROFILE (FAT IN BLOOD)": [
                "Total Cholesterol", "LDL Cholesterol",
                "HDL Cholesterol", "Triglycerides"
            ],
            "🍬 BLOOD SUGAR (DIABETES SCREENING)": [
                "Fasting Blood Sugar", "HbA1c"
            ]
        }

        for category, param_list in categories.items():
            explanation += f"\n## {category}\n\n"

            for param in param_list:
                data = next((p for p in parameters if p["name"] == param), None)
                if not data:
                    continue

                # Determine status indicator
                if data['status'] == 'High':
                    indicator = '❌'
                elif data['status'] == 'Low':
                    indicator = '❌'
                else:
                    indicator = '✅'
                
                explanation += f"### {indicator} {param.upper()}: {data['status'].upper()}\n"
                explanation += f"- **Your value:** {data['value']}\n"
                explanation += f"- **Normal range:** {data['range']}\n\n"
                
                # Add detailed explanation
                if param in param_info:
                    info = param_info[param]
                    explanation += f"**What is {param}?**\n"
                    explanation += f"- {info['what_is']}\n\n"
                    
                    if data['status'] == 'High':
                        explanation += f"**⚠️ Consequences of HIGH {param}:**\n"
                        explanation += f"- {info['high_consequence']}\n\n"
                        explanation += f"**💊 What to do:**\n"
                        explanation += f"- {info['high_action']}\n\n"
                    elif data['status'] == 'Low':
                        explanation += f"**⚠️ Consequences of LOW {param}:**\n"
                        explanation += f"- {info['low_consequence']}\n\n"
                        explanation += f"**💊 What to do:**\n"
                        explanation += f"- {info['low_action']}\n\n"
                    else:  # Normal
                        explanation += f"**✅ Status:**\n"
                        explanation += f"- Your {param} is within normal range and healthy. Keep maintaining your current lifestyle.\n\n"
                
                explanation += "---\n\n"

        explanation += """\n## 🏥 IMPORTANT RECOMMENDATIONS

### ⚠️ When to See a Doctor:

**See your doctor IMMEDIATELY if you have:**
- Very high LDL cholesterol (> 190 mg/dL)
- High blood sugar with symptoms (extreme thirst, frequent urination, fatigue)
- High bilirubin with jaundice (yellowing of skin/eyes)
- Low platelet count (< 50,000)
- Abnormal liver function tests (ALT/AST very high)

**Schedule an appointment with your doctor if you have:**
- Any values that are out of normal range
- Multiple abnormal results
- Results that concern you
- Symptoms like fatigue, weakness, shortness of breath, chest pain, or abdominal pain

### 📋 What to bring to your doctor:

1. This medical report and analysis
2. A list of your current medications
3. A record of your symptoms
4. Previous test results for comparison
5. Questions or concerns about your health

---

## ⚠️ IMPORTANT DISCLAIMER

This analysis is for **educational purposes only** and is **NOT** a medical diagnosis or medical advice.

- This is a comparison of your laboratory values against reference ranges
- It does NOT provide a diagnosis
- It does NOT replace professional medical consultation
- You should **ALWAYS consult with a qualified healthcare professional** for:
  - Interpretation of your specific results
  - Medical diagnosis or treatment
  - Personalized health guidance
  - Medication recommendations

**Your doctor should evaluate these results in the context of your complete medical history, symptoms, and other clinical findings.**

**Health is personal. Please consult your healthcare provider for guidance tailored to YOUR specific situation.**
"""

        return explanation

        return explanation
    
    def generate_explanation(self, query: str, relevant_chunks: List[str]) -> str:
        """Generate explanation from relevant chunks (API method)"""
        combined_context = "\n\n".join(relevant_chunks)
        return self._generate_output(combined_context)

    # =========================
    # FULL PIPELINE
    # =========================
    def process_report(self, pdf_file) -> Tuple[str, List[str]]:

        raw_text = self.extract_text_from_pdf(pdf_file)

        chunks = self.chunk_text(raw_text)

        embeddings = self.create_embeddings(chunks)

        self.build_vector_store(embeddings)

        query = "medical test results with reference ranges"

        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=5)

        combined_context = "\n\n".join(relevant_chunks)

        explanation = self._generate_output(combined_context)

        return explanation, relevant_chunks
