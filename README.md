# Medical Report Analyzer - Hybrid RAG + Rule-Based System

A hybrid medical report analyzer that combines **Retrieval-Augmented Generation (RAG)** with **rule-based parameter extraction** to analyze lab test results and provide educational comparisons with reference ranges.

## ✨ What This System Does

✅ **Intelligent Retrieval** - Uses RAG (embedding-based search) to find relevant test sections  
✅ **Parameter Extraction** - Regex-based extraction of medical lab values  
✅ **Reference Range Comparison** - Compares patient values against standard ranges  
✅ **Educational Explanations** - Provides context about what each parameter represents  
✅ **Structured Output** - Organizes results by medical test category (CBC, Liver, Lipids, Blood Sugar)  
✅ **No API Key Required** - Runs completely locally with free embedding model  

## 🏗️ Technical Architecture

This is a **6-step hybrid pipeline** combining RAG (steps 1-5) with rule-based logic (step 6):

### Steps 1-5: Retrieval-Augmented Generation (RAG)

**Step 1️⃣: Extract Text from PDF**
- Reads uploaded PDF file  
- Extracts all text from each page  
- Cleans unnecessary characters and whitespace  

**Step 2️⃣: Split Text into Chunks**
- Divides large medical reports into manageable chunks (500-800 characters)  
- Maintains overlap between chunks for context continuity  

**Step 3️⃣: Create Embeddings**
- Converts each chunk into vector representation using `SentenceTransformers`  
- Uses pre-trained model: `all-MiniLM-L6-v2` (384-dimensional embeddings)  
- Free, no external API calls needed  

**Step 4️⃣: Store in Vector Database**
- Stores embeddings in FAISS (Facebook AI Similarity Search)  
- Enables fast semantic similarity search  

**Step 5️⃣: Retrieve Relevant Chunks**
- Queries are converted to embeddings  
- FAISS performs similarity search to find relevant text  
- Retrieves top-5 chunks most relevant to medical parameters  

### Step 6: Rule-Based Parameter Extraction & Explanation

**Instead of using an LLM for generation**, this system uses:

1. **Regex Pattern Matching** - Extracts parameter values and reference ranges from text
2. **Deterministic Comparison Logic** - Compares patient values against known reference ranges
3. **Predefined Knowledge Dictionary** - Maps parameters to educational explanations
4. **Reference Range Thresholds** - Uses standard medical lab ranges for classification

**Why this approach?**
- ✅ Deterministic output (no hallucinations)
- ✅ Explainable results (exact extraction rules visible)
- ✅ No API costs
- ✅ Works offline
- ✅ Fast (<1 second parameter extraction)

## 📁 File Structure

```
medical-rag/
├── app.py                 # Streamlit UI (main application)
├── rag_backend.py         # RAG pipeline + rule-based extraction
├── requirements.txt       # Python dependencies
├── .env.example          # Optional environment variables
└── README.md             # This file
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

**That's it!** No API keys, databases, or complex setup needed.

## 📊 System Flow

```
PDF Upload
    ↓
[RAG Step 1-5] Extract, Chunk, Embed, Index, Retrieve Medical Text
    ↓
[Rule-Based Step 6] Extract Parameters via Regex
    ↓
Compare Against Reference Ranges
    ↓
Generate Educational Explanations from Knowledge Dictionary
    ↓
Display Organized Results by Test Category
```

## 📚 Output Format

Results are organized by medical test category:

### 🩸 BLOOD COUNT (CBC)
Shows: Hemoglobin, RBC Count, WBC Count, Platelet Count, Hematocrit

### 🟡 LIVER FUNCTION (LFT)  
Shows: Total Bilirubin, ALT (SGPT), AST (SGOT), Alkaline Phosphatase

### 💔 CHOLESTEROL & LIPID PROFILE
Shows: Total Cholesterol, LDL, HDL, Triglycerides

### 🍬 BLOOD SUGAR (DIABETES SCREENING)
Shows: Fasting Blood Sugar, HbA1c

## 🔍 For Each Parameter, You Get:

✅ **Status Indicator**
- ✅ NORMAL
- ❌ HIGH / LOW

📋 **Your Results**
- Your exact value
- Reference range from standard medical tables

📖 **Educational Information**
- What this parameter measures
- What abnormal values indicate
- General health context

⚠️ **Important:** This is educational context only, not medical advice.

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Processing** | PyPDF2 | Extract text from medical PDFs |
| **Text Embeddings** | SentenceTransformers | Convert text to vectors (FREE model) |
| **Vector Search** | FAISS | Fast semantic similarity search |
| **Parameter Extraction** | Regex + Rule-Based Logic | Deterministic value extraction |
| **Data Structures** | Python dicts | Knowledge mapping |
| **UI Framework** | Streamlit | Interactive web interface |

## 🔧 How It Works: Technical Deep Dive

### RAG Component (Steps 1-5)
```
Advantages:
- Finds test sections even in unstructured reports
- Handles varied PDF layouts
- Semantic search finds related information

Limitations:
- Retrieves text chunks, not structured tables
- May retrieve irrelevant sections
```

### Rule-Based Component (Step 6)
```
Advantages:
- No hallucinations (deterministic)
- Transparent extraction logic
- Fast and reliable
- No external API calls

Limitations:
- Regex patterns may not match all formats
- Cannot interpret complex clinical relationships
- Relies on standard reference ranges
```

## 📊 Supported Medical Parameters (15)

| Category | Parameters |
|----------|-----------|
| **CBC** | Hemoglobin, RBC Count, WBC Count, Platelet Count, Hematocrit |
| **Liver** | Total Bilirubin, ALT (SGPT), AST (SGOT), Alkaline Phosphatase |
| **Lipids** | Total Cholesterol, LDL Cholesterol, HDL Cholesterol, Triglycerides |
| **Blood Sugar** | Fasting Blood Sugar, HbA1c |

## 💡 Example: What You See

**Input:** Medical report with "Hemoglobin: 10.2 g/dL (Normal: 13.0-17.0)"

**Output:**
```
❌ HEMOGLOBIN: LOW
- Your value: 10.2 g/dL
- Normal range: 13.0 – 17.0 g/dL

What is Hemoglobin?
Protein in red blood cells that carries oxygen throughout your body

What LOW values indicate:
Low hemoglobin can result in reduced oxygen delivery and fatigue

This comparison is for educational purposes only. 
Consult a healthcare professional for medical interpretation.
```

## 📈 Performance

- **PDF Extraction:** <1 second
- **Chunking:** <1 second
- **Embedding Generation:** 2-5 seconds
- **Vector Index Creation:** <1 second
- **Parameter Extraction:** <1 second
- **Total Processing Time:** ~5-10 seconds per report

**All processing is local - no API calls.**

## 🔐 Privacy & Security

- ✅ PDFs processed locally on your machine
- ✅ No data sent to external servers
- ✅ No cloud storage
- ✅ Results only shown in your browser session
- ✅ Free embedding model from Hugging Face (local download only)

## ⚠️ Important Disclaimer

**This tool is for educational and informational purposes only.**

### What This System Does NOT Do:
- ❌ Does NOT provide medical diagnosis
- ❌ Does NOT replace professional medical advice
- ❌ Does NOT make clinical recommendations
- ❌ Does NOT interpret complex clinical relationships
- ❌ Does NOT validate against your personal medical history

### Reference Ranges:
- Standard lab reference ranges are used (may vary by lab)
- Results should be evaluated in context of your specific situation
- Your doctor may have different normal ranges

### When to See a Doctor:
- **Always** consult a healthcare professional for:
  - Interpretation of your specific test results
  - Understanding what they mean for YOUR health
  - Medical advice and treatment decisions
  - Medication or lifestyle recommendations

**Your healthcare provider should evaluate results in context of your complete medical history, symptoms, and clinical findings.**

## 🎯 What This Is Good For

✅ Understanding what medical tests measure  
✅ Learning about standard reference ranges  
✅ Educational exploration of lab reports  
✅ Preliminary information before doctor consultation  
✅ Demo of RAG + rule-based hybrid systems  
✅ Portable analysis of medical documents  

## 🎯 What This Is NOT Good For

❌ Replacing professional medical interpretation  
❌ Making health decisions without a doctor  
❌ Complex clinical decision support  
❌ Real-world medical applications  
❌ Precision medicine or personalized care  

## 🛠️ Architecture: RAG + Rule-Based Hybrid

**Why Hybrid?**

Pure RAG + LLM would be:
- More expensive (API calls)
- Prone to hallucinations
- Harder to explain/audit
- Medical liability risk

Rule-based extraction is:
- Deterministic and explainable
- Fast and reliable
- Privacy-preserving
- No API dependency

This hybrid approach balances:
- **Intelligent retrieval** (RAG finds relevant sections)
- **Reliable extraction** (rule-based parameter parsing)
- **Safe explanations** (predefined, validated knowledge)

## 🚀 If You Wanted Full LLM Integration

To convert this to a pure RAG + LLM system:

```python
# Instead of rule-based extraction:
parameter_json = {
    "Hemoglobin": {"value": "10.2", "unit": "g/dL", "reference": "13.0-17.0"}
}

# Send to LLM:
explanation = llm.generate(
    f"Explain this lab result: {parameter_json}. Context: {retrieved_chunks}"
)
```

This would enable:
- Natural language explanations
- Relationship detection between parameters
- Context-aware interpretation
- But at cost of hallucinations and API calls

## 📝 Future Improvements

- [ ] Support for more lab test types
- [ ] Better PDF layout handling
- [ ] Multi-language support
- [ ] Comparison with previous results
- [ ] Custom reference ranges per lab
- [ ] High-resolution charting
- [ ] Export to PDF report

## 📞 Troubleshooting

**App doesn't show results?**
- Verify PDF contains readable text (not scanned image)
- Check if medical parameters are in standard format
- Some PDFs may have unusual layouts

**"No parameters found"?**
- PDF may use different formatting than expected
- Parameters may use different naming conventions
- Try a different medical report

## 🔬 Technical Notes for Engineers

**Why FAISS for medical data?**
- Medical reports have varied structure and wording
- Semantic search finds relevant sections better than keyword matching
- 384-dim embeddings sufficient for most medical text

**Why regex for extraction?**
- Lab values follow predictable formats: "Parameter: VALUE UNIT"
- Regex is transparent and debuggable
- No black-box extraction failures
- Easy to add new patterns

**Why predefined knowledge?**
- Medical reference ranges are standardized (NCCLS, CLSI)
- Explanations should be validated medically
- Reduces hallucination risk
- Faster and deterministic

## 📚 Reference Implementation

This is a reference implementation showing:
- How to build hybrid intelligent systems
- RAG for intelligent document retrieval
- Rule-based systems for explainability
- Medical data handling (educational use case)

**Use this as:**
- Learning material for RAG + rule-based pipelines
- Prototype for document retrieval systems
- Educational medical report explorer
- Not as production medical software

---

**Built with ❤️ for learning and exploration**

**Version:** 2.0 (Hybrid RAG + Rule-Based System | FREE | Local Processing)
