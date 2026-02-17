# Medical Report Analyzer - RAG Backend

A comprehensive medical report analyzer that extracts lab test results and provides detailed educational explanations with health recommendations.

## ✨ Key Features

✅ **FREE Mode** - No API key required!  
✅ **Detailed Explanations** - Understand what each test means  
✅ **Health Consequences** - Learn what happens if values are abnormal  
✅ **Actionable Recommendations** - Specific steps to improve health  
✅ **Doctor Guidance** - Know when to consult healthcare professionals  
✅ **Well-Organized Output** - Test results grouped by category (CBC, Liver, Lipids, Blood Sugar)

## 🏗️ How It Works

The system uses a 6-step RAG pipeline to analyze medical reports:

### Step 1️⃣: Extract Text from PDF
- Reads uploaded PDF file  
- Extracts all text from each page  
- Cleans unnecessary characters and whitespace  
- **Output:** Raw medical report text

### Step 2️⃣: Split Text into Chunks
- Divides large medical reports into manageable chunks (500-800 characters)  
- Maintains overlap between chunks for context continuity  
- **Output:** List of text chunks

### Step 3️⃣: Create Embeddings
- Converts each chunk into vector representation using `SentenceTransformers`  
- Uses pre-trained model: `all-MiniLM-L6-v2` (384-dimensional embeddings)  
- Efficient and lightweight (free, no API needed)  
- **Output:** Vector embeddings for each chunk

### Step 4️⃣: Store in Vector Database
- Stores embeddings in FAISS (Facebook AI Similarity Search)  
- Enables fast similarity search over large datasets  
- **Output:** Indexed vector store for retrieval

### Step 5️⃣: Retrieve Relevant Chunks
- Query is converted to embedding  
- FAISS performs similarity search to find top-k relevant chunks  
- Only relevant context is used for analysis  
- **Output:** Top-5 relevant chunks

### Step 6️⃣: Extract & Explain Parameters
- Extracts specific medical parameters (Hemoglobin, RBC, Cholesterol, etc.)
- Compares values against reference ranges
- Generates comprehensive explanations with:
  - What each parameter means
  - Consequences of abnormal values
  - Specific health recommendations
  - When to see a doctor
- **Output:** Detailed educational breakdown

## 📁 File Structure

```
medical-rag/
├── app.py                 # Streamlit UI (main application)
├── rag_backend.py         # RAG pipeline implementation (core logic)
├── requirements.txt       # Python dependencies
├── .env.example          # Optional environment variables
└── README.md             # This file
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set OpenAI API Key

The app runs perfectly in **FREE mode** without an API key.

If you want to create a `.env` file for optional future features:
```bash
OPENAI_API_KEY=sk-your-optional-api-key
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 📊 How It Works (Visual Flow)

```
PDF Upload
    ↓
[Step 1] Extract Text → Raw Text
    ↓
[Step 2] Split into Chunks → Text Chunks
    ↓
[Step 3] Create Embeddings → 384-dim Vectors
    ↓
[Step 4] Store in FAISS → Vector Index
    ↓
[Step 5] Retrieve Relevant Chunks (Similarity Search)
    ↓
[Step 6] Extract Medical Parameters & Generate Explanations
    ↓
Organized Medical Breakdown + Doctor Recommendations
```

## 📚 Output Format

The app generates a comprehensive breakdown organized by test category:

### 🩸 BLOOD COUNT FINDINGS (CBC)
- Hemoglobin
- RBC Count
- WBC Count
- Platelet Count
- Hematocrit

### 🟡 LIVER FUNCTION TESTS (LFT)
- Total Bilirubin
- ALT (SGPT)
- AST (SGOT)
- Alkaline Phosphatase

### 💔 CHOLESTEROL & LIPID PROFILE
- Total Cholesterol
- LDL Cholesterol (Bad Cholesterol)
- HDL Cholesterol (Good Cholesterol)
- Triglycerides

### 🍬 BLOOD SUGAR (DIABETES SCREENING)
- Fasting Blood Sugar
- HbA1c

## 🔍 For Each Parameter, You Get:

✅ **Status Indicator**
- ✅ NORMAL - Green checkmark
- ❌ HIGH - Red X
- ❌ LOW - Red X

📋 **Your Results**
- Your exact value
- Normal reference range

📖 **Educational Information**
- What the parameter is
- What high values mean (consequences)
- What low values mean (consequences)
- Specific health recommendations
- When to see a doctor

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Extraction** | PyPDF2 | Read and extract text from PDFs |
| **Text Embeddings** | SentenceTransformers | Convert text to vectors (FREE) |
| **Vector Database** | FAISS | Fast similarity search |
| **Framework** | Streamlit | Interactive web interface |
| **Environment** | Python 3.10+ | Cross-platform support |

## 🔧 Configuration Options

You can customize the analysis in the Streamlit sidebar:

- **Number of retrieved chunks** (1-10): Controls how much context is used
- **Chunk size** (300-1000 chars): Affects granularity of document splitting  
- **Chunk overlap** (0-200 chars): Ensures context continuity between chunks
- **Custom explanation focus** (optional): Ask the system to focus on specific aspects

## 📊 Supported Medical Parameters (15+)

### Blood Count (CBC)
- Hemoglobin, RBC Count, WBC Count, Platelet Count, Hematocrit

### Liver Function (LFT)
- Total Bilirubin, ALT (SGPT), AST (SGOT), Alkaline Phosphatase

### Cholesterol & Lipids
- Total Cholesterol, LDL Cholesterol, HDL Cholesterol, Triglycerides

### Blood Sugar
- Fasting Blood Sugar, HbA1c

## 💡 Example: How It Helps

**Your Medical Report Shows:** Hemoglobin 10.2 g/dL (Low)

**The App Explains:**
```
What is Hemoglobin?
- Protein in red blood cells that carries oxygen throughout your body

⚠️ Consequences of LOW Hemoglobin:
- Low hemoglobin means your blood cannot carry enough oxygen, 
  leading to fatigue, weakness, shortness of breath, and dizziness

💊 What to do:
- Eat iron-rich foods (spinach, red meat, beans)
- Increase vitamin C intake
- Rest more
- Consult your doctor for blood work
```

## 🏥 Doctor Recommendations

The app guides you on when to seek medical care:

**See Doctor IMMEDIATELY if:**
- Very high LDL cholesterol (> 190)
- High blood sugar with symptoms
- High bilirubin with jaundice
- Low platelet count (< 50,000)
- Abnormal liver function tests

**Schedule an Appointment if:**
- Any values out of normal range
- Multiple abnormal results
- Symptoms like fatigue, chest pain, or shortness of breath

## ⚠️ Important Disclaimer

This tool is **for educational purposes only** and is **NOT** a medical diagnosis or medical advice.

- This is a **comparison** of laboratory values against reference ranges
- It does **NOT** provide a diagnosis
- It does **NOT** replace professional medical consultation
- Always **consult qualified healthcare professionals** for:
  - Interpretation of your specific results
  - Medical diagnosis
  - Treatment recommendations
  - Personalized health guidance

**Your doctor should evaluate these results in the context of your complete medical history, symptoms, and other clinical findings.**

## 📈 Performance

- **PDF Extraction:** <1 second
- **Chunking & Embeddings:** 2-5 seconds
- **Vector Database Creation:** <1 second
- **Parameter Extraction:** <1 second
- **Total Processing Time:** ~5-10 seconds per report

## 🚨 Error Handling

The system gracefully handles:
- ❌ Empty or corrupted PDFs
- ❌ PDFs without readable text
- ❌ Unusual medical report formats
- ❌ Missing or incomplete test results

## 📝 Future Enhancements

- [ ] Support for multiple file formats (DOCX, images)
- [ ] Multi-language support
- [ ] Comparison with previous test results
- [ ] Risk assessment based on multiple parameters
- [ ] Custom medical knowledge base
- [ ] Integration with healthcare APIs
- [ ] User authentication and test history

## 🔐 Privacy

- PDF files are processed locally
- No data is stored on servers
- No files are sent to external services (in FREE mode)
- Your results are only shown in your browser session

## 📞 Troubleshooting

**App doesn't load?**
- Ensure all dependencies: `pip install -r requirements.txt`
- Check Python version: Python 3.10 or higher

**PDF not being processed?**
- Verify PDF contains readable text (not scanned image)
- Try a different PDF to test

**Results show "No data found"?**
- PDF format may be unusual
- Try uploading a different medical report
- Check if the report contains the medical parameters

---

**Made with ❤️ for medical accessibility**

**Version:** 2.0 (FREE Mode - No API Key Required!)
