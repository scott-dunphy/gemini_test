import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO
import time
import os
import tempfile

# PDF Processing
from pdfminer.high_level import extract_text as extract_text_miner
from pdfminer.layout import LAParams
import pytesseract
from PIL import Image
# import PyPDF2 # Alternative

# AI Integration - CHANGED
import openai

# Environment variables
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# --- Configuration ---
# Use st.secrets for deployment, otherwise use environment variables or direct input
# Configure OpenAI - CHANGED
try:
    # Try getting key from Streamlit secrets first (for deployment)
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback to environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file):
    """Extracts text from PDF using pdfminer.six, falling back to OCR if needed."""
    extracted_text = ""
    try:
        file_bytes = BytesIO(uploaded_file.getvalue())
        extracted_text = extract_text_miner(file_bytes, laparams=LAParams())
        st.write("Extracted text using pdfminer.six.")
    except Exception as e:
        st.warning(f"pdfminer.six failed: {e}. Attempting OCR...")
        extracted_text = ""

    if len(extracted_text.strip()) < 100: # Arbitrary threshold
        st.warning("Extracted text seems short, attempting OCR.")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_pdf.name

            st.write("Running OCR (this might take a moment)...")
            # You might need to install poppler-utils for pdf to image conversion
            # depending on your pytesseract setup and PDF type.
            extracted_text_ocr = pytesseract.image_to_string(tmp_pdf_path)
            if len(extracted_text_ocr.strip()) > len(extracted_text.strip()):
                 st.write("OCR produced more text.")
                 extracted_text = extracted_text_ocr
            else:
                 st.write("Keeping original extracted text (or OCR was empty).")

            os.unlink(tmp_pdf_path) # Clean up

        except Exception as ocr_e:
            st.error(f"OCR process failed: {ocr_e}")
            if not extracted_text:
                 return None

    if not extracted_text.strip():
         st.error("Could not extract significant text from the PDF.")
         return None

    return extracted_text

# --- OpenAI Specific Functions --- CHANGED ---

def configure_openai(api_key):
    """Configures the OpenAI client."""
    if not api_key:
        st.error("OpenAI API key is missing. Please add it via the sidebar or .env file.")
        return False
    try:
        openai.api_key = api_key
        # Optional: Test connectivity (lightweight call)
        # openai.models.list()
        st.sidebar.success("OpenAI API key configured.")
        return True
    except Exception as e:
        st.error(f"Error configuring OpenAI API: {e}")
        return False

def get_openai_response(prompt, model_name="gpt-4o", is_json=False, temperature=0.3):
    """Sends prompt to OpenAI ChatCompletion API and gets response."""
    messages = [{"role": "user", "content": prompt}]
    try:
        if is_json:
             # Instruct the model to produce JSON and use JSON mode if available
             messages = [
                 {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                 {"role": "user", "content": prompt + "\n\nPlease ensure your entire response is only the valid JSON object or array requested, enclosed in curly braces {} or square brackets [], with no other text before or after."}
             ]
             # Check if model supports JSON mode (gpt-4o, gpt-4-turbo*, gpt-3.5-turbo-0125+)
             json_mode_supported = any(m in model_name for m in ["gpt-4o", "turbo"])
             if json_mode_supported:
                 response = openai.chat.completions.create(
                     model=model_name,
                     messages=messages,
                     response_format={"type": "json_object"},
                     temperature=temperature # Lower temp for stricter JSON output
                 )
             else: # Fallback for models without explicit JSON mode
                 response = openai.chat.completions.create(
                     model=model_name,
                     messages=messages,
                     temperature=temperature
                 )

        else: # Standard text generation
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature
            )

        response_text = response.choices[0].message.content.strip()

        # Manual cleanup as fallback if JSON mode wasn't used or failed
        if is_json and not json_mode_supported:
             # Find first '{' or '[' and last '}' or ']'
            start = -1
            first_brace = response_text.find('{')
            first_bracket = response_text.find('[')
            if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                start = first_brace
            elif first_bracket != -1:
                start = first_bracket

            end = -1
            last_brace = response_text.rfind('}')
            last_bracket = response_text.rfind(']')
            if last_brace != -1 and (last_bracket == -1 or last_brace > last_bracket):
                end = last_brace + 1
            elif last_bracket != -1:
                end = last_bracket + 1

            if start != -1 and end != -1:
                response_text = response_text[start:end]
            else: # Basic ``` cleanup if boundaries are unclear
                 response_text = response_text.replace('```json', '').replace('```', '').strip()

        return response_text

    except openai.AuthenticationError:
        st.error("OpenAI Authentication Error: Invalid API Key. Please check your key in the sidebar or .env file.")
        return None
    except openai.RateLimitError:
        st.error("OpenAI Rate Limit Error: You've exceeded your usage limits. Please check your OpenAI account or wait a moment.")
        return None
    except openai.APIConnectionError as e:
        st.error(f"OpenAI Connection Error: Could not connect to OpenAI. {e}")
        return None
    except Exception as e:
        # Catch other potential OpenAI errors or general errors
        st.error(f"OpenAI API call failed: {e}")
        return None

# --- End of OpenAI Specific Functions ---

def validate_and_load_json(json_string):
    """Tries to load a JSON string, returns None on failure."""
    if not json_string:
        return None
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response as JSON: {e}")
        st.text_area("Invalid JSON received:", json_string, height=150)
        return None

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="PDF Data Extractor & Analyzer")
st.title("📄 Agentic PDF Data Extractor & Analyzer (using OpenAI)")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input - CHANGED
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        value=OPENAI_API_KEY or "",
        help="Get your key from OpenAI Platform. Uses OPENAI_API_KEY env var if set."
    )

    # Instantiate OpenAI client
    openai_configured = configure_openai(api_key_input)

    st.header("Inputs")
    uploaded_pdf = st.file_uploader("1. Upload PDF Document", type="pdf")
    uploaded_schema = st.file_uploader("2. Upload JSON Schema", type="json")

    # Choose Model - Added
    openai_model = st.selectbox(
        "Select OpenAI Model",
        ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"), # Add more models if needed
        index=0 # Default to gpt-4o
    )

    process_button = st.button(
        "Process PDF",
        type="primary",
        disabled=not (uploaded_pdf and uploaded_schema and api_key_input and openai_configured)
        )

    # Placeholder for download button
    download_placeholder = st.empty()

# --- Main Area for Processing and Results ---

# Initialize session state variables
if 'pdf_text' not in st.session_state: st.session_state.pdf_text = None
if 'extracted_json_str' not in st.session_state: st.session_state.extracted_json_str = None
if 'extracted_data' not in st.session_state: st.session_state.extracted_data = None
if 'verification_feedback' not in st.session_state: st.session_state.verification_feedback = None
if 'dataframe' not in st.session_state: st.session_state.dataframe = None
if 'analysis_requested' not in st.session_state: st.session_state.analysis_requested = False
if 'analysis_confirmed' not in st.session_state: st.session_state.analysis_confirmed = False
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None


# --- Processing Logic ---
if process_button:
    # Reset state for reprocessing
    st.session_state.pdf_text = None
    st.session_state.extracted_json_str = None
    st.session_state.extracted_data = None
    st.session_state.verification_feedback = None
    st.session_state.dataframe = None
    st.session_state.analysis_requested = False
    st.session_state.analysis_confirmed = False
    st.session_state.analysis_result = None

    # 1. Check AI Configuration (already done in sidebar, re-check just in case)
    if not openai_configured:
        st.error("OpenAI API Key not configured correctly. Please check the sidebar.")
        st.stop()

    # 2. Load Schema
    schema_string = StringIO(uploaded_schema.getvalue().decode("utf-8")).read()
    try:
        schema_dict = json.loads(schema_string)
        # st.sidebar.success("JSON schema loaded.") # Already implicitly checked by button enable
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON schema file: {e}")
        st.stop()

    # 3. Extract Text from PDF
    with st.spinner("Extracting text from PDF... (OCR might take longer)"):
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        if pdf_text:
            st.session_state.pdf_text = pdf_text
            st.success("✅ Text extracted from PDF.")
        else:
            st.error("❌ Failed to extract text from PDF.")
            st.stop()

    # 4. AI Agent: Structured Data Extraction (using OpenAI) - CHANGED
    if st.session_state.pdf_text:
        with st.spinner(f"🤖 AI Agent 1 ({openai_model}): Extracting structured data..."):
            # Truncate text if very long to avoid exceeding token limits
            max_chars = 16000 # Adjust based on model context window and expected schema size
            prompt_extract = f"""
            You are an AI assistant tasked with extracting structured data from the following text based on the provided JSON schema.

            **JSON Schema:**
            ```json
            {json.dumps(schema_dict, indent=2)}
            ```

            **Source Text (first {max_chars} characters):**
            ```text
            {st.session_state.pdf_text[:max_chars]}
            ```
            **(Note: Text might be truncated)**

            **Instructions:**
            1. Read the Source Text carefully.
            2. Identify information corresponding to the fields in the JSON Schema.
            3. Format the extracted information *strictly* according to the JSON Schema.
            4. If a field's value isn't found, use `null` or omit the field as appropriate for the schema.
            5. Ensure the output is a *single, valid JSON object* or a *valid array of JSON objects* if the schema suggests multiple entries.
            6. **Crucially: Your response must contain *only* the JSON data itself, with no surrounding text, comments, or explanations.**
            """
            # Using lower temperature for more deterministic JSON output
            extracted_json_str = get_openai_response(
                prompt_extract, model_name=openai_model, is_json=True, temperature=0.1
            )

            if extracted_json_str:
                st.session_state.extracted_json_str = extracted_json_str
                st.session_state.extracted_data = validate_and_load_json(extracted_json_str)
                if st.session_state.extracted_data is not None: # Check for successful parsing
                     st.success(f"✅ AI Agent 1 ({openai_model}): Structured data extracted.")
                else:
                     # Error message already shown by validate_and_load_json
                     st.error(f"❌ AI Agent 1 ({openai_model}): Failed to get valid JSON response.")
            else:
                st.error(f"❌ AI Agent 1 ({openai_model}): No response or error during extraction.")


    # 5. AI Agent: Verification (using OpenAI) - CHANGED
    if st.session_state.extracted_data: # Check if extraction was successful
        with st.spinner(f"🕵️ AI Agent 2 ({openai_model}): Verifying extracted data..."):
            max_chars_verify = 8000 # Use less context maybe for verification
            prompt_verify = f"""
            You are an AI verification agent. Review the JSON data supposedly extracted from a source text and check its accuracy against that text.

            **Source Text (first {max_chars_verify} characters):**
            ```text
            {st.session_state.pdf_text[:max_chars_verify]}
            ```
            **(Note: Text might be truncated)**

            **Extracted JSON Data:**
            ```json
            {st.session_state.extracted_json_str}
            ```

            **Instructions:**
            1. Compare the values in the Extracted JSON Data with the Source Text.
            2. Identify discrepancies, inaccuracies, or potentially missing information in the JSON based *only* on the provided Source Text.
            3. Provide a brief summary of your findings (e.g., bullet points). Mention specific fields if they seem incorrect or questionable.
            4. If everything looks accurate according to the text, state that clearly.
            5. Do NOT output JSON. Provide feedback as plain text.
            """
            # Slightly higher temperature for more nuanced feedback
            verification_feedback = get_openai_response(
                prompt_verify, model_name=openai_model, is_json=False, temperature=0.4
            )
            if verification_feedback:
                st.session_state.verification_feedback = verification_feedback
                st.success(f"✅ AI Agent 2 ({openai_model}): Verification complete.")
            else:
                st.warning(f"⚠️ AI Agent 2 ({openai_model}): Could not get verification feedback or error occurred.")


# --- Display Results --- (No changes needed in this section)

# Display Verification Feedback First
if st.session_state.verification_feedback:
    st.subheader("🕵️ Verification Agent Feedback")
    st.markdown(st.session_state.verification_feedback)
    st.divider()

# Display Extracted Data and Convert to DataFrame
if st.session_state.extracted_data is not None:
    st.subheader("📊 Extracted Data (JSON)")
    st.json(st.session_state.extracted_data)

    try:
        data_for_df = st.session_state.extracted_data
        if isinstance(data_for_df, dict):
            data_for_df = [data_for_df]

        if isinstance(data_for_df, list) and (not data_for_df or all(isinstance(item, dict) for item in data_for_df)):
             # Handle empty list case gracefully
             if not data_for_df:
                 st.info("Extracted data is an empty list.")
                 st.session_state.dataframe = pd.DataFrame() # Create empty DataFrame
             else:
                 df = pd.DataFrame(data_for_df)
                 st.session_state.dataframe = df
                 st.subheader("📋 DataFrame Preview")
                 st.dataframe(df)

             # Prepare CSV for download (works for empty df too)
             csv = st.session_state.dataframe.to_csv(index=False).encode('utf-8')
             with download_placeholder:
                  st.download_button(
                       label="📥 Download DataFrame as CSV",
                       data=csv,
                       file_name=f"{uploaded_pdf.name.split('.')[0]}_extracted_data.csv" if uploaded_pdf else "extracted_data.csv",
                       mime="text/csv",
                  )
        else:
             st.warning("Extracted data is not in a format suitable for a standard DataFrame (list of objects expected). Cannot create DataFrame.")
             st.session_state.dataframe = None

    except Exception as e:
        st.error(f"Error converting JSON to DataFrame: {e}")
        st.session_state.dataframe = None # Ensure dataframe is None on error

# --- Optional Analysis Section --- (Only changed Agent 3 call)
if st.session_state.dataframe is not None: # Check if DataFrame exists
    st.divider()
    st.subheader("🧠 Data Analysis")

    analysis_query = st.text_area(
        "Ask a question about the extracted data:",
        key="analysis_query_input",
        # Use callback to set analysis_requested flag when input changes
        on_change=lambda: setattr(st.session_state, 'analysis_requested', bool(st.session_state.analysis_query_input))
    )

    # Only show confirmation if analysis is requested AND not yet confirmed
    if st.session_state.analysis_requested and not st.session_state.analysis_confirmed:
        st.warning("**Review the extracted DataFrame above before proceeding.**")
        confirm_analysis = st.button("Yes, Analyze This Data")
        if confirm_analysis:
            st.session_state.analysis_confirmed = True
            # Clear previous analysis result if re-confirming
            st.session_state.analysis_result = None
            st.rerun() # Rerun to proceed to analysis step

    # Perform analysis only if confirmed
    if st.session_state.analysis_confirmed and st.session_state.analysis_result is None: # Prevent re-running analysis on cosmetic reruns
        with st.spinner(f"💭 AI Agent 3 ({openai_model}): Analyzing data..."):
            if not st.session_state.dataframe.empty:
                 # Convert DataFrame to string format suitable for LLM
                 max_rows_analysis = 100 # Limit rows sent for analysis
                 df_string = st.session_state.dataframe.head(max_rows_analysis).to_csv(index=False)
                 row_info = f"first {len(st.session_state.dataframe.head(max_rows_analysis))} rows" if len(st.session_state.dataframe) > max_rows_analysis else f"all {len(st.session_state.dataframe)} rows"

                 prompt_analyze = f"""
                 You are an AI data analyst. Analyze the following data based on the user's question.

                 **Data ({row_info}, CSV format):**
                 ```csv
                 {df_string}
                 ```

                 **User's Question:**
                 {analysis_query}

                 **Instructions:**
                 1. Understand the user's question.
                 2. Analyze the provided data to answer the question.
                 3. Provide a clear and concise answer based *only* on the data given.
                 4. If the data is insufficient or irrelevant to the question, state that clearly.
                 5. Present the analysis in a readable format (e.g., text, bullet points).
                 """
                 # Use a potentially higher temperature for more creative/insightful analysis
                 analysis_result = get_openai_response(
                     prompt_analyze, model_name=openai_model, is_json=False, temperature=0.5
                 )
                 if analysis_result:
                     st.session_state.analysis_result = analysis_result
                     st.success(f"✅ AI Agent 3 ({openai_model}): Analysis complete.")
                 else:
                     st.error(f"❌ AI Agent 3 ({openai_model}): Failed to get analysis result or error occurred.")
                     # Reset confirmation on failure to allow potential retry after fixing issue
                     # st.session_state.analysis_confirmed = False
            else:
                st.warning("Cannot perform analysis, the DataFrame is empty.")
                # Reset confirmation if data is empty
                st.session_state.analysis_confirmed = False

    # Display analysis result if available
    if st.session_state.analysis_result:
         st.subheader("📈 Analysis Results")
         st.markdown(st.session_state.analysis_result)


# Display Original Extracted Text (Optional)
if st.session_state.pdf_text:
    with st.expander("View Extracted Raw Text"):
        st.text(st.session_state.pdf_text[:5000] + "..." if len(st.session_state.pdf_text) > 5000 else st.session_state.pdf_text)
