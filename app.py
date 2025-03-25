import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO # StringIO might not be needed anymore for schema
import time
import os
import tempfile

# PDF Processing
from pdfminer.high_level import extract_text as extract_text_miner
from pdfminer.layout import LAParams
import pytesseract
from PIL import Image

# AI Integration
import openai

# Environment variables
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Helper Functions ---
# (extract_text_from_pdf, configure_openai, get_openai_response, validate_and_load_json remain the same)
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

def configure_openai(api_key):
    """Configures the OpenAI client."""
    if not api_key:
        # Error display handled in the main UI logic where it's called now
        return False
    try:
        openai.api_key = api_key
        return True
    except Exception as e:
        st.error(f"Error configuring OpenAI API: {e}")
        return False

def get_openai_response(prompt, model_name="gpt-4o", is_json=False, temperature=0.3):
    """Sends prompt to OpenAI ChatCompletion API and gets response."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response_args = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }
        if is_json:
             messages = [
                 {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                 {"role": "user", "content": prompt + "\n\nPlease ensure your entire response is only the valid JSON object or array requested, enclosed in curly braces {} or square brackets [], with no other text before or after."}
             ]
             response_args["messages"] = messages
             # Check if model supports JSON mode
             json_mode_supported = any(m in model_name for m in ["gpt-4o", "turbo"])
             if json_mode_supported:
                 response_args["response_format"] = {"type": "json_object"}

        response = openai.chat.completions.create(**response_args)
        response_text = response.choices[0].message.content.strip()

        # Manual cleanup as fallback if JSON mode wasn't used or failed
        if is_json and not response_args.get("response_format"):
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
            else:
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
        st.error(f"OpenAI API call failed: {e}")
        return None

def validate_and_load_json(json_string):
    """Tries to load a JSON string, returns None on failure."""
    if not json_string:
        return None
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Error message displayed where this function is called
        # st.error(f"Failed to parse AI response as JSON: {e}")
        # st.text_area("Invalid JSON received:", json_string, height=150)
        return None


# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="PDF Data Extractor & Analyzer")
st.title("📄 Agentic PDF Data Extractor & Analyzer (using OpenAI)")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        value=OPENAI_API_KEY or "",
        help="Get your key from OpenAI Platform. Uses OPENAI_API_KEY env var if set."
    )

    # Check API Key and Configure OpenAI
    openai_configured = False
    if api_key_input:
        openai_configured = configure_openai(api_key_input)
        if not openai_configured:
            st.sidebar.error("Invalid or missing OpenAI API Key.")
    else:
        st.sidebar.warning("Please enter your OpenAI API Key.")


    st.header("Inputs")
    uploaded_pdf = st.file_uploader("1. Upload PDF Document", type="pdf")

    # --- CHANGED: Schema Input from Text Area ---
    schema_input_str = st.text_area(
        "2. Paste JSON Schema Here",
        height=250, # Adjust height as needed
        placeholder='{\n  "field_name_1": "description or type",\n  "field_name_2": "description or type",\n  "nested_object": {\n    "sub_field": "type"\n  },\n  "list_field": ["item_type"]\n}',
        help="Paste the JSON structure you want to extract data into."
    )

    # Attempt to parse schema immediately to validate syntax for button state
    schema_dict = None
    is_schema_valid = False
    if schema_input_str:
        try:
            schema_dict = json.loads(schema_input_str)
            is_schema_valid = True
            # You could optionally add a small success indicator here if desired
            # st.sidebar.caption("✔️ Schema Syntax OK")
        except json.JSONDecodeError as e:
            st.sidebar.error(f"Invalid JSON Schema Syntax: {e}")
            is_schema_valid = False # Ensure flag is False on error
            schema_dict = None # Ensure dict is None on error
    # --- END OF SCHEMA CHANGE ---

    # Choose Model
    openai_model = st.selectbox(
        "Select OpenAI Model",
        ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"),
        index=0
    )

    # --- CHANGED: Button Disabled Logic ---
    # Enable button only if all inputs and configurations are valid
    all_inputs_ready = (
        uploaded_pdf is not None and
        is_schema_valid and # Check if schema JSON is syntactically valid
        api_key_input and # Check if key is entered (configuration check happens above)
        openai_configured # Check if configuration was successful
    )
    process_button = st.button(
        "Process PDF",
        type="primary",
        disabled=not all_inputs_ready
    )

    # Placeholder for download button
    download_placeholder = st.empty()

# --- Main Area for Processing and Results ---

# Initialize session state variables (if not already done)
# (Session state initialization code remains the same)
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

    # 1. Configuration checks (already performed for button state, maybe redundant but safe)
    if not openai_configured:
        st.error("OpenAI API Key not configured correctly. Please check the sidebar.")
        st.stop()
    if not is_schema_valid or not schema_dict:
        st.error("JSON Schema is invalid or missing. Please check the sidebar.")
        st.stop() # schema_dict should be valid here because the button required it

    # 2. Schema is already loaded into schema_dict from the sidebar logic

    # 3. Extract Text from PDF
    with st.spinner("Extracting text from PDF... (OCR might take longer)"):
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        if pdf_text:
            st.session_state.pdf_text = pdf_text
            st.success("✅ Text extracted from PDF.")
        else:
            st.error("❌ Failed to extract text from PDF.")
            st.stop()

    # 4. AI Agent: Structured Data Extraction
    if st.session_state.pdf_text:
        with st.spinner(f"🤖 AI Agent 1 ({openai_model}): Extracting structured data..."):
            max_chars = 16000
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
            extracted_json_str = get_openai_response(
                prompt_extract, model_name=openai_model, is_json=True, temperature=0.1
            )

            if extracted_json_str:
                st.session_state.extracted_json_str = extracted_json_str
                # Validate the JSON received from the AI
                temp_extracted_data = validate_and_load_json(extracted_json_str)
                if temp_extracted_data is not None:
                     st.session_state.extracted_data = temp_extracted_data
                     st.success(f"✅ AI Agent 1 ({openai_model}): Structured data extracted and parsed.")
                else:
                     # Validation failed, show error and the raw response
                     st.error(f"❌ AI Agent 1 ({openai_model}): Failed to parse AI response as valid JSON.")
                     st.text_area("Invalid JSON received:", extracted_json_str, height=150)
            else:
                st.error(f"❌ AI Agent 1 ({openai_model}): No response or error during extraction.")

    # 5. AI Agent: Verification
    # (Verification logic remains the same, uses st.session_state.extracted_data if valid)
    if st.session_state.extracted_data:
         # ... (verification prompt and call) ...
         with st.spinner(f"🕵️ AI Agent 2 ({openai_model}): Verifying extracted data..."):
            max_chars_verify = 8000
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
            verification_feedback = get_openai_response(
                prompt_verify, model_name=openai_model, is_json=False, temperature=0.4
            )
            if verification_feedback:
                st.session_state.verification_feedback = verification_feedback
                st.success(f"✅ AI Agent 2 ({openai_model}): Verification complete.")
            else:
                st.warning(f"⚠️ AI Agent 2 ({openai_model}): Could not get verification feedback or error occurred.")


# --- Display Results ---
# (Display logic remains the same)
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
             if not data_for_df:
                 st.info("Extracted data is an empty list.")
                 st.session_state.dataframe = pd.DataFrame()
             else:
                 df = pd.DataFrame(data_for_df)
                 st.session_state.dataframe = df
                 st.subheader("📋 DataFrame Preview")
                 st.dataframe(df)

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
        st.session_state.dataframe = None

# --- Optional Analysis Section ---
# (Analysis logic remains the same)
if st.session_state.dataframe is not None:
    st.divider()
    st.subheader("🧠 Data Analysis")

    analysis_query = st.text_area(
        "Ask a question about the extracted data:",
        key="analysis_query_input",
        on_change=lambda: setattr(st.session_state, 'analysis_requested', bool(st.session_state.analysis_query_input))
    )

    if st.session_state.analysis_requested and not st.session_state.analysis_confirmed:
        st.warning("**Review the extracted DataFrame above before proceeding.**")
        confirm_analysis = st.button("Yes, Analyze This Data")
        if confirm_analysis:
            st.session_state.analysis_confirmed = True
            st.session_state.analysis_result = None
            st.rerun()

    if st.session_state.analysis_confirmed and st.session_state.analysis_result is None:
        with st.spinner(f"💭 AI Agent 3 ({openai_model}): Analyzing data..."):
            if not st.session_state.dataframe.empty:
                 max_rows_analysis = 100
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
                 analysis_result = get_openai_response(
                     prompt_analyze, model_name=openai_model, is_json=False, temperature=0.5
                 )
                 if analysis_result:
                     st.session_state.analysis_result = analysis_result
                     st.success(f"✅ AI Agent 3 ({openai_model}): Analysis complete.")
                 else:
                     st.error(f"❌ AI Agent 3 ({openai_model}): Failed to get analysis result or error occurred.")
            else:
                st.warning("Cannot perform analysis, the DataFrame is empty.")
                st.session_state.analysis_confirmed = False # Reset confirmation

    if st.session_state.analysis_result:
         st.subheader("📈 Analysis Results")
         st.markdown(st.session_state.analysis_result)


# Display Original Extracted Text (Optional)
if st.session_state.pdf_text:
    with st.expander("View Extracted Raw Text"):
        st.text(st.session_state.pdf_text[:5000] + "..." if len(st.session_state.pdf_text) > 5000 else st.session_state.pdf_text)
