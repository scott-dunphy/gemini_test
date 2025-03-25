import streamlit as st
import pandas as pd
import json
# StringIO might not be needed for schema anymore
from io import BytesIO
import time
import os
import tempfile

# PDF Processing
from pdfminer.high_level import extract_text as extract_text_miner
from pdfminer.layout import LAParams
import pytesseract
from PIL import Image

# AI Integration - CHANGED for Mistral AI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralAPIException, MistralConnectionException, MistralAuthenticationError


# Environment variables
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# --- Configuration ---
# Use st.secrets for deployment, otherwise use environment variables or direct input
# Configure Mistral AI - CHANGED
try:
    # Try getting key from Streamlit secrets first (for deployment)
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback to environment variable
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- Helper Functions ---
# (extract_text_from_pdf remains the same)
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

# --- Mistral AI Specific Functions --- CHANGED ---

# Cache the Mistral client to avoid re-initializing on every interaction
@st.cache_resource
def configure_mistral(api_key):
    """Configures and returns the Mistral client."""
    if not api_key:
        return None # Error handled where called
    try:
        client = MistralClient(api_key=api_key)
        # Optional: Lightweight check to verify connection/key
        # client.list_models()
        return client
    except MistralAuthenticationError:
         # Error handled where called, but could log here
         return None
    except Exception as e:
        st.error(f"Error initializing Mistral client: {e}") # Show general init errors
        return None

def get_mistral_response(client, prompt, model_name="mistral-large-latest", is_json=False, temperature=0.3):
    """Sends prompt to Mistral AI chat API and gets response."""
    if not client:
         st.error("Mistral client not configured.")
         return None

    messages = [ChatMessage(role="user", content=prompt)]
    response_format_arg = None # Default to text format

    try:
        # Prepare arguments for client.chat
        chat_args = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if is_json:
            # Add system message and potentially request JSON format
            messages = [
                 ChatMessage(role="system", content="You are a helpful assistant designed to output JSON."),
                 ChatMessage(role="user", content=prompt + "\n\nPlease ensure your entire response is only the valid JSON object or array requested, enclosed in curly braces {} or square brackets [], with no other text before or after.")
             ]
            chat_args["messages"] = messages # Update messages in args

            # Check if model likely supports JSON mode (Mistral Large 2 does)
            # Consult Mistral documentation for definitive list
            if "large" in model_name or "medium" in model_name:
                response_format_arg = {"type": "json_object"}
                chat_args["response_format"] = response_format_arg

        # Make the API call
        response = client.chat(**chat_args)

        response_text = response.choices[0].message.content.strip()

        # Manual cleanup as fallback if JSON mode wasn't used or failed
        # (This is less likely needed if JSON mode is used correctly, but kept as safeguard)
        if is_json and not response_format_arg:
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

    except MistralAuthenticationError:
        st.error("Mistral Authentication Error: Invalid API Key. Please check your key in the sidebar or .env file.")
        return None
    except MistralAPIException as e:
         # Catch specific Mistral API errors (includes rate limits, bad requests, etc.)
         error_detail = str(e).lower()
         if "rate limit" in error_detail:
              st.error(f"Mistral Rate Limit Error: {e}. Please check your usage limits or wait a moment.")
         elif "quota" in error_detail or "credits" in error_detail:
              st.error(f"Mistral Quota/Billing Error: {e}. Please check your account balance/quota.")
         elif "invalid api key" in error_detail: # Double check auth error
              st.error(f"Mistral Authentication Error: {e}. Please check your API key.")
         else:
              st.error(f"Mistral API Error: {e}")
         return None
    except MistralConnectionException as e:
        st.error(f"Mistral Connection Error: Could not connect. {e}")
        return None
    except Exception as e:
        # Catch other potential errors during the call
        st.error(f"Mistral API call failed unexpectedly: {e}")
        return None

# (validate_and_load_json remains the same)
def validate_and_load_json(json_string):
    """Tries to load a JSON string, returns None on failure."""
    if not json_string:
        return None
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Error message displayed where this function is called
        return None

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="PDF Data Extractor & Analyzer")
st.title("📄 Agentic PDF Data Extractor & Analyzer (using Mistral AI)") # Changed Title

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input - CHANGED for Mistral
    api_key_input = st.text_input(
        "Enter your Mistral API Key:",
        type="password",
        value=MISTRAL_API_KEY or "",
        help="Get your key from console.mistral.ai. Uses MISTRAL_API_KEY env var if set."
    )

    # Instantiate Mistral client - CHANGED
    mistral_client = None
    if api_key_input:
        mistral_client = configure_mistral(api_key_input)
        if mistral_client:
            st.sidebar.success("Mistral client configured.")
        else:
             # configure_mistral might show specific errors, or we show a generic one
             if not any(isinstance(e, MistralAuthenticationError) for e in st.exception_container): # Avoid double printing auth errors
                 st.sidebar.error("Failed to configure Mistral client. Check key?")
    else:
        st.sidebar.warning("Please enter your Mistral API Key.")


    st.header("Inputs")
    uploaded_pdf = st.file_uploader("1. Upload PDF Document", type="pdf")

    # Schema Input from Text Area (remains the same)
    schema_input_str = st.text_area(
        "2. Paste JSON Schema Here",
        height=250,
        placeholder='{\n  "field_name_1": "description or type",\n ... }',
        help="Paste the JSON structure you want to extract data into."
    )

    # Attempt to parse schema immediately for button state
    schema_dict = None
    is_schema_valid = False
    if schema_input_str:
        try:
            schema_dict = json.loads(schema_input_str)
            is_schema_valid = True
        except json.JSONDecodeError as e:
            st.sidebar.error(f"Invalid JSON Schema Syntax: {e}")
            is_schema_valid = False
            schema_dict = None

    # Choose Model - CHANGED for Mistral
    # Clarification about model names might be useful here or in help text
    mistral_model = st.selectbox(
        "Select Mistral Model",
        (
            "mistral-large-latest",
            "open-mixtral-8x22b", # Large open model
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mixtral-8x7b", # Common Mixtral model
            # Add other models if needed, e.g., specific versions
        ),
        index=0 # Default to mistral-large-latest
    )

    # Button Disabled Logic - CHANGED for Mistral
    all_inputs_ready = (
        uploaded_pdf is not None and
        is_schema_valid and
        api_key_input and
        mistral_client is not None # Check if client was successfully created
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
# ... other state variables ...
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None


# --- Processing Logic ---
if process_button:
    # Reset state for reprocessing
    # (State reset code remains the same)
    st.session_state.pdf_text = None
    st.session_state.extracted_json_str = None
    st.session_state.extracted_data = None
    st.session_state.verification_feedback = None
    st.session_state.dataframe = None
    st.session_state.analysis_requested = False
    st.session_state.analysis_confirmed = False
    st.session_state.analysis_result = None

    # 1. Configuration checks (already performed for button state, added safety checks)
    if not mistral_client:
        st.error("Mistral client not configured correctly. Please check API key in sidebar.")
        st.stop()
    if not is_schema_valid or not schema_dict:
        st.error("JSON Schema is invalid or missing. Please check the sidebar.")
        st.stop()

    # 2. Schema is already loaded into schema_dict

    # 3. Extract Text from PDF
    # (Text extraction logic remains the same)
    with st.spinner("Extracting text from PDF... (OCR might take longer)"):
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        if pdf_text:
            st.session_state.pdf_text = pdf_text
            st.success("✅ Text extracted from PDF.")
        else:
            st.error("❌ Failed to extract text from PDF.")
            st.stop()


    # 4. AI Agent: Structured Data Extraction (using Mistral) - CHANGED
    if st.session_state.pdf_text:
        with st.spinner(f"🤖 AI Agent 1 ({mistral_model}): Extracting structured data..."):
            max_chars = 16000 # Adjust based on model context window if needed
            # Prompt remains largely the same, Mistral models understand similar instructions
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
            # Call the Mistral response function
            extracted_json_str = get_mistral_response(
                mistral_client, # Pass the client
                prompt_extract,
                model_name=mistral_model,
                is_json=True,
                temperature=0.1 # Low temp for JSON
            )

            if extracted_json_str:
                st.session_state.extracted_json_str = extracted_json_str
                temp_extracted_data = validate_and_load_json(extracted_json_str)
                if temp_extracted_data is not None:
                     st.session_state.extracted_data = temp_extracted_data
                     st.success(f"✅ AI Agent 1 ({mistral_model}): Structured data extracted and parsed.")
                else:
                     st.error(f"❌ AI Agent 1 ({mistral_model}): Failed to parse AI response as valid JSON.")
                     st.text_area("Invalid JSON received:", extracted_json_str, height=150)
            else:
                # Error message likely already shown by get_mistral_response
                st.error(f"❌ AI Agent 1 ({mistral_model}): No response or error during extraction.")

    # 5. AI Agent: Verification (using Mistral) - CHANGED
    if st.session_state.extracted_data:
         with st.spinner(f"🕵️ AI Agent 2 ({mistral_model}): Verifying extracted data..."):
            max_chars_verify = 8000
            # Prompt remains the same
            prompt_verify = f"""
            You are an AI verification agent. Review the JSON data supposedly extracted from a source text and check its accuracy against that text.
            {...} # Same prompt structure as before
            """
            # Call the Mistral response function
            verification_feedback = get_mistral_response(
                mistral_client, # Pass the client
                prompt_verify,
                model_name=mistral_model,
                is_json=False,
                temperature=0.4 # Slightly higher temp for feedback
            )
            if verification_feedback:
                st.session_state.verification_feedback = verification_feedback
                st.success(f"✅ AI Agent 2 ({mistral_model}): Verification complete.")
            else:
                st.warning(f"⚠️ AI Agent 2 ({mistral_model}): Could not get verification feedback or error occurred.")


# --- Display Results ---
# (Display logic remains the same)
# ...

# --- Optional Analysis Section --- (using Mistral) - CHANGED
if st.session_state.dataframe is not None:
    # (UI for asking question remains the same)
    # ...
    if st.session_state.analysis_confirmed and st.session_state.analysis_result is None:
        with st.spinner(f"💭 AI Agent 3 ({mistral_model}): Analyzing data..."):
            if not st.session_state.dataframe.empty:
                 # (Preparing data string remains the same)
                 # ...
                 # Prompt remains the same
                 prompt_analyze = f"""
                 You are an AI data analyst. Analyze the following data based on the user's question.
                 {...} # Same prompt structure as before
                 """
                 # Call the Mistral response function
                 analysis_result = get_mistral_response(
                     mistral_client, # Pass the client
                     prompt_analyze,
                     model_name=mistral_model,
                     is_json=False,
                     temperature=0.5 # Higher temp for analysis
                 )
                 if analysis_result:
                     st.session_state.analysis_result = analysis_result
                     st.success(f"✅ AI Agent 3 ({mistral_model}): Analysis complete.")
                 else:
                     st.error(f"❌ AI Agent 3 ({mistral_model}): Failed to get analysis result or error occurred.")
            else:
                # (Handling empty dataframe remains the same)
                # ...

    # (Displaying analysis result remains the same)
    # ...


# Display Original Extracted Text (Optional)
# (Remains the same)
# ...
