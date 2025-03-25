# -----------------------------------------------------------------------------
# Streamlit PDF Data Extractor using Mistral AI
# -----------------------------------------------------------------------------
# Description:
# This application allows users to upload a PDF document and a JSON schema
# (pasted as text). It extracts text from the PDF (using OCR if needed), then
# uses a selected Mistral AI model (via the client.chat endpoint) to structure
# the extracted text according to the provided schema. A second Mistral AI agent
# verifies the extraction. The data is displayed, converted to a DataFrame for
# preview/download. Optionally, a third agent analyzes the data based on user input.
#
# Requirements:
# pip install streamlit pandas pdfminer.six pytesseract Pillow mistralai python-dotenv
#
# Setup:
# 1. Install Tesseract OCR engine on your system (and potentially poppler).
# 2. Create a .env file in the same directory with your Mistral API key:
#    MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY_HERE
#    (Or use Streamlit secrets: [secrets] MISTRAL_API_KEY = "...")
# 3. Run the app: streamlit run app.py
#
# Notes:
# - OCR requires Tesseract to be installed and accessible.
# - Uses pdfminer.six / pytesseract for text/OCR extraction.
# - Uses MistralClient and client.chat endpoint with JSON prompting.
# - Includes 'pixtral-12b-latest' in model list, but effectiveness may vary.
# - Rate limits depend on your Mistral AI usage tier.
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import json
from io import BytesIO
import time
import os
import tempfile

# PDF Processing
from pdfminer.high_level import extract_text as extract_text_miner
from pdfminer.layout import LAParams
import pytesseract
from PIL import Image # Pillow is used indirectly by pytesseract

# AI Integration - Mistral AI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralAPIException, MistralConnectionException, MistralAuthenticationError

# Environment variables
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# --- Configuration ---
# Use st.secrets for deployment, otherwise use environment variables or direct input
try:
    # Try getting key from Streamlit secrets first (for deployment)
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback to environment variable
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- Helper Functions ---

@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(uploaded_file_content):
    """Extracts text from PDF bytes using pdfminer.six, falling back to OCR."""
    extracted_text = ""
    try:
        file_bytes = BytesIO(uploaded_file_content)
        # Try pdfminer.six first (good for text-based PDFs)
        extracted_text = extract_text_miner(file_bytes, laparams=LAParams())
        st.sidebar.caption("Text extracted via pdfminer.")
    except Exception as e:
        st.sidebar.warning(f"pdfminer.six failed: {e}. Attempting OCR...")
        extracted_text = "" # Reset text if pdfminer failed

    # Fallback to OCR if pdfminer extraction seems insufficient
    # This threshold is arbitrary and might need adjustment
    if len(extracted_text.strip()) < 100:
        st.sidebar.warning("Extracted text seems short, attempting OCR.")
        try:
            # Use temp file for pytesseract compatibility
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(uploaded_file_content)
                tmp_pdf_path = tmp_pdf.name

            ocr_start_time = time.time()
            st.sidebar.caption("Running OCR (may take time)...")
            # Pass the path to pytesseract
            # Requires Tesseract (+ potentially Poppler) installed and in PATH
            extracted_text_ocr = pytesseract.image_to_string(tmp_pdf_path)
            ocr_end_time = time.time()
            st.sidebar.caption(f"OCR attempt took {ocr_end_time - ocr_start_time:.2f}s.")

            # Use OCR text only if it's significantly longer
            if len(extracted_text_ocr.strip()) > len(extracted_text.strip()):
                 st.sidebar.caption("OCR produced more text.")
                 extracted_text = extracted_text_ocr
            else:
                 st.sidebar.caption("Keeping original extracted text (or OCR was empty).")

            os.unlink(tmp_pdf_path) # Clean up temp file

        except Exception as ocr_e:
            st.sidebar.error(f"OCR process failed: {ocr_e}")
            # Return whatever pdfminer got, or None if it also failed
            if not extracted_text:
                 return None

    if not extracted_text.strip():
         st.error("Could not extract significant text from the PDF.")
         return None

    return extracted_text

# Cache the Mistral client to avoid re-initializing on every interaction
@st.cache_resource # Caching the client connection object
def configure_mistral(api_key):
    """Configures and returns the Mistral client."""
    if not api_key:
        return None # Error handled where called
    try:
        client = MistralClient(api_key=api_key)
        # Optional: Lightweight check like listing models can verify the key/connection
        # client.list_models()
        print("Mistral client configured successfully.") # Log for debugging
        return client
    except MistralAuthenticationError:
         # Let the caller handle the UI error reporting
         print("Mistral Authentication Error during configuration.") # Log for debugging
         return None
    except Exception as e:
        # Catch other potential errors during client init
        st.error(f"Error initializing Mistral client: {e}")
        return None

def get_mistral_response(client, prompt, model_name="mistral-large-latest", is_json=False, temperature=0.3):
    """Sends prompt to Mistral AI chat API (client.chat) and gets response."""
    if not client:
         st.error("Mistral client is not configured.")
         return None

    messages = [ChatMessage(role="user", content=prompt)]
    response_format_arg = None # Default to text format

    try:
        # Prepare arguments for client.chat
        chat_args = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            # Add safety settings or other parameters if desired
            # "safe_prompt": True,
        }

        if is_json:
            # Add system message and potentially request JSON format
            messages = [
                 ChatMessage(role="system", content="You are a helpful assistant designed to output JSON."),
                 ChatMessage(role="user", content=prompt + "\n\nPlease ensure your entire response is only the valid JSON object or array requested, enclosed in curly braces {} or square brackets [], with no other text before or after.")
             ]
            chat_args["messages"] = messages # Update messages in args

            # Check if model likely supports JSON mode (e.g., Mistral Large 2 does)
            # Consult Mistral documentation for definitive list for the specific version
            if "large" in model_name or "medium" in model_name:
                response_format_arg = {"type": "json_object"}
                chat_args["response_format"] = response_format_arg
                print(f"Requesting JSON mode for model {model_name}") # Log for debugging

        # Make the API call
        api_start_time = time.time()
        response = client.chat(**chat_args)
        api_end_time = time.time()
        print(f"Mistral API call to {model_name} took {api_end_time - api_start_time:.2f}s") # Log for debugging

        response_text = response.choices[0].message.content.strip()

        # Manual cleanup as fallback if JSON mode wasn't used or failed
        # (This is less likely needed if JSON mode is used correctly, but kept as safeguard)
        if is_json and not response_format_arg:
            print("Attempting manual JSON cleanup (fallback).") # Log for debugging
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
         # Handle model not found specifically if possible (check exception type/message)
         elif "model not found" in error_detail:
              st.error(f"Mistral Model Not Found Error: The model '{model_name}' may not be available via this API endpoint or your account. Please try another model.")
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

def validate_and_load_json(json_string):
    """Tries to load a JSON string, returns None on failure."""
    if not json_string:
        st.warning("Received empty response from AI for JSON extraction.")
        return None
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Error message displayed where this function is called
        print(f"JSON Decode Error: {e}") # Log for debugging
        return None

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="PDF Data Extractor (Mistral AI)")
st.title("📄 Agentic PDF Data Extractor & Analyzer (using Mistral AI)")

# --- Initialize Session State ---
# Using session state to store results across reruns and manage UI flow
if 'pdf_text' not in st.session_state: st.session_state.pdf_text = None
if 'extracted_json_str' not in st.session_state: st.session_state.extracted_json_str = None
if 'extracted_data' not in st.session_state: st.session_state.extracted_data = None
if 'verification_feedback' not in st.session_state: st.session_state.verification_feedback = None
if 'dataframe' not in st.session_state: st.session_state.dataframe = None
if 'analysis_query_input' not in st.session_state: st.session_state.analysis_query_input = ""
if 'analysis_requested' not in st.session_state: st.session_state.analysis_requested = False
if 'analysis_confirmed' not in st.session_state: st.session_state.analysis_confirmed = False
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key Input
    api_key_input = st.text_input(
        "Enter your Mistral API Key:",
        type="password",
        value=MISTRAL_API_KEY or "",
        help="Get your key from console.mistral.ai. Uses MISTRAL_API_KEY env var if set."
    )

    # Instantiate Mistral client
    mistral_client = None
    client_configured_successfully = False
    if api_key_input:
        # Attempt to configure the client using the cached function
        mistral_client = configure_mistral(api_key_input)
        if mistral_client:
            st.sidebar.success("Mistral client ready.")
            client_configured_successfully = True
        else:
             # configure_mistral handles logging/errors, show simple sidebar error
             st.sidebar.error("Failed to configure Mistral client. Check API Key?")
             client_configured_successfully = False
    else:
        st.sidebar.warning("Please enter your Mistral API Key.")

    st.divider()
    st.header("📄 Inputs")
    uploaded_pdf = st.file_uploader("1. Upload PDF Document", type="pdf", key="pdf_uploader")

    # Schema Input from Text Area
    schema_input_str = st.text_area(
        "2. Paste JSON Schema Here",
        height=250,
        placeholder='{\n  "field_name_1": "description or type",\n  "field_name_2": "description or type",\n  "nested_object": {\n    "sub_field": "type"\n  },\n  "list_field": ["item_type"]\n}',
        help="Paste the JSON structure you want to extract data into. Use valid JSON syntax.",
        key="schema_input"
    )

    # Attempt to parse schema immediately to validate syntax for button state
    schema_dict = None
    is_schema_valid = False
    if schema_input_str:
        try:
            schema_dict = json.loads(schema_input_str)
            is_schema_valid = True
            st.sidebar.caption("✔️ Schema Syntax OK")
        except json.JSONDecodeError as e:
            st.sidebar.error(f"Invalid JSON Schema Syntax: {e}")
            is_schema_valid = False
            schema_dict = None # Ensure dict is None on error
    else:
        st.sidebar.caption("Waiting for JSON schema...")

    # Choose Model - Includes pixtral-12b-latest as requested
    mistral_model = st.selectbox(
        "3. Select Mistral Model",
        (
            "mistral-large-latest", # Default, generally good capability
            "pixtral-12b-latest",   # Added based on user example (may not work with client.chat)
            "open-mixtral-8x22b",   # Large open model via API
            "mistral-medium-latest",# Balanced option
            "mistral-small-latest", # Faster, cheaper option
            "open-mixtral-8x7b",    # Common Mixtral model via API
            # Add other models if needed, e.g., specific versions like 'mistral-large-2402'
        ),
        index=0, # Default to mistral-large-latest
        key="model_selector",
        help="Select the Mistral AI model. Note: 'pixtral-12b-latest' might require specific API endpoints not used here and could error."
    )

    st.divider()

    # Button Disabled Logic - Checks all necessary inputs/configs
    all_inputs_ready = (
        uploaded_pdf is not None and
        is_schema_valid and # Check if schema JSON is syntactically valid
        api_key_input and # Check if key is entered
        client_configured_successfully # Check if client was successfully created
    )
    process_button = st.button(
        "🚀 Process PDF",
        type="primary",
        disabled=not all_inputs_ready,
        key="process_button",
        help="Requires PDF, valid schema, and configured API key."
    )

    # Placeholder for download button (appears in sidebar after processing)
    download_placeholder = st.empty()

# --- Main Area for Processing and Results ---
st.subheader("📈 Results")
results_container = st.container() # Use a container for results area

# --- Processing Logic ---
if process_button:
    # Clear previous results from display area immediately
    results_container.empty()

    # Reset state variables for a fresh run
    st.session_state.pdf_text = None
    st.session_state.extracted_json_str = None
    st.session_state.extracted_data = None
    st.session_state.verification_feedback = None
    st.session_state.dataframe = None
    st.session_state.analysis_requested = False # Reset analysis state too
    st.session_state.analysis_confirmed = False
    st.session_state.analysis_result = None
    # Keep analysis_query_input or clear it? Let's clear it for a fresh process run
    # st.session_state.analysis_query_input = ""

    # 1. Configuration checks (mostly done for button state, safety checks here)
    if not mistral_client:
        results_container.error("Mistral client not configured correctly. Please check API key in sidebar.")
        st.stop()
    if not is_schema_valid or not schema_dict:
        results_container.error("JSON Schema is invalid or missing. Please check the sidebar.")
        st.stop() # schema_dict should be valid here because the button required it
    if not uploaded_pdf:
         results_container.error("Please upload a PDF file.") # Should be caught by button state
         st.stop()

    # Read PDF content here to pass to function
    pdf_content = uploaded_pdf.getvalue()

    # 2. Schema is already loaded into schema_dict from sidebar logic

    # 3. Extract Text from PDF
    # Spinner applied via @st.cache_data decorator on the function
    pdf_text = extract_text_from_pdf(pdf_content)
    if pdf_text:
        st.session_state.pdf_text = pdf_text
        results_container.success("✅ Text extracted from PDF.")
    else:
        results_container.error("❌ Failed to extract text from PDF. Cannot proceed.")
        st.stop() # Stop execution if text extraction fails

    # 4. AI Agent 1: Structured Data Extraction (using Mistral)
    if st.session_state.pdf_text:
        with results_container: # Show spinner within the results area
            with st.spinner(f"🤖 AI Agent 1 ({mistral_model}): Extracting structured data..."):
                # Context window limits vary; check Mistral docs. 16k chars is often safe.
                max_chars_extract = 16000 # Consider making this configurable
                prompt_extract = f"""
                You are an AI assistant tasked with extracting structured data from the following text based on the provided JSON schema.

                **JSON Schema:**
                ```json
                {json.dumps(schema_dict, indent=2)}
                ```

                **Source Text (first {max_chars_extract} characters):**
                ```text
                {st.session_state.pdf_text[:max_chars_extract]}
                ```
                **(Note: Text might be truncated if very long)**

                **Instructions:**
                1. Read the Source Text carefully.
                2. Identify information corresponding to the fields in the JSON Schema.
                3. Format the extracted information *strictly* according to the JSON Schema.
                4. If a field's value isn't found, use `null` or omit the field as appropriate for the schema.
                5. Ensure the output is a *single, valid JSON object* or a *valid array of JSON objects* if the schema suggests multiple entries.
                6. **Crucially: Your response must contain *only* the JSON data itself, with no surrounding text, comments, or explanations.**
                """
                extracted_json_str = get_mistral_response(
                    mistral_client, # Pass the client
                    prompt_extract,
                    model_name=mistral_model,
                    is_json=True,
                    temperature=0.1 # Low temp for precise JSON extraction
                )

                if extracted_json_str:
                    st.session_state.extracted_json_str = extracted_json_str
                    # Validate the JSON received from the AI
                    temp_extracted_data = validate_and_load_json(extracted_json_str)
                    if temp_extracted_data is not None:
                         st.session_state.extracted_data = temp_extracted_data
                         st.success(f"✅ AI Agent 1 ({mistral_model}): Structured data extracted and parsed.")
                    else:
                         # Validation failed, show error and the raw response
                         st.error(f"❌ AI Agent 1 ({mistral_model}): Failed to parse AI response as valid JSON.")
                         st.text_area("Invalid JSON received:", extracted_json_str, height=150, key="invalid_json_agent1")
                else:
                    # Error message likely already shown by get_mistral_response
                    st.error(f"❌ AI Agent 1 ({mistral_model}): No response or error during extraction.")

    # 5. AI Agent 2: Verification (using Mistral)
    # Run only if extraction was successful and produced valid data structure
    if st.session_state.extracted_data is not None:
         with results_container: # Show spinner in results area
            with st.spinner(f"🕵️ AI Agent 2 ({mistral_model}): Verifying extracted data..."):
                # Use less text for verification to potentially save tokens/time
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
                1. Compare the values in the Extracted JSON Data with the Source Text provided.
                2. Identify specific discrepancies, inaccuracies, or potentially missing information in the JSON based *only* on the Source Text shown.
                3. Provide a brief summary of your findings (e.g., using bullet points). Mention specific fields if they seem incorrect or questionable.
                4. If everything looks accurate according to the text provided, state that clearly.
                5. Do NOT output JSON. Provide feedback as plain text. Focus on mismatches or confirmations.
                """
                verification_feedback = get_mistral_response(
                    mistral_client, # Pass the client
                    prompt_verify,
                    model_name=mistral_model,
                    is_json=False,
                    temperature=0.4 # Slightly higher temp for descriptive feedback
                )
                if verification_feedback:
                    st.session_state.verification_feedback = verification_feedback
                    st.success(f"✅ AI Agent 2 ({mistral_model}): Verification check complete.")
                else:
                    # Error likely shown by get_mistral_response
                    st.warning(f"⚠️ AI Agent 2 ({mistral_model}): Could not get verification feedback or error occurred.")


# --- Display Results --- (Executed on rerun after processing or if state exists)

# Use columns within the main container for better layout
with results_container:
    # Display only if processing has occurred or state exists
    if st.session_state.pdf_text or st.session_state.extracted_data or st.session_state.dataframe is not None:
        col1, col2 = st.columns(2)

        with col1:
            # Display Verification Feedback First if available
            if st.session_state.verification_feedback:
                st.subheader("🕵️ Verification Agent Feedback")
                st.markdown(st.session_state.verification_feedback)
                st.divider()

            # Display Extracted Data (JSON) if available
            if st.session_state.extracted_data is not None:
                st.subheader("📊 Extracted Data (JSON)")
                st.json(st.session_state.extracted_data)

        with col2:
            # Convert to DataFrame and Display if data exists
            dataframe_created_or_existed = False
            if st.session_state.extracted_data is not None and 'dataframe' not in st.session_state:
                 # Attempt conversion only if data is present and df doesn't exist yet
                 try:
                     data_for_df = st.session_state.extracted_data
                     if isinstance(data_for_df, dict):
                         data_for_df = [data_for_df] # Wrap single object

                     if isinstance(data_for_df, list) and (not data_for_df or all(isinstance(item, dict) for item in data_for_df)):
                          if not data_for_df:
                              st.info("Extracted data is empty list. DataFrame is empty.")
                              st.session_state.dataframe = pd.DataFrame()
                          else:
                              st.session_state.dataframe = pd.DataFrame(data_for_df)

                          dataframe_created_or_existed = True # Mark success
                     else:
                          st.warning("Extracted data not list of objects. Cannot create DataFrame.")
                          st.session_state.dataframe = None

                 except Exception as e:
                     st.error(f"Error converting JSON to DataFrame: {e}")
                     st.session_state.dataframe = None
            elif st.session_state.dataframe is not None:
                 # DataFrame already exists in state
                 dataframe_created_or_existed = True


            # Display DataFrame if it was created or already existed
            if dataframe_created_or_existed and st.session_state.dataframe is not None:
                 st.subheader("📋 DataFrame Preview")
                 st.dataframe(st.session_state.dataframe)

                 # Add download button to sidebar placeholder (only if df exists)
                 if not st.session_state.dataframe.empty:
                     csv = st.session_state.dataframe.to_csv(index=False).encode('utf-8')
                     with download_placeholder: # Use the placeholder in the sidebar
                          st.download_button(
                               label="📥 Download DataFrame as CSV",
                               data=csv,
                               file_name=f"{st.session_state.get('pdf_uploader_name', 'extracted_data')}.csv", # Use stored name if available
                               mime="text/csv",
                               key="download_csv_button"
                          )
                 else:
                      with download_placeholder:
                           st.caption("DataFrame is empty.") # Indicate empty df in download area

# --- Optional Analysis Section ---
# Display this section regardless of processing button press if dataframe exists
st.divider()
st.subheader("🧠 Data Analysis")

if st.session_state.dataframe is None:
    st.info("Extract data first to enable analysis.")
else:
    if st.session_state.dataframe.empty:
        st.warning("DataFrame is empty, cannot perform analysis.")
    else:
        # Use a callback to set analysis_requested flag when text is entered
        def request_analysis():
            st.session_state.analysis_requested = bool(st.session_state.analysis_query_input)
            # Reset confirmation and result if query changes
            if st.session_state.analysis_confirmed or st.session_state.analysis_result:
                st.session_state.analysis_confirmed = False
                st.session_state.analysis_result = None

        analysis_query = st.text_area(
            "Ask a question about the extracted data:",
            key="analysis_query_input",
            on_change=request_analysis,
            value=st.session_state.analysis_query_input, # Persist query on non-processing reruns
            help="Type your question and press Enter/Return. Then click 'Yes, Analyze This Data'."
        )

        # Show confirmation button only if analysis is requested but not yet confirmed
        if st.session_state.analysis_requested and not st.session_state.analysis_confirmed:
            st.warning("**Review the extracted DataFrame above before proceeding.**")
            confirm_analysis = st.button("Yes, Analyze This Data", key="confirm_analysis_button")
            if confirm_analysis:
                st.session_state.analysis_confirmed = True
                st.session_state.analysis_result = None # Ensure previous result is cleared
                st.rerun() # Rerun to trigger analysis logic below

        # Perform analysis if confirmed and result not yet available
        # This block runs only *after* the rerun triggered by the confirmation button
        if st.session_state.analysis_confirmed and st.session_state.analysis_result is None:
            with st.spinner(f"💭 AI Agent 3 ({mistral_model}): Analyzing data..."):
                 # Limit rows sent for analysis to avoid excessive token usage/cost
                 max_rows_analysis = 100 # Configurable?
                 df_subset = st.session_state.dataframe.head(max_rows_analysis)
                 df_string = df_subset.to_csv(index=False)
                 num_rows_sent = len(df_subset)
                 total_rows = len(st.session_state.dataframe)
                 row_info = f"first {num_rows_sent} rows" if total_rows > max_rows_analysis else f"all {total_rows} rows"

                 prompt_analyze = f"""
                 You are an AI data analyst. Analyze the following data based on the user's question.

                 **Data ({row_info}, CSV format):**
                 ```csv
                 {df_string}
                 ```
                 **(Note: Data might be truncated for analysis)**

                 **User's Question:**
                 {st.session_state.analysis_query_input}

                 **Instructions:**
                 1. Understand the user's question.
                 2. Analyze the provided data to answer the question.
                 3. Provide a clear and concise answer based *only* on the data given.
                 4. If the data is insufficient or irrelevant to the question, state that clearly.
                 5. Present the analysis in a readable format (e.g., text, bullet points). Do not output CSV or code unless specifically asked.
                 """
                 analysis_result_text = get_mistral_response(
                     mistral_client,
                     prompt_analyze,
                     model_name=mistral_model,
                     is_json=False,
                     temperature=0.5 # Moderate temp for analysis/explanation
                 )
                 if analysis_result_text:
                     st.session_state.analysis_result = analysis_result_text
                     # Analysis result will be displayed below automatically on rerun
                 else:
                     st.error(f"❌ AI Agent 3 ({mistral_model}): Failed to get analysis result or error occurred.")
                     # Reset confirmation to allow potential retry if analysis failed
                     st.session_state.analysis_confirmed = False
                 st.rerun() # Rerun once more to display the analysis result cleanly

        # Display analysis result if available
        if st.session_state.analysis_result:
             st.subheader("📈 Analysis Results")
             st.markdown(st.session_state.analysis_result)

# Display Original Extracted Text (Optional, at the very bottom)
if st.session_state.pdf_text:
    with st.expander("View Extracted Raw Text (Truncated)", expanded=False):
        st.text(st.session_state.pdf_text[:5000] + ("..." if len(st.session_state.pdf_text) > 5000 else ""))
