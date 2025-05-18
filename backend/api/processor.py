import pandas as pd
import re
from django.conf import settings
import os
import numpy as np
import google.generativeai as genai
import traceback 
import json 


df_global = pd.DataFrame()

# --- Column Name Constants  ---
COL_AREA = 'final location' 
COL_YEAR = 'year'
COL_CITY = 'city'
COL_PRICE_RES = 'weighted average rsf residential'
COL_DEMAND_TOTAL_SOLD = 'total sold'
COL_TOTAL_SALES_VALUE = 'total_sales' 
COL_RES_SOLD_COUNT = 'res_sold' 
COL_OFFICE_SOLD_COUNT = 'office_sold' 
COL_SHOP_SOLD_COUNT = 'shop_sold' 

EXCEL_FILE_PATH = None
GOOGLE_API_KEY = None
gemini_model = None

try:
    EXCEL_FILE_PATH = settings.DATA_FILE_PATH
    if not EXCEL_FILE_PATH: # Handles if settings.DATA_FILE_PATH is an empty string
        print("WARNING (processor.py): DATA_FILE_PATH in Django settings is empty or not set.")
        EXCEL_FILE_PATH = None # Ensure it's explicitly None
except AttributeError:
    print("CRITICAL ERROR (processor.py): DATA_FILE_PATH not found as an attribute in Django settings.")

try:
    GOOGLE_API_KEY = settings.GOOGLE_API_KEY
except AttributeError:
    print("WARNING (processor.py): GOOGLE_API_KEY not found as an attribute in Django settings.")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("INFO (processor.py): Google Gemini API configured successfully with 'gemini-1.5-flash-latest'.")
    except Exception as e:
        print(f"ERROR (processor.py): Failed to configure Google Gemini API: {e}")
        traceback.print_exc()
else:
    print("INFO (processor.py): GOOGLE_API_KEY not set. LLM features will be mocked or limited.")

# --- ACTUAL DATA LOADING LOGIC ---
if EXCEL_FILE_PATH and os.path.exists(EXCEL_FILE_PATH):
    try:
        print(f"INFO (processor.py): Attempting to load Excel file from: {EXCEL_FILE_PATH}")
        df_temp = pd.read_excel(EXCEL_FILE_PATH, na_values=['#######'])
        print(f"INFO (processor.py): Excel file loaded into temporary DataFrame. Shape: {df_temp.shape}")

        if not df_temp.empty:
            if COL_AREA in df_temp.columns:
                df_temp.rename(columns={COL_AREA: 'Area'}, inplace=True)
                df_temp['Area'] = df_temp['Area'].astype(str).str.strip().str.title()
            else:
                print(f"WARNING (processor.py): Column '{COL_AREA}' (for Area) not found in Excel.")

            if 'Area' not in df_temp.columns:
                 print("CRITICAL (processor.py): 'Area' column is missing after potential rename. df_global will remain empty.")
            else:
                numeric_cols_to_process = { 
                    COL_YEAR: 'Int64', COL_PRICE_RES: 'float', COL_DEMAND_TOTAL_SOLD: 'float',
                    COL_TOTAL_SALES_VALUE: 'float', COL_RES_SOLD_COUNT: 'Int64',
                    COL_OFFICE_SOLD_COUNT: 'Int64', COL_SHOP_SOLD_COUNT: 'Int64',
                    'total units': 'Int64', 'net area supplied': 'float', 'loc_lat': 'float', 'loc_lng': 'float',
                    'weighted average rsf office': 'float', 'weighted average rsf retail': 'float',
                    'weighted average rsf others': 'float', 'flat total': 'Int64',
                    'shop total': 'Int64', 'office total': 'Int64', 'others_sold': 'Int64',
                    'commercial_sold': 'Int64'
                }
                for col, target_type in numeric_cols_to_process.items():
                    if col in df_temp.columns:
                        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                        # Apply Int64 conversion only if it's numeric and not all NaNs
                        if target_type == 'Int64' and pd.api.types.is_numeric_dtype(df_temp[col]) and not df_temp[col].isnull().all():
                            df_temp[col] = df_temp[col].round().astype('Int64')
                    # else:
                    #     print(f"INFO (processor.py): Column '{col}' for numeric conversion not found in Excel.")

                if COL_CITY in df_temp.columns: # Handle city if present
                     df_temp[COL_CITY] = df_temp[COL_CITY].astype(str).str.strip().str.title()

                df_global = df_temp.copy() # Assign to global df
                print("INFO (processor.py): DataFrame loaded and preprocessed successfully. df_global shape:", df_global.shape)
                # print("First 5 rows of df_global:\n", df_global.head()) # For debugging
        else:
            print("WARNING (processor.py): Excel file was loaded but is empty. df_global remains empty.")
    except Exception as e:
        print(f"CRITICAL ERROR (processor.py): During Excel loading or initial processing: {e}")
        traceback.print_exc() # Print full traceback for this specific error
elif EXCEL_FILE_PATH: # Path was set but file doesn't exist
    print(f"WARNING (processor.py): Data file specified at {EXCEL_FILE_PATH} was not found. df_global will be empty.")
else: # EXCEL_FILE_PATH was None from the start
     print("WARNING (processor.py): EXCEL_FILE_PATH is not configured in settings. df_global will be empty.")


def get_known_area_names_for_prompt():
    if not df_global.empty and 'Area' in df_global.columns:
        unique_areas = df_global['Area'].dropna().unique()
        if len(unique_areas) > 0:
            return ", ".join(sorted(unique_areas.tolist())) # Sorted for consistency
    print("INFO (processor.py): Using fallback KNOWN_AREAS list as df_global is empty or 'Area' column is missing.")
    return "Wakad, Hinjewadi, Akurdi, Aundh, Ambegaon Budruk" 

KNOWN_AREAS = get_known_area_names_for_prompt()
print(f"INFO (processor.py): KNOWN_AREAS set to: {KNOWN_AREAS}")


def get_relevant_table_columns(df_input):
    cols_map = { 
        'Area': 'Area',
        COL_YEAR: 'Year',
        COL_CITY: 'City',
        COL_PRICE_RES: 'Avg. Res. Price (RSF)',
        COL_DEMAND_TOTAL_SOLD: 'Total Units Sold',
        COL_RES_SOLD_COUNT: 'Res. Units Sold',
        COL_OFFICE_SOLD_COUNT: 'Office Units Sold',
        COL_SHOP_SOLD_COUNT: 'Shop Units Sold',
        'total units': 'Total Units Avail.',
        'net area supplied': 'Net Area Supplied',
        'weighted average rsf office': 'Avg. Office Price',
        'weighted average rsf retail': 'Avg. Retail Price'
        
    }
    relevant_cols_df = pd.DataFrame()
    for original_col, display_name in cols_map.items():
        if original_col in df_input.columns: 
            relevant_cols_df[display_name] = df_input[original_col]
    return relevant_cols_df

# --- LLM NLU Function ---
def understand_query_with_gemini(user_query):
    if not gemini_model:
        print("WARNING (processor.py): Gemini model not available for NLU.")
        return None

    prompt = f"""
    You are an expert real estate market query analyzer.
    Your task is to analyze the user's query and extract specific information.
    You MUST return your response as a single, valid JSON object. Do not include any other text, explanations, or markdown formatting like ```json.

    Follow these steps:
    1. Identify the user's primary intent. Possible intents are: "analyze_single_area", "compare_area_demand", "price_growth_area", "unknown".
    2. Extract relevant entities based on the intent.

    Entities to extract:
    - "area_1": (STRING, REQUIRED for "analyze_single_area" and "price_growth_area", and first area for "compare_area_demand") The primary real estate area name. If the query is "analyze Wakad", area_1 MUST be "Wakad".
    - "area_2": (STRING, OPTIONAL, only for "compare_area_demand") The second area name.
    - "years": (INTEGER, OPTIONAL, only for "price_growth_area") The number of years.

    Normalization Rules:
    - If an area name mentioned by the user closely matches one of these known areas, use the known area name: {KNOWN_AREAS}.
    - For example, "info about Wakad" or "tell me about wakad area" should result in "area_1": "Wakad".
    - If a query is simply an area name like "Wakad", the intent is "analyze_single_area" and "area_1" is "Wakad".

    Example of a perfect JSON output for "analyze wakad":
    {{
        "intent": "analyze_single_area",
        "entities": {{
            "area_1": "Wakad"
        }}
    }}

    Another Example for "compare hinjewadi and ambegaon demand":
    {{
        "intent": "compare_area_demand",
        "entities": {{
            "area_1": "Hinjewadi",
            "area_2": "Ambegaon Budruk"
        }}
    }}

    User Query: "{user_query}"

    JSON Response:
    """
    print(f"INFO (processor.py): Sending to Gemini for NLU: '{user_query}'")
    try:
        custom_safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=custom_safety_settings)
        
        if response.parts:
            raw_json_response = response.text.strip()
            print(f"INFO (processor.py): Gemini NLU Raw Response: {raw_json_response}")
            
            if raw_json_response.startswith("```json"): raw_json_response = raw_json_response[len("```json"):]
            if raw_json_response.startswith("```"): raw_json_response = raw_json_response[len("```"):]
            if raw_json_response.endswith("```"): raw_json_response = raw_json_response[:-len("```")]
            
            parsed_response = json.loads(raw_json_response.strip())
            print(f"INFO (processor.py): Gemini NLU Parsed Response: {parsed_response}")

            # Debug checks for missing entities based on intent
            intent_check = parsed_response.get("intent")
            entities_check = parsed_response.get("entities")
            if intent_check in ["analyze_single_area", "price_growth_area"]:
                if not entities_check or "area_1" not in entities_check or not entities_check.get("area_1"):
                    print(f"WARNING (processor.py): NLU intent '{intent_check}' but MISSED 'area_1'. Entities: {entities_check}. Query: '{user_query}'")
            elif intent_check == "compare_area_demand":
                 if not entities_check or "area_1" not in entities_check or not entities_check.get("area_1") or \
                    "area_2" not in entities_check or not entities_check.get("area_2"):
                    print(f"WARNING (processor.py): NLU intent '{intent_check}' but MISSED 'area_1' or 'area_2'. Entities: {entities_check}. Query: '{user_query}'")
            
            return parsed_response
        else:
            block_reason = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
            print(f"WARNING (processor.py): Gemini NLU returned no parts. Block reason: {block_reason}. Feedback: {response.prompt_feedback}")
            return {"intent": "unknown", "error": f"LLM NLU no parts, reason: {block_reason}"}
    except json.JSONDecodeError as jde:
        raw_json_response_for_error = 'Not available'
        if 'raw_json_response' in locals(): raw_json_response_for_error = raw_json_response
        print(f"ERROR (processor.py): Failed to decode JSON from Gemini NLU: {jde}. Raw response: '{raw_json_response_for_error}'")
        return {"intent": "unknown", "error": "LLM NLU JSON decode error"}
    except Exception as e:
        print(f"ERROR (processor.py): Exception during Gemini NLU call: {e}")
        traceback.print_exc()
        return {"intent": "unknown", "error": "LLM NLU API call exception"}

# --- LLM Summarization Function ---
def generate_llm_summary(context_data_str, display_name, analysis_type="general"):
    llm_prefix = "[Answered by Ravilesh-LLM]: "
    mock_prefix_not_configured = "[Mocked Summary - LLM Not Configured]: "
    mock_prefix_api_fail = "[Mocked Summary - LLM API Call Failed]: "

    if not gemini_model:
        return f"{mock_prefix_not_configured}Summary for {display_name} based on local data patterns."

    prompt = "" # Default empty prompt
    if analysis_type == "general_area_analysis":
        prompt = f"Provide a concise (2-3 sentences) real estate market summary for {display_name}, focusing on residential prices (column '{COL_PRICE_RES}') and sales volume (column '{COL_DEMAND_TOTAL_SOLD}'). Data:\n{context_data_str}"
    elif analysis_type == "price_growth":
        prompt = f"Summarize the residential price growth for {display_name} (column '{COL_PRICE_RES}') based on this data:\n{context_data_str}\nProvide a 1-2 sentence insight."
    elif analysis_type == "demand_comparison":
        prompt = f"Compare real estate demand (column '{COL_DEMAND_TOTAL_SOLD}') for {display_name} using this context:\n{context_data_str}\nOffer a brief 1-2 sentence comparative insight."
    else:
        return f"{mock_prefix_not_configured}Unknown analysis type '{analysis_type}' for {display_name}."

    print(f"INFO (processor.py): Calling Gemini for final summary: {display_name}, Type: {analysis_type}")
    try:
        custom_safety_settings = [ # Re-apply less restrictive safety settings for summarization too
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=custom_safety_settings)
        if response.parts:
            return llm_prefix + response.text.strip()
        else:
            block_reason_sum = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_sum = response.prompt_feedback.block_reason.name
            print(f"WARNING (processor.py): Gemini summary returned no parts for {display_name}. Block reason: {block_reason_sum}. Feedback: {response.prompt_feedback}")
            return f"{mock_prefix_api_fail}Summary for {display_name} had an issue (no parts, reason: {block_reason_sum})."
    except Exception as e:
        print(f"ERROR (processor.py): Exception during Gemini summary call for {display_name}: {e}")
        traceback.print_exc()
        return f"{mock_prefix_api_fail}Summary for {display_name} used a fallback due to API error."

# --- Core Analysis Functions ---
def analyze_area(area_name_clean):
    print(f"INFO (processor.py): analyze_area called for '{area_name_clean}'.")
    if df_global.empty: return {"summary": "[Local Data]: Data unavailable for analysis.", "chart_data": None, "table_data": []}
    
    area_name_processed = area_name_clean.strip().title()
    filtered_df = df_global[df_global['Area'].str.lower() == area_name_processed.lower()].copy()

    if filtered_df.empty:
        return {"summary": f"[Local Data]: No specific data found for {area_name_processed}.", "chart_data": None, "table_data": []}

    filtered_df.sort_values(by=COL_YEAR, inplace=True, na_position='first')
    
    llm_context_cols = ['Area', COL_YEAR, COL_PRICE_RES, COL_DEMAND_TOTAL_SOLD, COL_RES_SOLD_COUNT] # Key cols for context
    existing_llm_cols = [col for col in llm_context_cols if col in filtered_df.columns]
    llm_context_df = filtered_df[existing_llm_cols].dropna(subset=[COL_YEAR]).sort_values(by=COL_YEAR, ascending=False).head(5) # Last 5 valid years
    context_data_str = llm_context_df.to_string(index=False) if not llm_context_df.empty else f"Limited recent data for {area_name_processed}."
    final_summary = generate_llm_summary(context_data_str, area_name_processed, analysis_type="general_area_analysis")

    chart_data = None
    if COL_PRICE_RES in filtered_df.columns and COL_YEAR in filtered_df.columns:
        chartable_df = filtered_df.dropna(subset=[COL_YEAR, COL_PRICE_RES])
        if not chartable_df.empty:
            chart_data = { "type": "price_trend", "labels": chartable_df[COL_YEAR].tolist(),
                           "datasets": [{"label": f"Res. Price - {area_name_processed}", "data": chartable_df[COL_PRICE_RES].tolist(),
                                         "borderColor": 'rgb(75, 192, 192)', "tension": 0.1, "fill": False}]}
    
    table_df_subset = get_relevant_table_columns(filtered_df)
    table_data = table_df_subset.fillna('N/A').to_dict(orient='records')
    return {"summary": final_summary, "chart_data": chart_data, "table_data": table_data}

def compare_demand_trends(area1_clean, area2_clean):
    print(f"INFO (processor.py): compare_demand_trends processing for '{area1_clean}' and '{area2_clean}'.")
    if df_global.empty:
        return {"summary": "[Local Data]: Data not loaded for comparison.", "chart_data": None, "table_data": []}

    critical_cols = [COL_DEMAND_TOTAL_SOLD, COL_YEAR, 'Area']
    if not all(col in df_global.columns for col in critical_cols):
         missing_cols_str = ", ".join([col for col in critical_cols if col not in df_global.columns])
         return {"summary": f"[Local Data]: Missing critical data columns ({missing_cols_str}) for comparison.", "chart_data": None, "table_data": []}

    df_area1 = df_global[df_global['Area'].str.lower() == area1_clean.lower()].copy()
    df_area2 = df_global[df_global['Area'].str.lower() == area2_clean.lower()].copy()

    if df_area1.empty and df_area2.empty:
        return {"summary": f"[Local Data]: No data found for either {area1_clean} or {area2_clean}.", "chart_data": None, "table_data": []}
    
    no_data_messages = []
    if df_area1.empty: no_data_messages.append(f"No data for {area1_clean}.")
    if df_area2.empty: no_data_messages.append(f"No data for {area2_clean}.")

    llm_context_parts = []
    def get_demand_summary_str_for_llm(df, name): 
        if df.empty or COL_DEMAND_TOTAL_SOLD not in df.columns: return f"{name}: No demand data."
        df_s = df.dropna(subset=[COL_YEAR, COL_DEMAND_TOTAL_SOLD]).sort_values(COL_YEAR)
        if df_s.empty: return f"{name}: No valid demand data points."
        avg_d = df_s[COL_DEMAND_TOTAL_SOLD].mean(); min_d = df_s[COL_DEMAND_TOTAL_SOLD].min(); max_d = df_s[COL_DEMAND_TOTAL_SOLD].max()
        latest_y = df_s[COL_YEAR].iloc[-1]; latest_d = df_s[COL_DEMAND_TOTAL_SOLD].iloc[-1]
        return f"For {name}: Latest demand in {latest_y} was {latest_d:,.0f}. Range: {min_d:,.0f}-{max_d:,.0f}, Avg: {avg_d:,.0f} units."

    if not df_area1.empty: llm_context_parts.append(get_demand_summary_str_for_llm(df_area1, area1_clean))
    if not df_area2.empty: llm_context_parts.append(get_demand_summary_str_for_llm(df_area2, area2_clean))
    context_data_str = "\n".join(llm_context_parts) if llm_context_parts else "Insufficient data for detailed comparison context."
    
    final_summary = generate_llm_summary(context_data_str, f"{area1_clean} vs {area2_clean}", analysis_type="demand_comparison")
    if no_data_messages: final_summary = " ".join(no_data_messages) + " " + final_summary

    chart_data = None; datasets_chart = []; all_years_list = []
    if not df_area1.empty and COL_YEAR in df_area1: all_years_list.extend(df_area1[COL_YEAR].dropna().tolist())
    if not df_area2.empty and COL_YEAR in df_area2: all_years_list.extend(df_area2[COL_YEAR].dropna().tolist())
    combined_years = sorted(list(set(map(int, filter(pd.notna, all_years_list))))) if all_years_list else []

    for i, (df_curr, name, color) in enumerate([(df_area1, area1_clean, 'rgb(255, 99, 132)'), (df_area2, area2_clean, 'rgb(54, 162, 235)')]):
        if not df_curr.empty and COL_DEMAND_TOTAL_SOLD in df_curr:
            df_s = df_curr.dropna(subset=[COL_YEAR, COL_DEMAND_TOTAL_SOLD]).sort_values(by=COL_YEAR)
            demand_map = pd.Series(df_s[COL_DEMAND_TOTAL_SOLD].values, index=df_s[COL_YEAR].values)
            points = [demand_map.get(year) for year in combined_years]
            datasets_chart.append({"label": f"Sold - {name}", "data": points, "borderColor": color, "tension": 0.1, "fill": False})
    if datasets_chart and combined_years: chart_data = {"type": "demand_comparison", "labels": combined_years, "datasets": datasets_chart}

    table_data = []; table_frames = [df for df in [df_area1, df_area2] if not df.empty]
    if table_frames:
        comb_df = pd.concat(table_frames).drop_duplicates().sort_values(['Area', COL_YEAR])
        table_data = get_relevant_table_columns(comb_df).fillna('N/A').to_dict(orient='records')
    return {"summary": final_summary, "chart_data": chart_data, "table_data": table_data}

def show_price_growth(area_clean, num_years):
    print(f"INFO (processor.py): show_price_growth processing for '{area_clean}' over {num_years} years.")
    if df_global.empty: return {"summary": "[Local Data]: Data unavailable for price growth.", "chart_data": None, "table_data": []}

    crit_cols = ['Area', COL_PRICE_RES, COL_YEAR]
    if not all(col in df_global.columns for col in crit_cols):
         missing_str = ", ".join([col for col in crit_cols if col not in df_global.columns])
         return {"summary": f"[Local Data]: Missing critical columns ({missing_str}) for price growth.", "chart_data": None, "table_data": []}

    df_orig = df_global[df_global['Area'].str.lower() == area_clean.lower()].copy()
    if df_orig.empty: return {"summary": f"[Local Data]: No data for {area_clean}.", "chart_data": None, "table_data": []}

    growth_df = df_orig.dropna(subset=[COL_YEAR, COL_PRICE_RES]) \
                       .sort_values(by=COL_YEAR, ascending=False) \
                       .head(num_years) \
                       .sort_values(by=COL_YEAR, ascending=True)
    if growth_df.empty or len(growth_df) < 1:
        return {"summary": f"[Local Data]: Not enough price data for {area_clean} for last {num_years} available years.", "chart_data": None, "table_data": []}

    llm_ctx_df = growth_df[['Area', COL_YEAR, COL_PRICE_RES]].copy()
    ctx_str = llm_ctx_df.to_string(index=False) if not llm_ctx_df.empty else f"Limited price data for {area_clean}."
    summary_llm = generate_llm_summary(ctx_str, area_clean, analysis_type="price_growth")
    
    manual_sum_part = ""
    if len(growth_df) > 1:
        ip, fp, iy, fy = growth_df[COL_PRICE_RES].iloc[0], growth_df[COL_PRICE_RES].iloc[-1], growth_df[COL_YEAR].iloc[0], growth_df[COL_YEAR].iloc[-1]
        if pd.notna(ip) and pd.notna(fp) and ip != 0:
            gp = ((fp - ip) / ip) * 100
            manual_sum_part = f" Prices changed by {gp:.2f}% (from {ip:,.0f} in {iy} to {fp:,.0f} in {fy})."
    final_summary = summary_llm + manual_sum_part

    chart_data = {"type": "price_trend", "labels": growth_df[COL_YEAR].tolist(),
                  "datasets": [{"label": f"Price Growth - {area_clean}", "data": growth_df[COL_PRICE_RES].tolist(),
                                "borderColor": 'rgb(153, 102, 255)', "tension": 0.1, "fill": False}]}
    table_data = get_relevant_table_columns(growth_df).fillna('N/A').to_dict(orient='records')
    return {"summary": final_summary, "chart_data": chart_data, "table_data": table_data}

# --- Main Query Processing Function ---
def process_query(user_query):
    if df_global.empty:
        msg_prefix = "[Local Data Check]: "
        startup_err_msg = "Data could not be loaded. Check server logs for startup errors."
        data_unavail_msg = "Real estate data file not configured or accessible."
        msg = startup_err_msg if (EXCEL_FILE_PATH and os.path.exists(EXCEL_FILE_PATH)) else data_unavail_msg
        return {"summary": msg_prefix + msg, "chart_data": None, "table_data": []}

    parsed_nlu = understand_query_with_gemini(user_query)

    if not parsed_nlu or parsed_nlu.get("intent") == "unknown" or not parsed_nlu.get("intent"):
        print(f"INFO (processor.py): NLU uncertain/failed for '{user_query}'. NLU: {parsed_nlu}. Trying regex fallback.")
        query_lower_fb = user_query.lower()
        match_analyze_fb = re.search(r"(?:analyze|analysis of|tell me about|info on)\s*(?:about|the\s+area\s+of\s*)?([\w\s]+?)(?:\s+real\s+estate|\s+market)?$", query_lower_fb)
        if match_analyze_fb:
            area_fb = match_analyze_fb.group(1).strip()
            print(f"INFO (processor.py): Regex fallback for 'analyze', area: '{area_fb}'")
            return analyze_area(area_fb)
        
      
        # match_compare_fb = re.search(r"compare ([\w\s]+?) and ([\w\s]+?) demand trends", query_lower_fb)
        # if match_compare_fb: ... return compare_demand_trends(...)
        # match_growth_fb = re.search(r"show price growth for ([\w\s]+?) over last (\d+) years", query_lower_fb)
        # if match_growth_fb: ... return show_price_growth(...)

        return {"summary": "[Query Error]: Couldn't understand request via LLM or basic parsing. Please rephrase.", "chart_data": None, "table_data": []}

    intent = parsed_nlu.get("intent")
    entities = parsed_nlu.get("entities", {})

    if intent == "analyze_single_area":
        area1 = entities.get("area_1")
        if area1 and isinstance(area1, str) and area1.strip(): return analyze_area(area1)
        else: 
            print(f"WARNING (processor.py): NLU intent '{intent}' but 'area_1' missing/invalid. Entities: {entities}")
            return {"summary": "[NLU Error]: Area for analysis not properly identified by LLM.", "chart_data": None, "table_data": []}

    elif intent == "compare_area_demand":
        area1 = entities.get("area_1"); area2 = entities.get("area_2")
        if area1 and isinstance(area1, str) and area1.strip() and \
           area2 and isinstance(area2, str) and area2.strip():
            return compare_demand_trends(area1, area2)
        else: 
            print(f"WARNING (processor.py): NLU intent '{intent}' but 'area_1' or 'area_2' missing/invalid. Entities: {entities}")
            return {"summary": "[NLU Error]: Both areas for comparison not properly identified by LLM.", "chart_data": None, "table_data": []}

    elif intent == "price_growth_area":
        area1 = entities.get("area_1"); years_val = entities.get("years")
        if area1 and isinstance(area1, str) and area1.strip() and years_val is not None:
            try:
                num_years = int(years_val)
                if num_years <= 0: raise ValueError("Years must be positive.")
                return show_price_growth(area1, num_years)
            except (ValueError, TypeError):
                print(f"WARNING (processor.py): NLU intent '{intent}' but 'years' ('{years_val}') is invalid. Entities: {entities}")
                return {"summary": f"[NLU Error]: Invalid number of years ('{years_val}') from LLM.", "chart_data": None, "table_data": []}
        else: 
            print(f"WARNING (processor.py): NLU intent '{intent}' but 'area_1' or 'years' missing/invalid. Entities: {entities}")
            return {"summary": "[NLU Error]: Area or years for price growth not properly identified by LLM.", "chart_data": None, "table_data": []}
    
    else:
        print(f"WARNING (processor.py): Unhandled NLU intent: {intent}. Entities: {entities}")
        return {"summary": "[Query Error]: Unrecognized command after LLM analysis.", "chart_data": None, "table_data": []}