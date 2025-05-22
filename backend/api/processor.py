import pandas as pd
import re
from django.conf import settings
import os
import numpy as np
import google.generativeai as genai
import traceback # For detailed error logging
import json # For parsing Gemini's structured output

# --- INITIALIZE df_global RIGHT AT THE TOP AFTER IMPORTS ---
df_global = pd.DataFrame()

# --- Column Name Constants (ADJUST THESE TO EXACTLY MATCH YOUR EXCEL HEADERS) ---
COL_AREA = 'final location'
COL_YEAR = 'year'
COL_CITY = 'city'
COL_LOC_LAT = 'loc_lat'
COL_LOC_LNG = 'loc_lng'
COL_TOTAL_SALES_VALUE = 'total_sales'
COL_DEMAND_TOTAL_SOLD = 'total sold' # General demand/sales (units)

# Property Type Specific Sales Counts
COL_RES_SOLD_COUNT = 'res_sold'
COL_OFFICE_SOLD_COUNT = 'office_sold'
COL_OTHERS_SOLD_COUNT = 'others_sold'
COL_SHOP_SOLD_COUNT = 'shop_sold'
COL_COMMERCIAL_SOLD_COUNT = 'commercial_sold'

# Property Type Specific Prices (Weighted Average RSF)
COL_PRICE_RES = 'weighted average rsf residential'
COL_PRICE_OFFICE = 'weighted average rsf office'
COL_PRICE_RETAIL = 'weighted average rsf retail' # For Shops
COL_PRICE_OTHERS = 'weighted average rsf others'

# Property Type Specific Prevailing Rents
COL_RENT_RES = 'prevailing rent residential'
COL_RENT_OFFICE = 'prevailing rent office'
COL_RENT_RETAIL = 'prevailing rent retail' # For Shops
COL_RENT_OTHERS = 'prevailing rent others'

# Property Type Specific Supply/Total Units
COL_TOTAL_UNITS_GENERAL = 'total units' # General total units
COL_NET_AREA_SUPPLIED = 'net area supplied' # General supply metric
COL_FLAT_TOTAL = 'flat total' # Residential
COL_SHOP_TOTAL = 'shop total' # Shops
COL_OFFICE_TOTAL = 'office total' # Offices
COL_OTHERS_TOTAL = 'others total' # Others

# --- Load settings variables and configure API ---
EXCEL_FILE_PATH = None
GOOGLE_API_KEY = None
gemini_model = None
try:
    EXCEL_FILE_PATH = settings.DATA_FILE_PATH
    if not EXCEL_FILE_PATH: print("WARNING (processor.py): DATA_FILE_PATH in settings is empty."); EXCEL_FILE_PATH = None
except AttributeError: print("CRITICAL ERROR (processor.py): DATA_FILE_PATH not in settings.")
try:
    GOOGLE_API_KEY = settings.GOOGLE_API_KEY
except AttributeError: print("WARNING (processor.py): GOOGLE_API_KEY not in settings.")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("INFO (processor.py): Gemini API configured.")
    except Exception as e: print(f"ERROR (processor.py): Failed to configure Gemini API: {e}"); traceback.print_exc()
else: print("INFO (processor.py): GOOGLE_API_KEY not set. LLM limited.")

# --- ACTUAL DATA LOADING LOGIC ---
if EXCEL_FILE_PATH and os.path.exists(EXCEL_FILE_PATH):
    try:
        print(f"INFO (processor.py): Loading Excel: {EXCEL_FILE_PATH}")
        df_temp = pd.read_excel(EXCEL_FILE_PATH, na_values=['#######', 'N/A', 'NA', '']) # Added more NA values
        if not df_temp.empty:
            if COL_AREA in df_temp.columns:
                df_temp.rename(columns={COL_AREA: 'Area'}, inplace=True)
                df_temp['Area'] = df_temp['Area'].astype(str).str.strip().str.title()
            else: print(f"WARNING (processor.py): Column '{COL_AREA}' not in Excel.")

            if 'Area' not in df_temp.columns: print("CRITICAL (processor.py): 'Area' column missing. df_global empty.")
            else:
                numeric_cols = {
                    COL_YEAR: 'Int64',
                    COL_LOC_LAT: 'float', COL_LOC_LNG: 'float',
                    COL_TOTAL_SALES_VALUE: 'float', COL_DEMAND_TOTAL_SOLD: 'float', # General sales/demand
                    COL_RES_SOLD_COUNT: 'Int64', COL_OFFICE_SOLD_COUNT: 'Int64',
                    COL_OTHERS_SOLD_COUNT: 'Int64', COL_SHOP_SOLD_COUNT: 'Int64',
                    COL_COMMERCIAL_SOLD_COUNT: 'Int64',
                    COL_PRICE_RES: 'float', COL_PRICE_OFFICE: 'float',
                    COL_PRICE_RETAIL: 'float', COL_PRICE_OTHERS: 'float',
                    COL_RENT_RES: 'float', COL_RENT_OFFICE: 'float',
                    COL_RENT_RETAIL: 'float', COL_RENT_OTHERS: 'float',
                    COL_TOTAL_UNITS_GENERAL: 'Int64', COL_NET_AREA_SUPPLIED: 'float',
                    COL_FLAT_TOTAL: 'Int64', COL_SHOP_TOTAL: 'Int64',
                    COL_OFFICE_TOTAL: 'Int64', COL_OTHERS_TOTAL: 'Int64'
                }
                for col, t_type in numeric_cols.items():
                    if col in df_temp.columns:
                        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                        if t_type=='Int64' and pd.api.types.is_numeric_dtype(df_temp[col]) and not df_temp[col].isnull().all():
                            df_temp[col] = df_temp[col].round().astype('Int64')
                        elif t_type=='float' and pd.api.types.is_numeric_dtype(df_temp[col]):
                             df_temp[col] = df_temp[col].astype('float') # Ensure it's float if intended
                if COL_CITY in df_temp.columns: df_temp[COL_CITY] = df_temp[COL_CITY].astype(str).str.strip().str.title()
                df_global = df_temp.copy()
                print("INFO (processor.py): DataFrame loaded. Shape:", df_global.shape)
                # print("INFO (processor.py): Loaded columns:", df_global.columns.tolist())
                # print("INFO (processor.py): Sample data (head):\n", df_global.head())

        else: print("WARNING (processor.py): Excel empty. df_global empty.")
    except Exception as e: print(f"CRITICAL ERROR (processor.py): Excel loading: {e}"); traceback.print_exc()
elif EXCEL_FILE_PATH: print(f"WARNING (processor.py): File {EXCEL_FILE_PATH} not found. df_global empty.")
else: print("WARNING (processor.py): EXCEL_FILE_PATH not configured. df_global empty.")

def get_known_area_names_for_prompt():
    if not df_global.empty and 'Area' in df_global.columns:
        ua = df_global['Area'].dropna().unique()
        if len(ua) > 0: return ", ".join(sorted(ua.tolist()))
    print("INFO (processor.py): Using fallback KNOWN_AREAS.")
    return "Wakad, Hinjewadi, Akurdi, Aundh, Ambegaon Budruk" # Fallback
KNOWN_AREAS = get_known_area_names_for_prompt()
print(f"INFO (processor.py): KNOWN_AREAS: {KNOWN_AREAS}")

# --- Configuration for Property Type Comparisons ---
PROPERTY_CONFIG = {
    "residential": {
        "display_name": "Residential",
        "sales_col": COL_RES_SOLD_COUNT,
        "price_col": COL_PRICE_RES,
        "rent_col": COL_RENT_RES,
        "supply_col": COL_FLAT_TOTAL,
    },
    "office": {
        "display_name": "Office",
        "sales_col": COL_OFFICE_SOLD_COUNT,
        "price_col": COL_PRICE_OFFICE,
        "rent_col": COL_RENT_OFFICE,
        "supply_col": COL_OFFICE_TOTAL,
    },
    "shop": {
        "display_name": "Shop/Retail",
        "sales_col": COL_SHOP_SOLD_COUNT,
        "price_col": COL_PRICE_RETAIL,
        "rent_col": COL_RENT_RETAIL,
        "supply_col": COL_SHOP_TOTAL,
    },
    "others": {
        "display_name": "Others",
        "sales_col": COL_OTHERS_SOLD_COUNT,
        "price_col": COL_PRICE_OTHERS,
        "rent_col": COL_RENT_OTHERS,
        "supply_col": COL_OTHERS_TOTAL,
    },
    "commercial": {
        "display_name": "Commercial",
        "sales_col": COL_COMMERCIAL_SOLD_COUNT,
        # price_col, rent_col, supply_col intentionally omitted as not in user's list for "commercial"
    }
}

METRIC_MAPPING = {
    "sales": {"key_in_prop_config": "sales_col", "agg_func": "sum", "display_name": "Total Sales (Units)"},
    "price": {"key_in_prop_config": "price_col", "agg_func": "mean", "display_name": "Avg. Price (RSF)"},
    "rent": {"key_in_prop_config": "rent_col", "agg_func": "mean", "display_name": "Avg. Prevailing Rent"},
    "supply": {"key_in_prop_config": "supply_col", "agg_func": "sum", "display_name": "Total Units (Supply)"}
}
DEFAULT_COMPARISON_METRICS = ["sales", "price", "rent", "supply"] # Default metrics if user doesn't specify

def get_relevant_table_columns(df_input):
    # This function is used by older analysis functions. Keep it for now or adapt.
    # For the new comparison, table columns are dynamically built.
    cols_map = {
        'Area': 'Area', COL_YEAR: 'Year', COL_CITY: 'City',
        COL_PRICE_RES: 'Avg. Res. Price', COL_DEMAND_TOTAL_SOLD: 'Total Sold',
        COL_RES_SOLD_COUNT: 'Res. Sold', COL_OFFICE_SOLD_COUNT: 'Office Sold',
        COL_SHOP_SOLD_COUNT: 'Shop Sold', COL_TOTAL_UNITS_GENERAL: 'Total Units',
        COL_PRICE_OFFICE: 'Avg. Office Price', COL_PRICE_RETAIL: 'Avg. Shop Price'
    }
    rel_df = pd.DataFrame()
    for orig_col, disp_name in cols_map.items():
        if orig_col in df_input.columns: rel_df[disp_name] = df_input[orig_col]
    return rel_df

# --- LLM NLU Function (understand_query_with_gemini) ---
def understand_query_with_gemini(user_query):
    if not gemini_model: print("WARNING (processor.py): Gemini NLU: Model unavailable."); return None
    
    prompt = f"""
    You are an expert real estate market query analyzer.
    Task: Analyze the user's query to identify intent and extract entities.
    Output: Return a single, valid JSON object ONLY. No extra text or markdown.

    Intents:
    - "analyze_single_area": General analysis of one area.
    - "compare_area_demand": Compare demand (total units sold) between two areas.
    - "price_growth_area": Price trends for one area over years (typically residential).
    - "evaluate_property_types": User asks to compare different property types (e.g., shop, office, residential) based on certain metrics (e.g., sales, price, rent, supply), possibly within an area or generally.
    - "unknown": If unclear or out of scope.

    Entities:
    - "area_1": (STRING, OPTIONAL) Primary area for context or filtering. Normalize using: {KNOWN_AREAS}.
    - "area_2": (STRING, OPTIONAL, only for compare_area_demand) Second area.
    - "years": (INTEGER, OPTIONAL, only for price_growth_area) Number of years.
    - "property_types": (LIST of STRINGS, REQUIRED for evaluate_property_types) List of property types. Normalize to singular: "residential", "office", "shop", "others", "commercial".
    - "metrics": (LIST of STRINGS, OPTIONAL for evaluate_property_types) List of metrics for comparison. Normalize to: "sales", "price", "rent", "supply". If not specified or user says "all aspects", you can omit "metrics" or set to ["all"].

    Known Property Types for Normalization: residential, office, shop, commercial, others.
    Known Metrics for Normalization: sales, price, rent, supply.

    Examples:
    User: "analyze wakad" -> {{"intent": "analyze_single_area", "entities": {{"area_1": "Wakad"}}}}
    User: "compare hinjewadi and ambegaon demand" -> {{"intent": "compare_area_demand", "entities": {{"area_1": "Hinjewadi", "area_2": "Ambegaon Budruk"}}}}
    User: "which is better for buying shop or office in wakad based on price and sales" -> {{"intent": "evaluate_property_types", "entities": {{"property_types": ["shop", "office"], "metrics": ["price", "sales"], "area_1": "Wakad"}}}}
    User: "shops or offices better to buy" -> {{"intent": "evaluate_property_types", "entities": {{"property_types": ["shop", "office"]}}}}
    User: "compare residential and commercial properties in Hinjewadi for all metrics" -> {{"intent": "evaluate_property_types", "entities": {{"property_types": ["residential", "commercial"], "metrics": ["all"], "area_1": "Hinjewadi"}}}}
    User: "how do rents for shops and offices stack up in Aundh?" -> {{"intent": "evaluate_property_types", "entities": {{"property_types": ["shop", "office"], "metrics": ["rent"], "area_1": "Aundh"}}}}
    User: "compare all aspects of residential and office properties" -> {{"intent": "evaluate_property_types", "entities": {{"property_types": ["residential", "office"], "metrics": ["all"]}}}}

    User Query: "{user_query}"
    JSON Response:
    """
    print(f"INFO (processor.py): NLU call for: '{user_query}'")
    try:
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        if response.parts:
            raw_json = response.text.strip()
            print(f"INFO (processor.py): NLU Raw: {raw_json}")
            # Clean the raw_json string
            if raw_json.startswith("```json"): raw_json = raw_json[7:]
            elif raw_json.startswith("```"): raw_json = raw_json[3:] # Handle if ``` only
            if raw_json.endswith("```"): raw_json = raw_json[:-3]
            
            parsed = json.loads(raw_json.strip())
            print(f"INFO (processor.py): NLU Parsed: {parsed}")
            if not isinstance(parsed, dict) or "intent" not in parsed:
                print(f"WARNING (processor.py): NLU malformed. Parsed: {parsed}")
                return {"intent": "unknown", "error": "LLM NLU malformed"}
            return parsed
        else: # Blocked or no candidates
            reason = "Unknown";
            if response.prompt_feedback and response.prompt_feedback.block_reason: reason = response.prompt_feedback.block_reason.name
            print(f"WARNING (processor.py): NLU no parts. Block: {reason}. Feedback: {response.prompt_feedback}")
            return {"intent": "unknown", "error": f"LLM NLU no parts, reason: {reason}"}
    except json.JSONDecodeError as jde:
        raw_json_err = 'Not available';
        if 'raw_json' in locals(): raw_json_err = raw_json
        print(f"ERROR (processor.py): NLU JSONDecodeError: {jde}. Raw: '{raw_json_err}'")
        return {"intent": "unknown", "error": "LLM NLU JSON decode error"}
    except Exception as e:
        print(f"ERROR (processor.py): NLU Exception: {e}"); traceback.print_exc()
        return {"intent": "unknown", "error": "LLM NLU API exception"}

# --- LLM Summarization Function (generate_llm_summary) ---
def generate_llm_summary(context_data_str, display_name_or_query, analysis_type="general"):
    llm_prefix = "[Answered by Ravilesh-LLM]: "; mock_base = f"Summary for {display_name_or_query} based on local data."
    mock_prefixes = {"not_configured": "[Mocked - LLM Not Configured]: ", "api_fail": "[Mocked - LLM API Fail]: ", "no_parts": "[Mocked - LLM Issue (No Parts)]:"}
    if not gemini_model: return mock_prefixes["not_configured"] + mock_base
    prompt_map = {
        "general_area_analysis": f"Concise (2-3 sentences) real estate market summary for {display_name_or_query}, focusing on residential prices ('{COL_PRICE_RES}') and sales volume ('{COL_DEMAND_TOTAL_SOLD}'). Data:\n{context_data_str}",
        "price_growth": f"Summarize residential price growth for {display_name_or_query} ('{COL_PRICE_RES}') based on this data:\n{context_data_str}\nGive a 1-2 sentence insight.",
        "demand_comparison": f"Compare real estate demand ('{COL_DEMAND_TOTAL_SOLD}') for {display_name_or_query} using this context:\n{context_data_str}\nOffer a brief 1-2 sentence comparative insight.",
        "evaluate_property_types": f"User asked about comparing property types: '{display_name_or_query}'. Based on the following data summary, provide insights on how the requested property types compare on the given metrics. Mention key supporting data points. If data is inconclusive or missing for some types/metrics, state that. Be concise.\nData Summary:\n{context_data_str}"
    }
    prompt = prompt_map.get(analysis_type)
    if not prompt: return f"{mock_prefixes['not_configured']}Unknown analysis type '{analysis_type}' for {display_name_or_query}."
    print(f"INFO (processor.py): LLM summary call: {display_name_or_query}, Type: {analysis_type}")
    try:
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]] # Adjusted safety
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        if response.parts: return llm_prefix + response.text.strip()
        else:
            reason = "Unknown";
            if response.prompt_feedback and response.prompt_feedback.block_reason: reason = response.prompt_feedback.block_reason.name
            print(f"WARNING (processor.py): LLM summary no parts for {display_name_or_query}. Block: {reason}.")
            return f"{mock_prefixes['no_parts']}Summary for {display_name_or_query} ({reason})."
    except Exception as e:
        print(f"ERROR (processor.py): LLM summary Exception for {display_name_or_query}: {e}"); traceback.print_exc()
        return f"{mock_prefixes['api_fail']}" + mock_base

# --- Core Analysis Functions (analyze_area, compare_demand_trends, show_price_growth - keep as is) ---
def analyze_area(area_name_clean):
    print(f"INFO (processor.py): analyze_area for '{area_name_clean}'.")
    if df_global.empty: return {"summary": "[Local Data]: Data unavailable.", "chart_data": None, "table_data": []}
    area_name_proc = area_name_clean.strip().title()
    filt_df = df_global[df_global['Area'].str.lower() == area_name_proc.lower()].copy()
    if filt_df.empty: return {"summary": f"[Local Data]: No data for {area_name_proc}.", "chart_data": None, "table_data": []}
    filt_df.sort_values(by=COL_YEAR, inplace=True, na_position='first')
    llm_ctx_cols = ['Area', COL_YEAR, COL_PRICE_RES, COL_DEMAND_TOTAL_SOLD, COL_RES_SOLD_COUNT]
    exist_cols = [c for c in llm_ctx_cols if c in filt_df.columns]; llm_ctx_df = filt_df[exist_cols].dropna(subset=[COL_YEAR]).sort_values(by=COL_YEAR, ascending=False).head(5)
    ctx_str = llm_ctx_df.to_string(index=False) if not llm_ctx_df.empty else f"Limited data for {area_name_proc}."
    fin_sum = generate_llm_summary(ctx_str, area_name_proc, "general_area_analysis")
    chart_data = None
    if COL_PRICE_RES in filt_df.columns and COL_YEAR in filt_df.columns:
        chart_df = filt_df.dropna(subset=[COL_YEAR, COL_PRICE_RES])
        if not chart_df.empty: chart_data = {"type": "price_trend", "labels": chart_df[COL_YEAR].tolist(), "datasets": [{"label": f"Res. Price - {area_name_proc}", "data": chart_df[COL_PRICE_RES].tolist(), "borderColor": 'rgb(75,192,192)', "tension": 0.1, "fill": False}]}
    tbl_subset = get_relevant_table_columns(filt_df); tbl_data = tbl_subset.fillna('N/A').to_dict(orient='records')
    return {"summary": fin_sum, "chart_data": chart_data, "table_data": tbl_data}

def compare_demand_trends(area1_clean, area2_clean):
    print(f"INFO (processor.py): compare_demand_trends for '{area1_clean}' vs '{area2_clean}'.")
    if df_global.empty: return {"summary": "[Local Data]: Data unavailable.", "chart_data": None, "table_data": []}
    crit_cols = [COL_DEMAND_TOTAL_SOLD, COL_YEAR, 'Area'] # COL_DEMAND_TOTAL_SOLD is general total sold units
    if not all(c in df_global.columns for c in crit_cols): return {"summary": f"[Local Data]: Missing critical columns for demand comparison.", "chart_data": None, "table_data": []}
    df_a1 = df_global[df_global['Area'].str.lower() == area1_clean.lower()].copy(); df_a2 = df_global[df_global['Area'].str.lower() == area2_clean.lower()].copy()
    if df_a1.empty and df_a2.empty: return {"summary": f"[Local Data]: No data for {area1_clean} or {area2_clean}.", "chart_data": None, "table_data": []}
    no_data_msgs = [f"No data for {name}." for name, df_a in [(area1_clean, df_a1), (area2_clean, df_a2)] if df_a.empty]
    llm_ctx_parts = []
    def get_dem_sum_str(df, name):
        if df.empty or COL_DEMAND_TOTAL_SOLD not in df.columns: return f"{name}: No demand data."
        df_s = df.dropna(subset=[COL_YEAR, COL_DEMAND_TOTAL_SOLD]).sort_values(COL_YEAR)
        if df_s.empty: return f"{name}: No valid demand points."
        avg, mi, ma = df_s[COL_DEMAND_TOTAL_SOLD].mean(), df_s[COL_DEMAND_TOTAL_SOLD].min(), df_s[COL_DEMAND_TOTAL_SOLD].max()
        ly_data = df_s.iloc[-1]; ly, ld = ly_data[COL_YEAR], ly_data[COL_DEMAND_TOTAL_SOLD]
        return f"{name}: Latest demand ({ly}): {ld:,.0f}. Range: {mi:,.0f}-{ma:,.0f}, Avg: {avg:,.0f} units."
    if not df_a1.empty: llm_ctx_parts.append(get_dem_sum_str(df_a1, area1_clean))
    if not df_a2.empty: llm_ctx_parts.append(get_dem_sum_str(df_a2, area2_clean))
    ctx_str = "\n".join(llm_ctx_parts) if llm_ctx_parts else "Insufficient data for comparison context."
    fin_sum = generate_llm_summary(ctx_str, f"{area1_clean} vs {area2_clean}", "demand_comparison")
    if no_data_msgs: fin_sum = " ".join(no_data_msgs) + " " + fin_sum
    chart_data = None; datasets_chart = []; all_years = []
    if not df_a1.empty and COL_YEAR in df_a1: all_years.extend(df_a1[COL_YEAR].dropna().tolist())
    if not df_a2.empty and COL_YEAR in df_a2: all_years.extend(df_a2[COL_YEAR].dropna().tolist())
    comb_years = sorted(list(set(map(int, filter(pd.notna, all_years))))) if all_years else []
    for df_curr, name, color in [(df_a1, area1_clean, 'rgb(255,99,132)'), (df_a2, area2_clean, 'rgb(54,162,235)')]:
        if not df_curr.empty and COL_DEMAND_TOTAL_SOLD in df_curr:
            df_s = df_curr.dropna(subset=[COL_YEAR, COL_DEMAND_TOTAL_SOLD]).sort_values(by=COL_YEAR)
            dem_map = pd.Series(df_s[COL_DEMAND_TOTAL_SOLD].values, index=df_s[COL_YEAR].values)
            points = [dem_map.get(y) for y in comb_years]; datasets_chart.append({"label": f"Total Sold Units - {name}", "data": points, "borderColor": color, "tension": 0.1, "fill": False})
    if datasets_chart and comb_years: chart_data = {"type": "demand_comparison", "labels": comb_years, "datasets": datasets_chart}
    tbl_data = []; tbl_frames = [df for df in [df_a1, df_a2] if not df.empty]
    if tbl_frames: comb_df = pd.concat(tbl_frames).drop_duplicates().sort_values(['Area', COL_YEAR]); tbl_data = get_relevant_table_columns(comb_df).fillna('N/A').to_dict(orient='records')
    return {"summary": fin_sum, "chart_data": chart_data, "table_data": tbl_data}

def show_price_growth(area_clean, num_years):
    print(f"INFO (processor.py): show_price_growth for '{area_clean}' over {num_years} years.")
    if df_global.empty: return {"summary": "[Local Data]: Data unavailable.", "chart_data": None, "table_data": []}
    crit_cols = ['Area', COL_PRICE_RES, COL_YEAR];
    if not all(c in df_global.columns for c in crit_cols): return {"summary": f"[Local Data]: Missing critical columns.", "chart_data": None, "table_data": []}
    df_orig = df_global[df_global['Area'].str.lower() == area_clean.lower()].copy()
    if df_orig.empty: return {"summary": f"[Local Data]: No data for {area_clean}.", "chart_data": None, "table_data": []}
    growth_df = df_orig.dropna(subset=[COL_YEAR, COL_PRICE_RES]).sort_values(by=COL_YEAR,ascending=False).head(num_years).sort_values(by=COL_YEAR,ascending=True)
    if growth_df.empty: return {"summary": f"[Local Data]: Not enough price data for {area_clean} for last {num_years} years.", "chart_data": None, "table_data": []}
    llm_ctx_df = growth_df[['Area', COL_YEAR, COL_PRICE_RES]].copy(); ctx_str = llm_ctx_df.to_string(index=False) if not llm_ctx_df.empty else f"Limited price data for {area_clean}."
    sum_llm = generate_llm_summary(ctx_str, area_clean, "price_growth")
    man_sum_part = ""
    if len(growth_df) > 1:
        ip, fp, iy, fy = growth_df[COL_PRICE_RES].iloc[0], growth_df[COL_PRICE_RES].iloc[-1], growth_df[COL_YEAR].iloc[0], growth_df[COL_YEAR].iloc[-1]
        if pd.notna(ip) and pd.notna(fp) and ip != 0: gp = ((fp - ip) / ip) * 100; man_sum_part = f" Prices changed by {gp:.2f}% (from {ip:,.0f} in {iy} to {fp:,.0f} in {fy})."
    fin_sum = sum_llm + man_sum_part
    chart_data = {"type": "price_trend", "labels": growth_df[COL_YEAR].tolist(), "datasets": [{"label": f"Price Growth - {area_clean}", "data": growth_df[COL_PRICE_RES].tolist(), "borderColor": 'rgb(153,102,255)', "tension": 0.1, "fill": False}]}
    tbl_data = get_relevant_table_columns(growth_df).fillna('N/A').to_dict(orient='records')
    return {"summary": fin_sum, "chart_data": chart_data, "table_data": tbl_data}

# --- NEW Enhanced Handler Function for Property Type Comparison ---
def compare_property_data(property_types_query, metrics_query, target_area_query=None, user_query_for_llm="property type comparison"):
    print(f"INFO (processor.py): compare_property_data for types: {property_types_query}, metrics: {metrics_query}, area: {target_area_query}")
    if df_global.empty:
        return {"summary": "[Local Data]: Data unavailable for property type comparison.", "chart_data": None, "table_data": []}

    # Normalize property types from query (e.g., "shops" -> "shop")
    normalized_prop_types = []
    if property_types_query:
        for pt in property_types_query:
            pt_lower = pt.lower()
            if pt_lower in PROPERTY_CONFIG: # Exact match after lowercasing
                normalized_prop_types.append(pt_lower)
            elif pt_lower.rstrip('s') in PROPERTY_CONFIG: # Simple singularization
                normalized_prop_types.append(pt_lower.rstrip('s'))
            # else: print(f"Warning: Property type '{pt}' not recognized in PROPERTY_CONFIG.")
    if not normalized_prop_types:
        return {"summary": "[Input Error]: No valid property types provided for comparison.", "chart_data": None, "table_data": []}

    # Determine metrics to compare
    actual_metrics_to_compare = []
    if metrics_query and metrics_query != ["all"]:
        for mq in metrics_query:
            if mq.lower() in METRIC_MAPPING:
                actual_metrics_to_compare.append(mq.lower())
            # else: print(f"Warning: Metric '{mq}' not recognized in METRIC_MAPPING.")
    else: # "all" or no metrics specified means use defaults
        actual_metrics_to_compare = [m for m in DEFAULT_COMPARISON_METRICS if m in METRIC_MAPPING]

    if not actual_metrics_to_compare:
        return {"summary": "[Input Error]: No valid metrics specified or derivable for comparison.", "chart_data": None, "table_data": []}

    table_rows = []
    llm_context_parts = []
    
    areas_to_process = []
    if target_area_query:
        processed_target_area = target_area_query.strip().title()
        if processed_target_area in KNOWN_AREAS.split(", "): # Check against known areas for validity
             areas_to_process.append(processed_target_area)
        else:
            # If target area is specified but not found, perhaps it's better to return no data for that area
            # or search all areas. For now, let's assume if specified, it must be valid.
            # If KNOWN_AREAS is just a few examples, this check might be too restrictive.
            # For broader search, remove this validity check or expand KNOWN_AREAS.
            print(f"WARNING: Target area '{processed_target_area}' not in KNOWN_AREAS. Processing all areas if any.")
            # Decide: either process all, or return error. Let's process all if target is not in a strict list.
            # areas_to_process = sorted(df_global['Area'].dropna().unique().tolist())
            # For now, strict: if target_area_query is given, it must be processable.
            # If it's not in a pre-vetted list, we might want to state that the specific area data is missing.
            # Let's assume for now `target_area_query` if provided, is a valid area name from the data.
            areas_to_process.append(processed_target_area) # Use as is, filtering will handle if it exists.
    else:
        areas_to_process = sorted(df_global['Area'].dropna().unique().tolist())

    if not areas_to_process:
         return {"summary": f"[Local Data]: No areas found to process the comparison.", "chart_data": None, "table_data": []}


    for area_name in areas_to_process:
        area_df = df_global[df_global['Area'].str.lower() == area_name.lower()]
        if area_df.empty:
            if target_area_query: # If a specific area was requested and not found
                 llm_context_parts.append(f"\nNo data found for the area: {area_name}.")
            continue

        area_has_data = False
        for prop_type_key in normalized_prop_types:
            if prop_type_key not in PROPERTY_CONFIG:
                continue # Skip if property type is somehow not in our config

            prop_config = PROPERTY_CONFIG[prop_type_key]
            row_data = {"Area": area_name, "Property Type": prop_config["display_name"]}
            llm_prop_summary = [f"  {prop_config['display_name']}:"]
            
            data_found_for_prop_type_in_area = False
            for metric_key in actual_metrics_to_compare:
                metric_conf = METRIC_MAPPING[metric_key]
                col_key_in_prop = metric_conf["key_in_prop_config"]
                
                actual_col_name = prop_config.get(col_key_in_prop)
                metric_display_name = metric_conf["display_name"]
                
                val_to_display = "N/A"
                if actual_col_name and actual_col_name in area_df.columns and not area_df[actual_col_name].dropna().empty:
                    series = area_df[actual_col_name].dropna()
                    agg_val = np.nan
                    if metric_conf["agg_func"] == "mean":
                        agg_val = series.mean()
                    elif metric_conf["agg_func"] == "sum":
                        agg_val = series.sum()
                    
                    if pd.notna(agg_val):
                        val_to_display = f"{agg_val:,.0f}" # Format as integer for now, floats for price/rent might be better with decimals
                        if "price" in metric_key or "rent" in metric_key :
                             val_to_display = f"{agg_val:,.2f}" # two decimal places for price/rent
                        llm_prop_summary.append(f"{metric_display_name}: {val_to_display}")
                        data_found_for_prop_type_in_area = True
                        area_has_data = True
                
                row_data[metric_display_name] = val_to_display

            if data_found_for_prop_type_in_area: # Only add to LLM context if some data was found for this prop type
                 if len(llm_prop_summary) > 1: # more than just display name
                    llm_context_parts.append(f"\nIn {area_name}:\n" + "\n    ".join(llm_prop_summary))
            
            # Add row to table if any metric has data or to show N/A structure
            # Only add if at least one metric value was attempted to be filled (even if N/A)
            if len(row_data) > 2: # Has Area, Property Type, and at least one metric column
                table_rows.append(row_data)
        
    if not table_rows and not llm_context_parts: # Check if any data was processed at all
         no_data_message = "[Local Data]: Insufficient data for the requested property types and metrics"
         if target_area_query:
             no_data_message += f" in {target_area_query}."
         else:
             no_data_message += "."
         return {"summary": no_data_message, "chart_data": None, "table_data": []}


    final_summary_context = "\n".join(llm_context_parts) if llm_context_parts else "No specific data points found for comparison context."
    final_summary = generate_llm_summary(final_summary_context, user_query_for_llm, analysis_type="evaluate_property_types")

    # Chart Data Generation (Simple bar chart for one metric across property types in a specific area)
    chart_data = None
    if target_area_query and table_rows:
        # Filter table_rows for the target_area (already done by areas_to_process if target_area_query was specific)
        # For chart, let's pick the first metric from actual_metrics_to_compare that has valid data
        
        chart_metric_display_name = None
        first_metric_key = actual_metrics_to_compare[0]
        chart_metric_display_name = METRIC_MAPPING[first_metric_key]["display_name"]

        chart_labels = []
        chart_dataset_data = []
        
        # Ensure we are only using data for the target area for the chart
        # (table_rows might contain multiple areas if target_area_query was None initially and then defaulted)
        # However, current logic means areas_to_process has one item if target_area_query is set.
        
        temp_chart_data_points = {} # Store {prop_type_display_name: value}

        for row in table_rows:
            if row["Area"].lower() == target_area_query.strip().title().lower(): # Ensure correct area
                prop_type_display = row["Property Type"]
                metric_val_str = row.get(chart_metric_display_name, "N/A")
                try:
                    # Convert back from formatted string to float for charting
                    metric_val = float(metric_val_str.replace(",", "")) if metric_val_str != "N/A" else np.nan
                    if pd.notna(metric_val):
                        temp_chart_data_points[prop_type_display] = metric_val
                except ValueError:
                    pass # Keep as N/A or skip

        # Sort property types for consistent chart order if desired, e.g., by PROPERTY_CONFIG order
        sorted_prop_display_names_for_chart = [
            PROPERTY_CONFIG[ptk]["display_name"] for ptk in normalized_prop_types 
            if PROPERTY_CONFIG[ptk]["display_name"] in temp_chart_data_points
        ]

        for prop_disp_name in sorted_prop_display_names_for_chart:
            chart_labels.append(prop_disp_name)
            chart_dataset_data.append(temp_chart_data_points[prop_disp_name])

        if chart_labels and chart_dataset_data:
            chart_data = {
                "type": "bar", # Bar chart for comparing property types on one metric
                "labels": chart_labels,
                "datasets": [{
                    "label": f"{chart_metric_display_name} in {target_area_query.strip().title()}",
                    "data": chart_dataset_data,
                    # Add colors later if needed
                    "backgroundColor": [
                        'rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 206, 86, 0.5)', 'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)', 'rgba(255, 159, 64, 0.5)'
                    ] * (len(chart_labels) // 6 + 1) # Cycle colors
                }]
            }

    return {"summary": final_summary, "chart_data": chart_data, "table_data": table_rows}


# --- Main Query Processing Function ---
def process_query(user_query):
    if df_global.empty:
        msg_prefix = "[Local Data Check]: "; startup_err = "Data load failed. Check logs."
        data_unavail = "Data file not configured/accessible."
        msg = startup_err if (EXCEL_FILE_PATH and os.path.exists(EXCEL_FILE_PATH)) else data_unavail
        return {"summary": msg_prefix + msg, "chart_data": None, "table_data": []}

    parsed_nlu = understand_query_with_gemini(user_query)

    if not parsed_nlu or not parsed_nlu.get("intent") or parsed_nlu.get("intent") == "unknown":
        nlu_error_msg = parsed_nlu.get("error", "Could not understand the request via LLM.") if parsed_nlu else "LLM NLU parsing failed."
        print(f"INFO (processor.py): NLU uncertain/failed for '{user_query}'. NLU Error: {nlu_error_msg}. Trying regex fallback.")
        # Regex fallback (simplified, you might want to enhance this or remove if NLU is reliable)
        query_lower_fb = user_query.lower()
        match_analyze_fb = re.search(r"(?:analyze|analysis of|tell me about|info on)\s*(?:about|the\s+area\s+of\s*)?([\w\s]+?)(?:\s+real\s+estate|\s+market)?$", query_lower_fb)
        if match_analyze_fb:
            area_fb = match_analyze_fb.group(1).strip().title()
            if area_fb: print(f"INFO (processor.py): Regex fallback 'analyze', area: '{area_fb}'"); return analyze_area(area_fb)
        
        # Default error if NLU fails and regex doesn't match
        return {"summary": f"[Query Error]: {nlu_error_msg}. Try 'Analyze [Area]', 'Compare [Area1] and [Area2] demand', 'Price growth for [Area] over N years', or 'Compare shops and offices in [Area] by price and sales'.",
                "chart_data": None, "table_data": []}

    intent = parsed_nlu.get("intent")
    entities = parsed_nlu.get("entities", {})

    if intent == "analyze_single_area":
        area1 = entities.get("area_1")
        if area1 and isinstance(area1, str) and area1.strip(): return analyze_area(area1)
        else: print(f"W (proc): NLU '{intent}' but 'area_1' invalid. Ent: {entities}"); return {"summary": "[NLU Error]: Area for analysis not identified.", "chart_data": None, "table_data": []}
    
    elif intent == "compare_area_demand":
        area1, area2 = entities.get("area_1"), entities.get("area_2")
        if area1 and isinstance(area1, str) and area1.strip() and \
           area2 and isinstance(area2, str) and area2.strip(): return compare_demand_trends(area1, area2)
        else: print(f"W (proc): NLU '{intent}' but areas invalid. Ent: {entities}"); return {"summary": "[NLU Error]: Both areas for comparison not identified.", "chart_data": None, "table_data": []}

    elif intent == "price_growth_area":
        area1, years_val = entities.get("area_1"), entities.get("years")
        if area1 and isinstance(area1, str) and area1.strip() and years_val is not None:
            try:
                num_years = int(years_val)
                if num_years <= 0: raise ValueError("Years positive")
                return show_price_growth(area1, num_years)
            except (ValueError, TypeError): print(f"W (proc): NLU '{intent}' but years '{years_val}' invalid. Ent: {entities}"); return {"summary": f"[NLU Error]: Invalid years '{years_val}'.", "chart_data": None, "table_data": []}
        else: print(f"W (proc): NLU '{intent}' but area/years invalid. Ent: {entities}"); return {"summary": "[NLU Error]: Area or years for price growth not identified.", "chart_data": None, "table_data": []}
    
    elif intent == "evaluate_property_types":
        property_types = entities.get("property_types")
        target_area = entities.get("area_1") # Optional
        metrics = entities.get("metrics")     # Optional, NLU might set to ["all"] or omit

        if property_types and isinstance(property_types, list) and len(property_types) > 0:
            return compare_property_data(property_types, metrics, target_area, user_query_for_llm=user_query)
        else:
            print(f"WARNING (processor.py): NLU intent '{intent}' but 'property_types' missing/invalid. Entities: {entities}")
            return {"summary": "[NLU Error]: Property types for evaluation not identified by LLM.", "chart_data": None, "table_data": []}
            
    else:
        print(f"WARNING (processor.py): Unhandled NLU intent: {intent}. Entities: {entities}")
        return {"summary": "[Query Error]: Unrecognized command after LLM analysis.", "chart_data": None, "table_data": []}