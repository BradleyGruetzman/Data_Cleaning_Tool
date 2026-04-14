import pandas as pd
import tkinter as tk
import os
import json
from tkinter import filedialog, scrolledtext
from tkinter import ttk
from openai import OpenAI

selected_file = None
selected_column = None
last_ai_summary = None
action_history = []
undo_stack = []

# ------------------------
# FUNCTIONS
# ------------------------

def load_file():
    global selected_file

    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not file_path:
        return

    selected_file = pd.read_csv(file_path)
    undo_stack.clear()

    # Populate column list
    column_listbox.delete(0, tk.END)
    for col in selected_file.columns:
        column_listbox.insert(tk.END, col)

    update_preview_table(selected_file)

    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, f"Loaded file: {file_path}\n")
    log_action(f"Loaded file: {file_path}")

def log_action(message):
    action_history.append(message)
    history_box.config(state="normal")
    history_box.insert(tk.END, message + "\n")
    history_box.see(tk.END)
    history_box.config(state="disabled")

def save_state():
    global selected_file, undo_stack

    if selected_file is not None:
        undo_stack.append(selected_file.copy(deep=True))

def undo_last_change():
    global selected_file, selected_column

    if not undo_stack:
        output_box.insert(tk.END, "\nNo changes to undo.\n")
        return

    selected_file = undo_stack.pop()
    selected_column = None

    refresh_column_list()
    update_preview_table(selected_file)

    output_box.insert(tk.END, "\nUndid last change.\n")
    log_action("Undid last change")


def file_summary():
    global selected_file
    output_box.delete(1.0, tk.END)

    if selected_file is None:
        output_box.insert(tk.END, "No file loaded\n")
        return

    df = selected_file

    output_box.insert(tk.END, "Dataset Overview\n", "header")
    output_box.insert(tk.END, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")

    output_box.insert(tk.END, "Column Names and Types:\n", "header")
    for col, dtype in df.dtypes.items():
        output_box.insert(tk.END, f"  {col}: {dtype}\n")
    output_box.insert(tk.END, "\n")

    output_box.insert(tk.END, "Missing Values per Column:\n", "header")
    for col, nulls in df.isnull().sum().items():
        output_box.insert(tk.END, f"  {col}: {nulls}\n")
    output_box.insert(tk.END, "\n")

    output_box.insert(tk.END, "Numerical Summary:\n", "header")
    output_box.insert(tk.END, str(df.describe()) + "\n\n")

    output_box.insert(tk.END, "Categorical Summary:\n", "header")

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        output_box.insert(tk.END, f"\n{col}\n", "subheader")
        output_box.insert(tk.END, f"  Count: {df[col].count()}\n")
        output_box.insert(tk.END, f"  Unique: {df[col].nunique()}\n")

        mode_series = df[col].mode()
        if not mode_series.empty:
            top_value = mode_series.iloc[0]
            top_freq = df[col].value_counts().iloc[0]
            output_box.insert(tk.END, f"  Top: {top_value}\n")
            output_box.insert(tk.END, f"  Frequency: {top_freq}\n")

    output_box.insert(tk.END, "Unique Values per Column:\n", "header")
    for col, unique in df.nunique().items():
        output_box.insert(tk.END, f"  {col}: {unique}\n")
    output_box.insert(tk.END, "\n")

    output_box.insert(tk.END, "Most Frequent Values (Mode):\n", "header")
    output_box.insert(tk.END, str(df.mode().iloc[0]) + "\n")


def build_dataset_profile(df):
    rows, cols = df.shape

    missing_counts = df.isnull().sum()
    missing_percent = ((missing_counts / len(df)) * 100).round(2) if len(df) > 0 else missing_counts

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    profile = {
        "rows": int(rows),
        "columns": int(cols),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_counts": missing_counts.to_dict(),
        "missing_percent": missing_percent.to_dict(),
        "unique_counts": df.nunique(dropna=True).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "columns_with_high_missing": [
            col for col in df.columns if len(df) > 0 and (df[col].isnull().mean() * 100) >= 30
        ],
        "single_value_columns": [
            col for col in df.columns if df[col].nunique(dropna=True) <= 1
        ],
        "likely_id_columns": [
            col for col in df.columns if len(df) > 0 and df[col].nunique(dropna=True) == len(df)
        ],
        "top_values": {},
        "numeric_summary": {}
    }

    for col in categorical_cols[:12]:
        value_counts = df[col].value_counts(dropna=False).head(3)
        cleaned = {}
        for k, v in value_counts.items():
            key = "NULL" if pd.isna(k) else str(k)
            cleaned[key] = int(v)
        profile["top_values"][col] = cleaned

    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        for col in numeric_cols[:12]:
            if col in desc.columns:
                profile["numeric_summary"][col] = {
                    stat: (None if pd.isna(val) else float(val))
                    for stat, val in desc[col].items()
                }

    return profile

def generate_ai_summary():
    global selected_file, last_ai_summary

    if selected_file is None:
        output_box.insert(tk.END, "\nNo file loaded.\n")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        output_box.insert(
            tk.END,
            "\nAI summary unavailable. Set OPENAI_API_KEY in your environment to enable this feature.\n"
        )
        return

    try:
        profile = build_dataset_profile(selected_file)

        prompt = f"""
You are helping a beginner analyst understand a dataset.

Using ONLY the structured dataset profile below:
1. Write a short dataset overview in plain English.
2. List the 3 most important data quality issues.
3. Recommend 3 practical cleaning steps.
4. Suggest 2 useful analysis ideas.

Rules:
- Be concrete and concise.
- Do not invent facts that are not in the profile.
- If an issue is not present, do not mention it.
- Format the response with these headers exactly:
Dataset Overview
Key Data Quality Issues
Recommended Cleaning Steps
Possible Analysis Ideas

Dataset profile:
{json.dumps(profile, indent=2)}
""".strip()

        output_box.delete(1.0, tk.END)
        output_box.insert(tk.END, "Generating AI summary...\n")

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=prompt
        )

        ai_text = response.output_text.strip()
        last_ai_summary = ai_text

        output_box.delete(1.0, tk.END)
        output_box.insert(tk.END, "AI Dataset Summary\n", "header")
        output_box.insert(tk.END, ai_text + "\n")

        log_action("Generated AI summary")

    except Exception as e:
        output_box.insert(tk.END, f"\nError generating AI summary: {e}\n")

def show_last_ai_summary():
    global last_ai_summary

    output_box.delete(1.0, tk.END)

    if not last_ai_summary:
        output_box.insert(tk.END, "No AI summary has been generated yet.\n")
        return

    output_box.insert(tk.END, "AI Dataset Summary\n", "header")
    output_box.insert(tk.END, last_ai_summary + "\n")

def on_column_select(event):
    global selected_file, selected_column

    if selected_file is None:
        return

    selection = column_listbox.curselection()
    if not selection:
        return

    selected_column = column_listbox.get(selection[0])
    col = selected_file[selected_column]

    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, f"Column: {selected_column}\n", "header")
    output_box.insert(tk.END, f"Type: {col.dtype}\n")
    output_box.insert(tk.END, f"Total Values: {len(col)}\n")
    output_box.insert(tk.END, f"Missing: {col.isnull().sum()} ({col.isnull().mean()*100:.1f}%)\n")
    output_box.insert(tk.END, f"Unique Values: {col.nunique()}\n\n")
    output_box.insert(tk.END, "Top Values:\n", "header")
    output_box.insert(tk.END, str(col.value_counts().head()) + "\n\n")


def apply_missing_data_action(method, custom_value, popup):
    global selected_file, selected_column

    if selected_file is None or selected_column is None:
        output_box.insert(tk.END, "\nNo file or column selected.\n")
        return

    col = selected_column
    df = selected_file

    save_state()

    try:
        if method == "Fill with Mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
                output_box.insert(tk.END, f"\nFilled missing values in '{col}' with mean.\n")
                log_action(f"Filled missing values in '{col}' with mean")
            else:
                output_box.insert(tk.END, f"\nCannot use mean on non-numeric column '{col}'.\n")

        elif method == "Fill with Median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                output_box.insert(tk.END, f"\nFilled missing values in '{col}' with median.\n")
                log_action(f"Filled missing values in '{col}' with median.")
            else:
                output_box.insert(tk.END, f"\nCannot use median on non-numeric column '{col}'.\n")

        elif method == "Fill with Mode":
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series[0])
                output_box.insert(tk.END, f"\nFilled missing values in '{col}' with mode.\n")
                log_action(f"Filled missing values in '{col}' with mode.")
            else:
                output_box.insert(tk.END, f"\nNo mode found for '{col}'.\n")

        elif method == "Fill with Custom Value":
            fill_value = custom_value

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = pd.to_numeric(custom_value, errors="coerce")

            df[col] = df[col].fillna(fill_value)
            output_box.insert(tk.END, f"\nFilled missing values in '{col}' with custom value: {fill_value}\n")
            log_action(f"Filled missing values in '{col}' with custom value: {fill_value}")

        elif method == "Drop Rows with Missing":
            before = len(df)
            selected_file = df.dropna(subset=[col])
            after = len(selected_file)
            output_box.insert(tk.END, f"\nDropped {before - after} rows with missing values in '{col}'.\n")
            log_action(f"Dropped {before - after} rows with missing values in '{col}'.")

        elif method == "Drop Column":
            selected_file = df.drop(columns=[col])
            output_box.insert(tk.END, f"\nDropped column '{col}'.\n")
            log_action(f"Dropped column '{col}'.")
            

            # Update the column listbox
            column_listbox.delete(0, tk.END)
            for column in selected_file.columns:
                column_listbox.insert(tk.END, column)

            selected_column = None
        
        update_preview_table(selected_file)

        popup.destroy()

    except Exception as e:
        output_box.insert(tk.END, f"\nError handling missing data for '{col}': {e}\n")

def open_missing_data_window():
    global selected_file, selected_column

    if selected_file is None or selected_column is None:
        output_box.insert(tk.END, "\nPlease load a file and select a column first.\n")
        return

    popup = tk.Toplevel(root)
    popup.title("Handle Missing Data")
    popup.geometry("350x220")

    tk.Label(popup, text=f"Selected Column: {selected_column}", font=("Helvetica", 10, "bold")).pack(pady=10)

    tk.Label(popup, text="Choose missing data action:").pack()

    method_var = tk.StringVar(value="Fill with Mean")

    method_menu = tk.OptionMenu(
        popup,
        method_var,
        "Fill with Mean",
        "Fill with Median",
        "Fill with Mode",
        "Fill with Custom Value",
        "Drop Rows with Missing",
        "Drop Column"
    )
    method_menu.pack(pady=5)

    tk.Label(popup, text="Custom Value (only used if selected):").pack()
    custom_value_entry = tk.Entry(popup, width=25)
    custom_value_entry.pack(pady=5)

    apply_button = tk.Button(
        popup,
        text="Apply",
        command=lambda: apply_missing_data_action(method_var.get(), custom_value_entry.get(), popup)
    )
    apply_button.pack(pady=15)

def apply_create_column(popup, new_col_name, method, primary_col, secondary_col, operation, custom_value, extra_value):
    global selected_file

    if selected_file is None:
        output_box.insert(tk.END, "\nNo file loaded.\n")
        return

    df = selected_file

    if not new_col_name.strip():
        output_box.insert(tk.END, "\nPlease enter a new column name.\n")
        return

    if new_col_name in df.columns:
        output_box.insert(tk.END, f"\nColumn '{new_col_name}' already exists.\n")
        return
    
    save_state()

    try:
        action_details = ""  # <-- build a detailed log message

        # 1. Copy / transform single column
        if method == "Copy / Transform Single Column":
            if operation == "Copy":
                df[new_col_name] = df[primary_col]
                action_details = f"Copied '{primary_col}'"

            elif operation == "Lowercase":
                df[new_col_name] = df[primary_col].where(
                    df[primary_col].isna(),
                    df[primary_col].astype(str).str.lower()
                )
                action_details = f"Lowercased '{primary_col}'"

            elif operation == "Uppercase":
                df[new_col_name] = df[primary_col].where(
                    df[primary_col].isna(),
                    df[primary_col].astype(str).str.upper()
                )
                action_details = f"Uppercased '{primary_col}'"

            elif operation == "Strip Whitespace":
                df[new_col_name] = df[primary_col].where(
                    df[primary_col].isna(),
                    df[primary_col].astype(str).str.strip()
                )
                action_details = f"Stripped whitespace from '{primary_col}'"

            elif operation == "Text Length":
                df[new_col_name] = df[primary_col].where(
                    df[primary_col].isna(),
                    df[primary_col].astype(str).str.len()
                )
                action_details = f"Calculated text length of '{primary_col}'"

        # 2. Math between two columns
        elif method == "Math Between Two Columns":
            col1 = pd.to_numeric(df[primary_col], errors="coerce")
            col2 = pd.to_numeric(df[secondary_col], errors="coerce")

            if operation == "Add":
                df[new_col_name] = col1 + col2
                action_details = f"{primary_col} + {secondary_col}"

            elif operation == "Subtract":
                df[new_col_name] = col1 - col2
                action_details = f"{primary_col} - {secondary_col}"

            elif operation == "Multiply":
                df[new_col_name] = col1 * col2
                action_details = f"{primary_col} * {secondary_col}"

            elif operation == "Divide":
                df[new_col_name] = col1 / col2
                action_details = f"{primary_col} / {secondary_col}"

        # 3. Combine text columns
        elif method == "Combine Text Columns":
            separator = custom_value if custom_value != "" else " "
            left = df[primary_col].fillna("").astype(str)
            right = df[secondary_col].fillna("").astype(str)
            df[new_col_name] = left + separator + right
            df[new_col_name] = df[new_col_name].str.strip()

            action_details = f"Combined '{primary_col}' and '{secondary_col}' with separator '{separator}'"

        # 4. Extract datetime part
        elif method == "Extract Datetime Part":
            dt_col = pd.to_datetime(df[primary_col], errors="coerce")

            if operation == "Year":
                df[new_col_name] = dt_col.dt.year
            elif operation == "Month":
                df[new_col_name] = dt_col.dt.month
            elif operation == "Day":
                df[new_col_name] = dt_col.dt.day
            elif operation == "Day of Week":
                df[new_col_name] = dt_col.dt.day_name()

            action_details = f"Extracted '{operation}' from '{primary_col}'"

        # 5. Conditional flag
        elif method == "Conditional Flag":
            numeric_col = pd.to_numeric(df[primary_col], errors="coerce")
            threshold = pd.to_numeric(custom_value, errors="coerce")

            if pd.isna(threshold):
                output_box.insert(tk.END, "\nPlease enter a valid numeric threshold.\n")
                return

            true_label = extra_value if extra_value.strip() else "True"
            false_label = "False"

            if operation == "Greater Than":
                df[new_col_name] = numeric_col.apply(
                    lambda x: true_label if pd.notna(x) and x > threshold else false_label
                )
                action_details = f"{primary_col} > {threshold}"

            elif operation == "Less Than":
                df[new_col_name] = numeric_col.apply(
                    lambda x: true_label if pd.notna(x) and x < threshold else false_label
                )
                action_details = f"{primary_col} < {threshold}"

        # 6. Bin numeric column
        elif method == "Bin Numeric Column":
            numeric_col = pd.to_numeric(df[primary_col], errors="coerce")
            bin_size = pd.to_numeric(custom_value, errors="coerce")

            if pd.isna(bin_size) or bin_size <= 0:
                output_box.insert(tk.END, "\nPlease enter a valid positive bin size.\n")
                return

            min_val = numeric_col.min()
            max_val = numeric_col.max()

            if pd.isna(min_val) or pd.isna(max_val):
                output_box.insert(tk.END, "\nColumn does not contain valid numeric values for binning.\n")
                return

            bins = list(range(int(min_val), int(max_val) + int(bin_size) + 1, int(bin_size)))
            df[new_col_name] = pd.cut(numeric_col, bins=bins, include_lowest=True)

            action_details = f"Binned '{primary_col}' with bin size {bin_size}"

        # Output + log
        message = f"Created column '{new_col_name}' using {method} ({action_details})"
        output_box.insert(tk.END, f"\n{message}\n")
        log_action(message)

        refresh_column_list()
        update_preview_table(selected_file)
        popup.destroy()

    except Exception as e:
        output_box.insert(tk.END, f"\nError creating new column: {e}\n")

def refresh_column_list():
    column_listbox.delete(0, tk.END)
    for column in selected_file.columns:
        column_listbox.insert(tk.END, column)



def apply_replace(find_val, replace_val, popup):
    global selected_file, selected_column

    if selected_file is None or selected_column is None:
        output_box.insert(tk.END, "\nNo file or column selected.\n")
        return

    col = selected_column
    df = selected_file

    save_state()

    try:
        # Smart type handling
        if pd.api.types.is_numeric_dtype(df[col]):
            find_val = pd.to_numeric(find_val, errors="coerce")
            replace_val = pd.to_numeric(replace_val, errors="coerce")

        before = df[col].isin([find_val]).sum()

        df[col] = df[col].replace(find_val, replace_val)

        after = df[col].isin([replace_val]).sum()

        output_box.insert(tk.END, f"\nReplaced {before} occurrences of '{find_val}' with '{replace_val}' in '{col}'.\n")
        log_action(f"Replaced {before} occurrences of '{find_val}' with '{replace_val}' in '{col}'.")

        update_preview_table(selected_file)
        popup.destroy()

    except Exception as e:
        output_box.insert(tk.END, f"\nError replacing values: {e}\n")

def apply_type_conversion(target_type, popup):
    global selected_file, selected_column

    if selected_file is None or selected_column is None:
        output_box.insert(tk.END, "\nNo file or column selected.\n")
        return

    col = selected_column
    df = selected_file

    save_state()

    try:
        if target_type == "Numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            output_box.insert(tk.END, f"\nConverted '{col}' to numeric (invalid values set to NaN).\n")
            log_action(f"Converted '{col}' to numeric (invalid values set to NaN).")

        elif target_type == "String":
            df[col] = df[col].astype(str)
            output_box.insert(tk.END, f"\nConverted '{col}' to string.\n")
            log_action(f"Converted '{col}' to string.")

        elif target_type == "Datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
            output_box.insert(tk.END, f"\nConverted '{col}' to datetime (invalid values set to NaT).\n")
            log_action(f"Converted '{col}' to datetime (invalid values set to NaT).")

        elif target_type == "Boolean":
            # Smart boolean conversion
            df[col] = df[col].astype(str).str.lower().map({
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False
            })

            output_box.insert(tk.END, f"\nConverted '{col}' to boolean.\n")
            log_action(f"Converted '{col}' to boolean.")
        update_preview_table(selected_file)
        popup.destroy()

    except Exception as e:
        output_box.insert(tk.END, f"\nError converting '{col}': {e}\n")


def update_preview_table(df, num_rows=20):
    preview_table.delete(*preview_table.get_children())

    preview_table["columns"] = list(df.columns)

    for col in df.columns:
        preview_table.heading(col, text=col)
        preview_table.column(col, width=120, anchor="w")

    preview_df = df.head(num_rows)

    for _, row in preview_df.iterrows():
        display_row = ["" if pd.isna(val) else val for val in row]
        preview_table.insert("", tk.END, values=display_row)

def open_convert_window():
    global selected_column

    if selected_column is None:
        output_box.insert(tk.END, "\nSelect a column first.\n")
        return

    popup = tk.Toplevel(root)
    popup.title("Convert Data Type")
    popup.geometry("300x180")

    tk.Label(popup, text=f"Column: {selected_column}", font=("Helvetica", 10, "bold")).pack(pady=10)

    tk.Label(popup, text="Choose target type:").pack()

    type_var = tk.StringVar(value="Numeric")

    type_menu = tk.OptionMenu(
        popup,
        type_var,
        "Numeric",
        "String",
        "Datetime",
        "Boolean"
    )
    type_menu.pack(pady=5)

    tk.Button(
        popup,
        text="Apply",
        command=lambda: apply_type_conversion(type_var.get(), popup)
    ).pack(pady=10)

def open_replace_window():
    global selected_column

    if selected_column is None:
        output_box.insert(tk.END, "\nSelect a column first.\n")
        return

    popup = tk.Toplevel(root)
    popup.title("Find & Replace")
    popup.geometry("300x200")

    tk.Label(popup, text=f"Column: {selected_column}", font=("Helvetica", 10, "bold")).pack(pady=10)

    tk.Label(popup, text="Find:").pack()
    find_entry = tk.Entry(popup)
    find_entry.pack(pady=5)

    tk.Label(popup, text="Replace with:").pack()
    replace_entry = tk.Entry(popup)
    replace_entry.pack(pady=5)

    tk.Button(
        popup,
        text="Apply",
        command=lambda: apply_replace(find_entry.get(), replace_entry.get(), popup)
    ).pack(pady=10)

def save_cleaned_csv():
    global selected_file

    if selected_file is None:
        output_box.insert(tk.END, "\nNo file loaded to save.\n")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not save_path:
        output_box.insert(tk.END, "\nSave cancelled.\n")
        return

    try:
        selected_file.to_csv(save_path, index=False)
        output_box.insert(tk.END, f"\nSaved cleaned file to: {save_path}\n")
        log_action(f"Saved cleaned file to: {save_path}")
    except Exception as e:
        output_box.insert(tk.END, f"\nError saving file: {e}\n")

def open_create_column_window():
    global selected_file, selected_column

    if selected_file is None or selected_column is None:
        output_box.insert(tk.END, "\nPlease load a file and select a column first.\n")
        return

    popup = tk.Toplevel(root)
    popup.title("Create New Column")
    popup.geometry("500x500")

    tk.Label(popup, text="Create New Column", font=("Helvetica", 12, "bold")).pack(pady=10)

    tk.Label(popup, text="New Column Name:").pack()
    new_col_entry = tk.Entry(popup, width=30)
    new_col_entry.pack(pady=5)

    tk.Label(popup, text="Creation Method:").pack()
    method_var = tk.StringVar(value="Math Between Two Columns")

    method_menu = tk.OptionMenu(
        popup,
        method_var,
        "Copy / Transform Single Column",
        "Math Between Two Columns",
        "Combine Text Columns",
        "Extract Datetime Part",
        "Conditional Flag",
        "Bin Numeric Column"
    )
    method_menu.pack(pady=5)

    tk.Label(popup, text="Primary Column:").pack()
    primary_col_var = tk.StringVar(value=selected_column)
    primary_col_menu = tk.OptionMenu(popup, primary_col_var, *selected_file.columns)
    primary_col_menu.pack(pady=5)

    tk.Label(popup, text="Secondary Column (if needed):").pack()
    secondary_col_var = tk.StringVar(value=selected_column)
    secondary_col_menu = tk.OptionMenu(popup, secondary_col_var, *selected_file.columns)
    secondary_col_menu.pack(pady=5)

    tk.Label(popup, text="Operation / Transform:").pack()
    operation_var = tk.StringVar(value="Multiply")
    operation_menu = tk.OptionMenu(
        popup,
        operation_var,
        "Copy",
        "Lowercase",
        "Uppercase",
        "Strip Whitespace",
        "Text Length",
        "Add",
        "Subtract",
        "Multiply",
        "Divide",
        "Year",
        "Month",
        "Day",
        "Day of Week",
        "Greater Than",
        "Less Than"
    )
    operation_menu.pack(pady=5)

    tk.Label(popup, text="Custom Value / Threshold / Separator / Bin Size:").pack()
    custom_value_entry = tk.Entry(popup, width=30)
    custom_value_entry.pack(pady=5)

    tk.Label(popup, text="True Label / Upper Bin Label (optional):").pack()
    extra_value_entry = tk.Entry(popup, width=30)
    extra_value_entry.pack(pady=5)

    tk.Button(
        popup,
        text="Apply",
        command=lambda: apply_create_column(
            popup,
            new_col_entry.get(),
            method_var.get(),
            primary_col_var.get(),
            secondary_col_var.get(),
            operation_var.get(),
            custom_value_entry.get(),
            extra_value_entry.get()
        )
    ).pack(pady=15)

# ------------------------
# UI SETUP
# ------------------------

root = tk.Tk()
root.title("Data Cleaning Tool")
root.geometry("1400x800")

# Top Frame (buttons)
top_frame = tk.Frame(root)
top_frame.pack(fill=tk.X)

tk.Button(top_frame, text="Load CSV", command=load_file).pack(side=tk.LEFT, padx=5, pady=5)
save_button = tk.Button(top_frame, text="Save Cleaned CSV", command=save_cleaned_csv)
save_button.pack(side=tk.LEFT, padx=5, pady=5)

undo_button = tk.Button(top_frame, text="Undo Last Change", command=undo_last_change)
undo_button.pack(side=tk.LEFT, padx=5, pady=5)


tk.Button(top_frame, text="File Summary", command=file_summary).pack(side=tk.LEFT, padx=5, pady=5)

ai_summary_button = tk.Button(top_frame, text="Generate AI Summary", command=generate_ai_summary)
ai_summary_button.pack(side=tk.LEFT, padx=5, pady=5)

show_ai_button = tk.Button(top_frame, text="Show Last AI Summary", command=show_last_ai_summary)
show_ai_button.pack(side=tk.LEFT, padx=5, pady=5)


# Main Frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# ------------------------
# Left Panel (columns)
# ------------------------
left_frame = tk.Frame(main_frame, width=200)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

column_label = tk.Label(left_frame, text="Columns", font=("Helvetica", 11, "bold"))
column_label.pack(anchor="w", padx=5, pady=(10, 0))

column_listbox = tk.Listbox(left_frame)
column_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
column_listbox.bind("<<ListboxSelect>>", on_column_select)

# ------------------------
# Center Panel (output + history + edit buttons)
# ------------------------
center_frame = tk.Frame(main_frame)
center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Output section
output_label = tk.Label(center_frame, text="Output", font=("Helvetica", 11, "bold"))
output_label.pack(anchor="w", padx=10, pady=(10, 0))

output_box = scrolledtext.ScrolledText(center_frame, width=80, height=20, wrap=tk.WORD)
output_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
output_box.tag_configure("header", font=("Helvetica", 12, "bold"))

# History section
history_label = tk.Label(center_frame, text="Action Log", font=("Helvetica", 11, "bold"))
history_label.pack(anchor="w", padx=10, pady=(10, 0))

history_box = scrolledtext.ScrolledText(center_frame, width=80, height=8, wrap=tk.WORD)
history_box.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))
history_box.config(state="disabled")

# Edit panel
edit_frame = tk.Frame(center_frame)
edit_frame.pack(fill=tk.X, padx=10, pady=5)

missing_button = tk.Button(edit_frame, text="Handle Missing Data", command=open_missing_data_window)
missing_button.pack(side=tk.LEFT, padx=5)

convert_type_button = tk.Button(edit_frame, text="Change Data Type", command=open_convert_window)
convert_type_button.pack(side=tk.LEFT, padx=5)

replace_button = tk.Button(edit_frame, text="Find & Replace", command=open_replace_window)
replace_button.pack(side=tk.LEFT, padx=5)

create_column_button = tk.Button(edit_frame, text="Create New Column", command=open_create_column_window)
create_column_button.pack(side=tk.LEFT, padx=5)

# ------------------------
# Right Panel (data preview)
# ------------------------
preview_frame = tk.Frame(main_frame, width=500)
preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

preview_label = tk.Label(preview_frame, text="Data Preview", font=("Helvetica", 11, "bold"))
preview_label.pack(anchor="w", padx=10, pady=(10, 0))

table_container = tk.Frame(preview_frame)
table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

preview_table = ttk.Treeview(table_container, show="headings")
preview_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

preview_scroll_y = tk.Scrollbar(table_container, orient="vertical", command=preview_table.yview)
preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

preview_scroll_x = tk.Scrollbar(preview_frame, orient="horizontal", command=preview_table.xview)
preview_scroll_x.pack(fill=tk.X, padx=10, pady=(0, 10))

preview_table.configure(
    yscrollcommand=preview_scroll_y.set,
    xscrollcommand=preview_scroll_x.set
)

# ------------------------
# RUN APP
# ------------------------

root.mainloop()