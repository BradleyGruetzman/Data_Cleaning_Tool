# Data Cleaning Tool (Python + Tkinter)

A desktop data cleaning and analysis tool built with Python, Pandas, and Tkinter.  
This application allows users to explore datasets, clean data, engineer features, and generate AI-powered insights — all without writing code.

---

## Features

- Load and preview CSV datasets
- Dataset summary and issue detection
- Handle missing values (mean, median, mode, custom)
- Change data types (numeric, datetime, etc.)
- Find and replace values
- Create new columns (feature engineering)
- Remove duplicates
- Undo last change
- Action log/history tracking
- Interactive data preview table
- AI-generated dataset insights (optional)

---

## Screenshots

### Main Interface
![Home Screen](Data_Cleaning_Tool_Screenshots/Home_Screen.png)

### Load CSV
![Load CSV](Data_Cleaning_Tool_Screenshots/Load_CSV.png)

### File Summary
![File Summary](Data_Cleaning_Tool_Screenshots/File_Summary.png)

### Column Selection
![Column Select](Data_Cleaning_Tool_Screenshots/Column_Select.png)

### Change Data Type
![Change Data Type](Data_Cleaning_Tool_Screenshots/Change_Data_Type.png)

### Handle Missing Data
![Handle Missing Data](Data_Cleaning_Tool_Screenshots/Handle_Missing_Data.png)

### Find & Replace
![Find & Replace](Data_Cleaning_Tool_Screenshots/Find_&_Replace.png)

### Create New Column
![Create New Column](Data_Cleaning_Tool_Screenshots/Create_New_Column.png)

### AI Summary
![AI Summary](Data_Cleaning_Tool_Screenshots/Generate_AI_Summary.png)

---

## File Structure
Data_Cleaning_Tool/
│── data_cleaning_tool.py
│── Data_Cleaning_Tool_Screenshots/
│ ├── Home_Screen.png
│ ├── File_Summary.png
│ ├── Change_Data_Type.png
│ ├── Column_Select.png
│ ├── Create_New_Column.png
│ ├── Find_&_Replace.png
│ ├── Generate_AI_Summary.png
│ ├── Handle_Missing_Data.png
│ ├── Load_CSV.png


---

## How to Run

1. Clone the repository  
2. Install dependencies:

```bash
pip install pandas openai```

3. Run the application:
python data_cleaning_tool.py

AI Summary Feature (Optional)

This tool includes an AI-powered dataset summary using the OpenAI API.

To enable this feature:
```bash
setx OPENAI_API_KEY "your_api_key_here"```

Restart your terminal or IDE after setting the key.
Note: The application does not store or include any API keys.

Technologies Used
Python
Pandas
Tkinter
OpenAI API

