from openpyxl import load_workbook

# Try loading the Excel file directly using openpyxl
try:
    wb = load_workbook(r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_labels.xlsx")
    print("Workbook loaded successfully.")
except Exception as e:
    print(f"Error loading workbook: {e}")
