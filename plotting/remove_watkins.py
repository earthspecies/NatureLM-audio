import os
import json
import shutil

def clean_jsonl_files(folder_path):
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist.")
        return

    # Iterate over files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl") and "cbi" in filename.lower():
            file_path = os.path.join(folder_path, filename)
            backup_path = file_path + ".bak"

            # Make a backup copy
            shutil.copy2(file_path, backup_path)
            print(f"📦 Backup created: {backup_path}")

            filtered_lines = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        # Skip entries with dataset_name = watkins
                        if obj.get("dataset_name") != "watkins":
                            filtered_lines.append(line)
                    except json.JSONDecodeError:
                        # Preserve malformed lines just in case
                        filtered_lines.append(line)

            # Write filtered lines back
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(filtered_lines)

            print(f"✅ Cleaned file: {filename} (removed watkins entries)\n")

# Example usage:
clean_jsonl_files("results/cbi")