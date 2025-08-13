import os

def create_project_structure(base_path):
    # Definirea structurii proiectului
    structure = {
        "Deblurring_Project": {
            "data": {
                "train": {
                    "blur": None,
                    "sharp": None
                },
                "val": {
                    "blur": None,
                    "sharp": None
                },
                "test": {
                    "blur": None
                }
            },
            "models": None,
            "results": None,
            "deblur_app.py": None
        }
    }

    # Funcție recursivă pentru crearea directoarelor și fișierelor
    def create_dirs(base, structure):
        for name, sub_structure in structure.items():
            path = os.path.join(base, name)
            if name.endswith('.py'):  # Crearea fișierului gol .py
                with open(path, 'w') as f:
                    pass
            else:  # Crearea directoarelor
                os.makedirs(path, exist_ok=True)
                if sub_structure:
                    create_dirs(path, sub_structure)

    create_dirs(base_path, structure)

# Setarea locației de bază pe desktop
base_path = os.path.join(os.path.expanduser("~"), "Desktop")
create_project_structure(base_path)

print("Structura proiectului a fost creată pe desktop!")
