import os

# folder w którym działasz (możesz zmienić na własny)
folder = "."

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        base = os.path.splitext(filename)[0]   # nazwa bez rozszerzenia
        txt_file = base + ".txt"
        
        if not os.path.exists(os.path.join(folder, txt_file)):
            jpg_path = os.path.join(folder, filename)
            print(f"Usuwam: {jpg_path}")
            os.remove(jpg_path)
