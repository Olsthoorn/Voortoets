# Made by chatGPT op 13-03-2025 als workaround om LyX tekst te kunnen spellchecken met hunspell
# Dit omdat de spellchecker in Lyx niet werkt.
# @TO 20250313

import os
import subprocess
import re
from collections import Counter

def export_lyx_to_tex(lyx_file):
    tex_file = lyx_file.replace(".lyx", ".tex")
    cmd = ["/System/Volumes/Data/Applications/LyX.app/Contents/MacOS/lyx", "--export", "latex", lyx_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(tex_file):
        print(f"Export geslaagd: {tex_file}")
        return tex_file
    else:
        print("Fout bij exporteren van LyX naar TeX:", result.stderr)
        return None

def clean_tex_content(tex_file):
    with open(tex_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Verwijder LaTeX-commando's
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", "", text)
    text = re.sub(r"%.*", "", text)  # Verwijder commentaarregels
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Behoud alleen letters en spaties
    
    return text

def check_spelling_with_hunspell(text, lang="nl_NL"):
    process = subprocess.Popen(
        ["hunspell", "-d", lang],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, _ = process.communicate(input=text)
    
    # Parse hunspell output (woorden die beginnen met "&" of "#")
    words = re.findall(r"^(?:&|#)\s(\S+)", output, re.MULTILINE)
    return Counter(words)

def main(lyx_file):
    if not lyx_file.endswith(".lyx"):
        print("Geef een geldig LyX-bestand op.")
        return
    
    tex_file = export_lyx_to_tex(lyx_file)
    if not tex_file:
        return
    
    text = clean_tex_content(tex_file)
    mistakes = check_spelling_with_hunspell(text)
    
    if mistakes:
        print("Fout gespelde woorden (gesorteerd op frequentie):")
        
        with open("/Users/Theo/Entiteiten/Hygea/2022-AGT/Doc_rapportage/mistakes.txt", 'w') as f:
            for word, count in mistakes.most_common():
                f.write(f"{word}: {count}\n")
                print(f"{word}: {count}")
    else:
        print("Geen spellingsfouten gevonden!")
    
if __name__ == "__main__":    
    pth = '/Users/Theo/Entiteiten/Hygea/2022-AGT/Doc_rapportage'
    
    os.chdir(pth)
    
    lyx_file = os.join(pth, "VrijeAfwatering.lyx")
    
    print(os.path.isfile(lyx_file))
    #lyx_file = input("Voer het pad in naar je LyX-bestand: ")
    main(lyx_file)
