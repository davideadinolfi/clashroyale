from pathlib import Path
from collections import Counter
from types import NoneType



def count_first_numbers(folder: str):
    counter = Counter()

    for txt_file in Path(folder).glob("*.txt"):
        with txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                first_token = line.split()[0]

                if first_token.isdigit():
                    counter[int(first_token)] += 1

    return counter

def labelcounter():
    folder = "../generation/dataset_output/labels"  # <-- cambia con la cartella dei .txt
    counts = count_first_numbers(folder)
    labels =     ['archer', 'barbarian', 'battleram', 'bomber', 'cannon', 'electrospirit', 'giant', 'goblin', 'goblinHut', 'hogrider', 'infernotower',
                  'knight', 'minipekka', 'mortar', 'skeleton', 'spearGoblin', 'superGoblin', 'tombstone', 'valk', 'wizard',
                  'bombtower', 'firespirit', 'furnace', 'goblinCage', 'musketeer'
        , 'bat', 'flyingmachine', 'megaminion', 'minion', 'skeletondragon', 'AllyBigHP', 'AllyTinyHP',
                  'EnemyBigHP', 'enemyLevel', 'EnemyTinyHP', 'tower']
    for key, value in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{labels[key]}: {value}")

def getSecondi(str):
    try:
        if(type(str) == NoneType):
            return "errore timer"
        str = str.replace(":", "")
        if(4>len(str)>2):
            return int(str[0])*60+int(str[-1])+int(str[-2])*10  #l'unica riga di codice non scritta da chatgpt
    except Exception:
        return "errore timer"

if __name__ == "__main__":
    labelcounter()