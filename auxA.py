import json

# Lista de IDs a conservar
ids = [
    6170, 2040, 9420, 1970, 9569, 7352, 3035, 9929, 4414, 4924,
    9134, 7017, 5985, 5361, 5810, 10588, 2784, 2546, 3614, 8735,
    1541, 6435, 7745, 8884, 1764, 1754, 636, 1907, 5046, 8714,
    6982, 1904, 2715, 9789, 1372, 7994, 6409, 8854, 157, 10149,
    9004, 7227, 10479, 6749, 3991, 1479, 790, 6452, 4351, 7766,
    10092, 7993, 1984, 3006, 1946, 10090, 9252, 8399, 1195, 3157,
    8683, 2204, 2597, 1838, 10745, 4207, 10862, 2001, 3389, 2187,
    7942, 674, 6275, 7419, 4521, 8231, 448, 8002, 689, 4056,
    9703, 3337, 9316, 3477, 7584, 6447, 8355, 8137, 3799, 1864,
    8777, 3861, 4223, 10181, 2915, 2164, 4919, 5282, 3865, 4814
]

# Cargar el JSON original
with open("support_cases_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Filtrar los elementos cuyo id está en la lista
filtrados = [item for item in data if item["id"] in ids]

# Guardar los resultados en un nuevo archivo
with open("datos_filtrados.json", "w", encoding="utf-8") as f:
    json.dump(filtrados, f, indent=4, ensure_ascii=False)
