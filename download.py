import gdown

# Boyd OPF
# file_id = "1IQDbN2aDkKly90UGCzbHpIFlVSnNgTea"

file_id = "1Q355eW-J_Shg-PEKilJV4Db0USR5zEfF"

url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, fuzzy=True)
