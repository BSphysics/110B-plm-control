import pandas as pd

def spot_index(row, col):
    """Return the column-wise spot index (1–49) from 1-based row and column."""
    return row + (col - 1) * 7

def letter_to_spots(pattern):
    """
    Convert a 7x7 pattern (list of 7 strings of 7 chars: '1' = on, '0' = off)
    into a list of spot indices.
    """
    spots = []
    for r, row in enumerate(pattern, start=1):
        for c, ch in enumerate(row, start=1):
            if ch == "1":
                spots.append(spot_index(r, c))
    return spots

letters_7x7 = {
"A": [
    "0011100",
    "0100010",
    "1000001",
    "1111111",
    "1000001",
    "1000001",
    "1000001",
],
"B": [
    "1111100",
    "1000010",
    "1000010",
    "1111100",
    "1000010",
    "1000010",
    "1111100",
],
"C": [
    "0111110",
    "1000001",
    "1000000",
    "1000000",
    "1000000",
    "1000001",
    "0111110",
],
"D": [
    "1111100",
    "1000010",
    "1000001",
    "1000001",
    "1000001",
    "1000010",
    "1111100",
],
"E": [
    "1111111",
    "1000000",
    "1000000",
    "1111110",
    "1000000",
    "1000000",
    "1111111",
],
"F": [
    "1111111",
    "1000000",
    "1000000",
    "1111110",
    "1000000",
    "1000000",
    "1000000",
],
"G": [
    "0111110",
    "1000001",
    "1000000",
    "1000111",
    "1000001",
    "1000001",
    "0111110",
],
"H": [
    "1000001",
    "1000001",
    "1000001",
    "1111111",
    "1000001",
    "1000001",
    "1000001",
],
"I": [
    "1111111",
    "0001000",
    "0001000",
    "0001000",
    "0001000",
    "0001000",
    "1111111",
],
"J": [
    "0001111",
    "0000010",
    "0000010",
    "0000010",
    "1000010",
    "1000010",
    "0111100",
],
"K": [
    "1000010",
    "1000100",
    "1001000",
    "1110000",
    "1001000",
    "1000100",
    "1000010",
],
"L": [
    "1000000",
    "1000000",
    "1000000",
    "1000000",
    "1000000",
    "1000000",
    "1111111",
],
"M": [
    "1000001",
    "1100011",
    "1010101",
    "1001001",
    "1000001",
    "1000001",
    "1000001",
],
"N": [
    "1000001",
    "1100001",
    "1010001",
    "1001001",
    "1000101",
    "1000011",
    "1000001",
],
"O": [
    "0111110",
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "0111110",
],
"P": [
    "1111110",
    "1000001",
    "1000001",
    "1111110",
    "1000000",
    "1000000",
    "1000000",
],
"Q": [
    "0111110",
    "1000001",
    "1000001",
    "1000001",
    "1000101",
    "1000010",
    "0111101",
],
"R": [
    "1111110",
    "1000001",
    "1000001",
    "1111110",
    "1001000",
    "1000100",
    "1000010",
],
"S": [
    "0111110",
    "1000001",
    "1000000",
    "0111110",
    "0000001",
    "1000001",
    "0111110",
],
"T": [
    "1111111",
    "0001000",
    "0001000",
    "0001000",
    "0001000",
    "0001000",
    "0001000",
],
"U": [
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "0111110",
],
"V": [
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "0100010",
    "0011100",
],
"W": [
    "1000001",
    "1000001",
    "1000001",
    "1001001",
    "1010101",
    "1100011",
    "1000001",
],
"X": [
    "1000001",
    "0100010",
    "0010100",
    "0001000",
    "0010100",
    "0100010",
    "1000001",
],
"Y": [
    "1000001",
    "0100010",
    "0010100",
    "0001000",
    "0001000",
    "0001000",
    "0001000",
],
"Z": [
    "1111111",
    "0000001",
    "0000010",
    "0000100",
    "0001000",
    "0010000",
    "1111111",
],
}

letter_spots = {letter: letter_to_spots(pattern) for letter, pattern in letters_7x7.items()}

# Example:
print("Letter E spots:", letter_spots["R"])


import matplotlib.pyplot as plt


letter = "T"
spots_to_plot = letter_spots[letter] 

# Grid size
cols = 7
rows = 7

# Function to convert spot number to (x, y) coordinates
def spot_to_coords(spot_number):
    # Column-wise numbering: 1-7 first column, 8-14 second, etc.
    col = (spot_number - 1) // rows
    row = (spot_number - 1) % rows
    # Flip y-axis if you want top-left as origin
    return col, rows - 1 - row

# Convert all spots to coordinates
coords = [spot_to_coords(spot) for spot in spots_to_plot]
x, y = zip(*coords)

# Plot
plt.figure(figsize=(5,5))
plt.scatter(x, y, s=200, c='blue')
plt.xlim(-0.5, cols-0.5)
plt.ylim(-0.5, rows-0.5)
plt.xticks(range(cols))
plt.yticks(range(rows))
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Spot positions in 7x7 grid")
plt.show()

input_file  = "multiBeamData_FLAT.xlsx"
output_file = "multiBeamData_" + letter + ".xlsx"

allowed_n = {1, 7, 9, 13, 17, 19, 25, 31, 33, 37, 41, 43, 49}

df = pd.read_excel(input_file, header=None)

# Clean columns robustly
df[1] = pd.to_numeric(df[1], errors="coerce")          # beam number
df[2] = df[2].astype(str).str.strip()                  # parameter name

mask = (
    df[2].str.fullmatch("Global amplitude") &
    (~df[1].isin(spots_to_plot))
)

df.loc[mask, 3] = 0

df.to_excel(output_file, index=False, header=False)

