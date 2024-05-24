import numpy as np
import easyocr
from tqdm.auto import tqdm
import csv

reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory

def apply_ocr(cell_coordinates, cropped_table):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text

    print("Max number of columns:", max_num_columns)
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data
    return data

def save_csv(data):
    with open('output.csv', 'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        for row, row_text in data.items():
            wr.writerow(row_text)
