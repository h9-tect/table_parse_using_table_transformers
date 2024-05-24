import streamlit as st
from PIL import Image
from utils.model import load_detection_model, load_structure_model
from utils.preprocessing import prepare_image, prepare_cropped_image
from utils.postprocessing import outputs_to_objects, objects_to_crops
from utils.detection import detect_tables, detect_cells
from utils.visualization import visualize_detected_tables, plot_results
from utils.ocr import apply_ocr, save_csv, get_cell_coordinates_by_row

# Load models
detection_model, device = load_detection_model()
structure_model = load_structure_model(device)

st.title("Table Detection and OCR with Transformers")

# Upload image
uploaded_file = st.file_uploader("Upload an image containing a table", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect tables
    pixel_values = prepare_image(image, device)
    outputs = detect_tables(detection_model, pixel_values)
    objects = outputs_to_objects(outputs, image.size, detection_model.config.id2label)

    st.write("Detected tables:")
    st.write(objects)

    # Visualize tables
    fig = visualize_detected_tables(image, objects)
    st.pyplot(fig)

    # Crop the table
    tokens = []
    detection_class_thresholds = {"table": 0.5, "table rotated": 0.5, "no object": 10}
    tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)
    cropped_table = tables_crops[0]['image'].convert("RGB")

    st.image(cropped_table, caption='Cropped Table', use_column_width=True)

    # Detect cells in the cropped table
    pixel_values = prepare_cropped_image(cropped_table, device)
    outputs = detect_cells(structure_model, pixel_values)
    cells = outputs_to_objects(outputs, cropped_table.size, structure_model.config.id2label)

    st.write("Detected cells:")
    st.write(cells)

    # Visualize cells
    plot_results(cells, "table row")
    st.pyplot()

    # Apply OCR
    cell_coordinates = get_cell_coordinates_by_row(cells)
    data = apply_ocr(cell_coordinates, cropped_table)

    # Display OCR results
    st.write("Extracted Table Data:")
    st.write(data)

    # Save results as CSV
    save_csv(data)
    st.write("CSV file saved as output.csv")
