import os

def hex_to_c_array(hex_data, var_name):
    c_str = f"const unsigned char {var_name}[] = {{\n"
    hex_array = [hex_data[i:i+2] for i in range(0, len(hex_data), 2)]
    
    for i, hex_val in enumerate(hex_array):
        c_str += f"0x{hex_val}, "
        if (i + 1) % 12 == 0:
            c_str += "\n"
            
    c_str += f"\n}};\nconst unsigned int {var_name}_len = {len(hex_array)};"
    return c_str

# Read the TFLite file
with open("model.tflite", "rb") as f:
    tflite_content = f.read()

# Convert to hex
hex_content = tflite_content.hex()
c_array_code = hex_to_c_array(hex_content, "model_tflite")

# Ensure the output directory exists
os.makedirs("esp32_firmware/src", exist_ok=True)

# Write to C++ file
with open("esp32_firmware/src/model.cc", "w") as f:
    f.write(c_array_code)

print("Successfully generated esp32_firmware/src/model.cc")