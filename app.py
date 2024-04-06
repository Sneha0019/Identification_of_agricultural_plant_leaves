from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torchvision import transforms
import pandas as pd
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app) 

# Load your model
class YourCNN(torch.nn.Module):
    # Define your model here...
    def __init__(self):
        super(YourCNN, self).__init__()
        # Define the convolutional layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Define the dense layers
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(50176, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(1024, 39)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        return x

# Load the model
model_path = "plant_disease_model_1_latest.pt"  # Update with your actual path
model = YourCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load disease info
disease_info_df = pd.read_csv("disease_info.csv", encoding='latin1')  # Update with your CSV path

def predict_single_image(image):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        predicted_index = torch.argmax(output).item()

    return predicted_index

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            image = Image.open(file)
            prediction_index = predict_single_image(image)

            # Get disease information
            row = disease_info_df.loc[disease_info_df['index'] == prediction_index]
            if not row.empty:
                disease_name = row['disease_name'].values[0]
                description = row['description'].values[0]
                steps = row['Possible Steps'].values[0]
                image_url = row['image_url'].values[0]
            else:
                disease_name = "Unknown"
                description = "No information available"
                steps = ""
                image_url = ""

            result = {
                'predicted_disease_name': disease_name,
                'predicted_disease_description': description,
                'predicted_disease_steps': steps,
                'predicted_disease_image_url': image_url
            }

            return jsonify(result)

    return 'Hello, this is your Plant Disease Prediction App!'

if __name__ == '__main__':
    app.run(debug=True)
