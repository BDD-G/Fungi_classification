print("Starting FastAPI application...")

import torch
import sys
import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, Response
from PIL import Image
import io
import torchvision.transforms as transforms
from torchvision.transforms import v2
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Jinja2 templates setup
templates = Jinja2Templates(directory="templates")

print("Imports done")

sys.path.append('/zhome/ac/d/174101/thesis/src')
from utils.DenseNet import DenseNet
import torchvision.datasets as datasets

print("Model and datasets loaded")

# DensNetX
model_parameters={}
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32]
model_parameters['densenet264'] = [6,12,64,48]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device set")

# Load the pre-trained model
base_path = '/work3/s220243/Thesis'
data_dir = Path(base_path) / 'data_split_resized'
model_name = "fungi_DenseNet_base_B32_E1000_lr1.000e-04_Adam_KLDIV.pth" #specify a model
checkpoint_path = Path(base_path) / f"models/{model_name}" 

print("Paths set")

app = FastAPI()

# Ensure the 'static' directory exists for serving static files
if not os.path.exists('static'):
    os.makedirs('static')

# Mount the static directory to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

print("FastAPI app created")

# Define the transformation function for input images
data_transforms = {
    'train': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.7306, 0.6204, 0.5511], [0.1087, 0.1948, 0.1759]) #Resized
    ]),
    'validation': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.7129, 0.6080, 0.5386], [0.1165, 0.1868, 0.1702]) #Resized
    ]),
}

print("Transformations set")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print("Datasets loaded")

model = DenseNet(model_parameters['densenet121'] , in_channels=3, num_classes=num_classes) #specify architecture
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("Model loaded and set to eval")

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.7306, 0.6204, 0.5511], [0.1087, 0.1948, 0.1759])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

print("Image transform function defined")

# Prediction endpoint
@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_path = f"static/{file.filename}"
        image.save(image_path)

        
        tensor = transform_image(image_bytes)
        outputs = model(tensor)
        
        # Apply sigmoid activation and convert to percentages
        #probabilities = torch.sigmoid(outputs) * 100.0
        probabilities = torch.nn.functional.softmax(outputs, dim=1) * 100.0
        
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # Prepare predictions
        predictions = []
        for i in range(top5_prob.size(1)):
            predictions.append({
                "class_name": class_names[top5_catid[0][i]],
                "probability": top5_prob[0][i].item()
            })
        
        # Render predictions.html template with predictions
        return templates.TemplateResponse("predictions.html", {"request": request, "predictions": predictions, "image_path": f"/static/{file.filename}"})
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return {"error": "Internal Server Error"}



print("Prediction endpoint defined")

# Serve the HTML form for uploading images
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    content = """
    <html>
    <head>
        <title>Upload Image</title>
    </head>
    <body>
        <form action="/predict/" enctype="multipart/form-data" method="post">
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

print("Main endpoint defined")

if __name__ == "__main__":
    print("Running app with uvicorn")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
