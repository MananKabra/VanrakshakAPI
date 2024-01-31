import requests
from flask import Flask, jsonify, request
import os
import cv2
import folium
from selenium import webdriver
import time
import numpy as np
import firebase_admin as admin;
from firebase_admin import storage
from geopy.geocoders import Nominatim
import sys
import torch
from deepforest import main
import os
import cv2
import urllib.request
import numpy as np


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

def predict_image_class(image_path, model, class_names):
    image = preprocess_image(image_path)
    if torch.cuda.is_available():
        image = image.cuda()
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        _, predicted_class = output.max(1)
    
    class_label = class_names[predicted_class.item()]
    
    return class_label


def preprocess_image(image_path):
    custom_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        }
    image = Image.open(image_path).convert('RGB')
    preprocess = custom_transforms['test']
    image = preprocess(image)
    image = image.unsqueeze(0)

    return image


cred = admin.credentials.Certificate("lib/resources/api/vanrakshak-1db86-firebase-adminsdk-thgz2-58a898d20b.json")
default_app = admin.initialize_app(
    cred,
    {'databaseURL': "https://vanrakshak-1db86-default-rtdb.asia-southeast1.firebasedatabase.app/",
     'storageBucket': "vanrakshak-1db86.appspot.com"}
)

app = Flask(__name__)

#########################################################################################################################################################################################
def get_neighboring_coords(base_coords, increments):
    neighboring_coords = []
    for lat_increment, lng_increment in increments:
        neighboring_coords.append([(lat + lat_increment, lng + lng_increment) for lat, lng in base_coords])
    return neighboring_coords

#########################################################################################################################################################################################

@app.route('/satelliteimage',methods=['GET'])
def satelliteImageScript():
    latlngzoom = str(request.args['LatLong']).split(",")
    imageID = str(request.args['ProjectID'])
    zoom = float(str(request.args['zoomlevel']))
    if zoom == 0: 
        zoom=15
    output = {}
    polygon_coords =[]
    elevationList = []

    for i in range(0,len(latlngzoom),2):
        url = "https://maps.googleapis.com/maps/api/elevation/json?locations=" + str(latlngzoom[i]) + "," + str(latlngzoom[i+1]) + "&key=AIzaSyC5QkMgaiQo3G7RH95BWJoqzWbKczWVCkU"
        x = requests.get(url)
        elevationList.append(x.json()["results"][0]["elevation"])
        polygon_coords.append((float(latlngzoom[i]), float(latlngzoom[i+1])))

    center_lat = sum(lat for lat, _ in polygon_coords) / len(polygon_coords)
    center_lng = sum(lng for _, lng in polygon_coords) / len(polygon_coords)

    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.reverse((center_lat, center_lng), language='en')
    address = location.address if location else None

    if(address == None):
        output['locationFromLatLong'] = 'Not Found'
    else:
        output['locationFromLatLong'] = address

    tile_layer = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

    my_map = folium.Map(location=[center_lat, center_lng],min_zoom=zoom-1, zoom_start=zoom,max_zoom=zoom, tiles=tile_layer, attr='Google Maps')
    my_map2 = my_map
    my_map2.fit_bounds(polygon_coords)
    my_map2.save(os.getcwd() + "\\assets\\satelliteMapNoPolygon.html")
    
    folium.Polygon(locations=polygon_coords, color='red').add_to(my_map)

    my_map.fit_bounds(polygon_coords)
    my_map.save(os.getcwd() + "\\assets\\satelliteMap.html")

    



    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  
    driver = webdriver.Chrome(options=options)  

    driver.get(os.getcwd() + "\\assets\\satelliteMap.html") 
    time.sleep(2)

    ##########################################################################################################################################################################

    screenshot_path = os.getcwd() + "\\assets\\" + imageID + "_ConstructionPolygonSatelliteImageUnmasked.png"
    driver.save_screenshot(screenshot_path)

    driver.get(os.getcwd() + "\\assets\\satelliteMapNoPolygon.html") 
    time.sleep(2)

    screenshot_path2 = os.getcwd() + "\\assets\\" + imageID + "_ConstructionSatelliteImageNoPolygon.png"
    driver.save_screenshot(screenshot_path2)

    # driver.quit()


    imgUnmasked = cv2.imread("assets/" + imageID + "_ConstructionPolygonSatelliteImageUnmasked.png")

    
    image_hsv = cv2.cvtColor(imgUnmasked, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0, 0, 100])
    upper_color = np.array([10, 255, 255])

    mask = cv2.inRange(image_hsv, lower_color, upper_color)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    # imgUnmasked = imgUnmasked[y:y+h, x:x+w]

    # cv2.imwrite("assets/" + imageID + "_ConstructionPolygonSatelliteImageUnmasked1.png", imgUnmasked)
    


    fileName = os.getcwd() + "\\assets\\" + imageID + "_ConstructionPolygonSatelliteImageUnmasked.png"
    bucket = storage.bucket()
    blob = bucket.blob(f"SatelliteImages/PolygonUnmasked/{imageID}_ConstructionPolygonSatelliteImageUnmasked.png")
    blob.upload_from_filename(fileName)

    blob.make_public()

    output['satelliteImageUnmasked'] =  blob.public_url


    imgNoPoly = cv2.imread("assets/"+ imageID + "_ConstructionSatelliteImageNoPolygon.png")

    imgNoPoly = imgNoPoly[y:y+h, x:x+w]
    cv2.imwrite("assets/" + imageID + "_ConstructionSatelliteImageNoPolygon1.png", imgNoPoly)
    

    fileName = os.getcwd() + "\\assets\\" + imageID + "_ConstructionSatelliteImageNoPolygon1.png"
    bucket = storage.bucket()
    blob = bucket.blob(f"SatelliteImages/InsidePolygon/{imageID}_ConstructionSatelliteImageNoPolygon.png")
    blob.upload_from_filename(fileName)

    blob.make_public()



    print("your file url", blob.public_url)

    output['result'] = "The map has successfully been created"
    output['satelliteImageNoPolygon'] =  blob.public_url
    output['elevationList'] = elevationList

    ##########################################################################################################################################################################
    
    img = cv2.imread(os.getcwd() + "\\assets\\" + imageID + "_ConstructionPolygonSatelliteImageUnmasked.png")
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([5, 255, 255])

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    mask = np.zeros_like(red_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    result_image = cv2.bitwise_and(img, img, mask=mask)

    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2BGRA)

    black_mask = np.all(result_image[:, :, :3] == [0, 0, 0], axis=-1)
    result_image[black_mask, 3] = 0
    # result_image[black_mask, :3] = [255, 255, 255]

    cv2.imwrite(os.getcwd() + "\\assets\\" + imageID + "_ConstructionPolygonSatelliteImageMasked.png", result_image)

    fileName = os.getcwd() + "\\assets\\" + imageID + "_ConstructionPolygonSatelliteImageMasked.png"
    bucket = storage.bucket()
    blob = bucket.blob(f"SatelliteImages/PolygonMasked/{imageID}_ConstructionPolygonSatelliteImageMasked.png")
    blob.upload_from_filename(fileName)

    blob.make_public()

    print("your file url", blob.public_url)

    output['satelliteImageMasked'] =  blob.public_url

    ##########################################################################################################################################################################

    assets_dir = "./assets"
    os.makedirs(assets_dir, exist_ok=True)
    mapbox_access_token = 'sk.eyJ1Ijoic2hydXRpMDMiLCJhIjoiY2xxNXNtZjFqMGtqNzJxa3p3dnMzc3dqcSJ9.F64C4BFDjHn9Pb4A76sfNA'
    increments = [(0.01, 0.01), (-0.01, 0.01), (0.01, -0.01), (-0.01, -0.01), (0.02, 0), (0, 0.02)]
    linkList = []

    for idx, coords in enumerate(get_neighboring_coords(polygon_coords, increments), start=1):
    
        min_lat, min_lng = min(coords, key=lambda x: x[0])
        max_lat, max_lng = max(coords, key=lambda x: x[0])
        min_lng, max_lng = min(coords, key=lambda x: x[1])[1], max(coords, key=lambda x: x[1])[1]

        
        neighboring_map = folium.Map(location=[coords[0][0], coords[0][1]], zoom_start=zoom,
                                    tiles=f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{{z}}/{{x}}/{{y}}?access_token={mapbox_access_token}', attr='Mapbox Satellite')

    
        neighboring_map.fit_bounds([(min_lat, min_lng), (max_lat, max_lng)])

        neighboring_map.save(os.getcwd() + "\\assets\\" + imageID + "NeighboringTile" + str(idx) + ".html")
        driver.get(os.getcwd() + "\\assets\\" + imageID + "NeighboringTile" + str(idx) + ".html")
        time.sleep(2)
        driver.save_screenshot(os.getcwd() + "\\assets\\" + imageID + "NeighboringTile" + str(idx) + ".png")

        fileName = os.getcwd() + "\\assets\\" + imageID + "NeighboringTile" + str(idx) + ".png"
        bucket = storage.bucket()
        blob = bucket.blob(f"SatelliteImages/NeighbouringTiles/{imageID}NeighboringTile" + str(idx) +".png")
        blob.upload_from_filename(fileName)

        blob.make_public()
        linkList.append(blob.public_url)
        os.remove(os.getcwd() + "\\assets\\" + imageID + "NeighboringTile" + str(idx) + ".png")
        os.remove(os.getcwd() + "\\assets\\" + imageID + "NeighboringTile" + str(idx) + ".html")

    output["neighboringTiles"] = linkList
    driver.quit()

    ##########################################################################################################################################################################

    os.remove(os.getcwd() + "\\assets\\" + imageID + "_ConstructionPolygonSatelliteImageMasked.png")
    os.remove(os.getcwd() + "\\assets\\" + imageID + "_ConstructionPolygonSatelliteImageUnmasked.png")
    os.remove(os.getcwd() + "\\assets\\" + imageID + "_ConstructionSatelliteImageNoPolygon.png")
    os.remove(os.getcwd() + "\\assets\\" + imageID + "_ConstructionSatelliteImageNoPolygon1.png")
    os.remove(os.getcwd() + "\\assets\\satelliteMap.html")
    os.remove(os.getcwd() + "\\assets\\satelliteMapNoPolygon.html")

    return jsonify(output)

#########################################################################################################################################################################################

@app.route('/treeEnumeration',methods=['GET'])
def treeEnumerationScript():
    imageLink = str(request.args['imageLink'])
    imageID = str(request.args['ProjectID'])

    path_1 = "D:\\SIHMODELS\\TreeEnumeration"
    sys.path.append(path_1)
    req = urllib.request.urlopen(imageLink)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img1 = cv2.imdecode(arr, -1)

    if img1.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB)

    path_to_model = "D:/SIHMODELS/FinalModel.pth"
    loaded_model = main.deepforest()
    loaded_model.model.load_state_dict(torch.load(path_to_model))
    loaded_model.eval()
    
    box_info2 = loaded_model.predict_image(img1, return_plot=False)
    num_trees2 = len(box_info2)
    print("Number of trees:", num_trees2)
    
    output = {}
    output['treeCount'] = num_trees2

    img1 = loaded_model.predict_image(img1, return_plot=True)
    
    path_for_images = os.getcwd() + "\\assets\\" + imageID + "_TreeEnumeratedImage.png"

    cv2.imwrite(path_for_images, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))

    fileName = os.getcwd() + "\\assets\\" + imageID + "_TreeEnumeratedImage.png"
    bucket = storage.bucket()
    blob = bucket.blob(f"EnumerationImages/{imageID}_TreeEnumeratedImage.png")
    blob.upload_from_filename(fileName)

    blob.make_public()

    print("your file url", blob.public_url)

    output['enumeratedImageLink'] =  blob.public_url

    os.remove(os.getcwd() + "\\assets\\" + imageID + "_TreeEnumeratedImage.png")

    return jsonify(output)

#########################################################################################################################################################################################


@app.route('/speciesDetection',methods=['GET'])
def speciesDetection():
    custom_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        }

    data_dir = "D:/ExtendedBarkDataset"
    train_dataset = datasets.ImageFolder(
    root=data_dir, transform=custom_transforms["train"],)
        
    class_names = train_dataset.classes
    model_transfer = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_transfer.fc = nn.Linear(2048, len(class_names))
    model_transfer.load_state_dict(torch.load("D:/SIH2023/model_transfer12345.pt"))
    image_path = "D:/ExtendedBarkDataset/Acacia/IMG_6348.JPG"
    predicted_species = predict_image_class(image_path, model_transfer, class_names)
    print(f"Predicted Tree Species: {predicted_species}")



if __name__ == '__main__':
    app.run()