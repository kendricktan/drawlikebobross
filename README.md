# Draw like Bob Ross using the power of Neural Networks!

![Sample](https://i.imgur.com/aU8ySZN.png)

## Learning Process Visualization

![webm](https://thumbs.gfycat.com/DefenselessEminentKookaburra-size_restricted.gif)

# Getting started
## Install dependecies
### Requires python3.x
```
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl 
pip install torchvision
```

# Run server (using pretrained model)
```
python app.py --resume pretrained_models/450epoch_aae.tar
```

Navigate to 127.0.0.1:5000

# Run server (from scatch)
## 1. Scrapping data 
```
./scrapper.sh
```

## 2. Preprocess data (should take around 5-10 mins)
```
python preprocess.py
```

## 3. Train data
```
cd aae
python train.py
```

## 4. Run server
```
cd PROJECT_ROOT
python app.py --resume TRAINEDMODEL.path.tar
```

Navigate to 127.0.0.1:5000
