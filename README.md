# drawlikebobross
Draw like Bob Ross using the power of GANs!

# Install dependecies
### Requires python3.x
```
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl 
pip install torchvision
```

# Steps
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
cd gan
python train.py
```

## 4. Run server
```
cd PROJECT_ROOT
python app.py --resume TRAINEDMODEL.path.tar
```
