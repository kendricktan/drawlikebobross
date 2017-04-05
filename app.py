import re
import argparse
import base64
import numpy as np

from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, send_file
from aae.train import trainer, transformers

# Parse args
parser = argparse.ArgumentParser(description='GAN trainer')
parser.add_argument('--resume', default='', type=str)

app = Flask(__name__)


# Index page
@app.route('/')
def index():
    return render_template('index.html')


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG')
    return base64.b64encode(img_io.getvalue())


# GAN interactive endpoint
@app.route('/gan', methods=['POST'])
def generate():
    global trainer

    if 'img' in request.form:
        # Prepare to convert base64 png to image file
        img_data = re.sub('^data:image/.+;base64,', '', request.form['img'])
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data))
        img = img.convert("RGB")

        img_np = np.asarray(img)

        rimg = trainer.reconstruct(img_np, transformers=transformers)
        return serve_pil_image(rimg)

    return {'error': 'img not found'}


if __name__ == '__main__':
    # Initialize args and load parser
    args, unknown = parser.parse_known_args()
    if args.resume:
        trainer.load_(args.resume)

    app.run()
