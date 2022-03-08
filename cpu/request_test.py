#make a POST request
import requests
import json
import matplotlib.pyplot as plt
from base64 import decodebytes, b64decode
import io
from PIL import Image
from time import time
from tqdm import tqdm
t0=time()
# for i in tqdm(range(1)):
# prompt = {'prompt':'the Eiffel tower on the moon','n_images':32}
prompt = {'prompt':'A blue table','n_images':32}

# prompt = {'cat':'A blue table'}
# res = requests.post('http://localhost:5000/generate', json=prompt)
res = requests.post('http://3.227.11.255:5000/generate',json=prompt)
res_json = json.loads(res.text)
print("Time: {:.3f} seconds".format(time()-t0))
if 'result' in res_json:
    for i in res_json['result']:
        im = Image.open(io.BytesIO(b64decode(i))) 
        plt.imshow(im)
        plt.show()
    # print('response from server:',len(res_json))
else:
    print(res.text)
# dictFromServer = res.json()