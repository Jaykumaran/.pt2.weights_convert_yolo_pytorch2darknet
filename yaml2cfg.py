import torch
import struct
import yaml
import os
from configparser import ConfigParser

# Input files
pt_file_name = "/content/drive/MyDrive/EDU VID Sample/yolov8n.pt"
yaml_file_name = "/content/drive/MyDrive/EDU VID Sample/yolov8.yaml"

# Output files
cfg_file_name = "yolov8.cfg"
weights_file_name = "yolov8n.weights"

if os.path.exists(cfg_file_name):
    os.remove(cfg_file_name)

if os.path.exists(weights_file_name):
    os.remove(weights_file_name)



# Config file conversion (YAML to CFG)
with open(cfg_file_name, 'w') as cfg:
    with open(yaml_file_name,'r') as f:
        f = yaml.load(f,Loader=yaml.FullLoader)
        net = 0
        for l in f:
            if ((l != 'anchors')and(l !='backbone')and(l !='head')):
                if net == 0:
                    cfg.write('\n[net]\n')
                    cfg.write("width={}\n".format(416))
                    cfg.write("height={}\n".format(416))
                    cfg.write("channels={}\n".format(3))
                    net =1
                cfg.write("{}={}\n".format(l,f[l]))
            elif (l == 'anchors'):
                anchors = []
                for v in f[l]:
                    anchors.extend(v)
                cfg.write("anchors={}\n".format(str(anchors)[1:-1]))
            elif((l == 'backbone') or (l =='head')):
                for vector in f[l]:
                    #net type
                    cfg.write('\n[{}]\n'.format(vector[2]))
                    #from
                    if type(vector[0]) == int:
                        cfg.write("from={}\n".format(str(vector[0])))
                    elif type(vector[0]) == list:
                        cfg.write("from={}\n".format(str(vector[0])[1:-1]))
                    #number
                    if type(vector[1]) == int:
                        cfg.write("number={}\n".format(str(vector[1])))
                    elif type(vector[1]) == list:
                        cfg.write("number={}\n".format(str(vector[1])[1:-1]))
                    #params
                    if type(vector[3]) == int:
                        cfg.write("args={}\n".format(str(vector[3])))
                    elif type(vector[3]) == list:
                        cfg.write("args={}\n".format(str(vector[3])[1:-1]))

with open(cfg_file_name,'r') as f:
    for l in f.readlines():
        print(l)