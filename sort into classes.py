from PIL import Image 
import xml.etree.ElementTree as ET
import os
for i in range(1,1445):
    try:
        #print(i)
        s=str(i)

        while len(s)<4:
            s="0"+s

        tree = ET.parse("/Users/abhinav/Documents/Labeled Data CEERI/GDXRAY--Images-Labelled/" +s + ".xml")
        root = tree.getroot()

        #os.makedir("./"+root[6][0].text+"/")
        #root[6][0].text!="gun":
        try:
            os.mkdir("./"+root[6][0].text+"/")
        except Exception as e:
            print(e)
        im = Image.open(r"/Users/abhinav/Documents/Labeled Data CEERI/GDXRAY--Images-Labelled/"+s+".png") 
        # # width, height = im.size 
        im.save(("./"+root[6][0].text+"/"+str(s)+".png"))
    except Exception as e:
        print("Error for "+str(i),e)
