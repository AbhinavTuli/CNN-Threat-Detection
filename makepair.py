import os
import shutil
entries=os.listdir('./')
i=0
# for entry in entries:
    # if entry[0]!='B':
    #     continue
    # k=int(entry[-7:-4])%90
    # l=int(entry[-7:-4])/90
    # os.rename(entry,str(l)+" "+str(k)+'.jpg') 
r=entries[0][:-7]
print(r)

for i in range(180):
    os.mkdir("./"+str(2*i))
    if((2*i)<10):
        s='00'+str(2*i)
    elif ((2*i)<100):
        s='0'+str(2*i)
    else:
        s=str(2*i)
    j=(2*i+90)%360
    if(j<10):
        u='00'+str(j)
    elif (j<100):
        u='0'+str(j)
    else:
        u=str(j)
    f1=r+s+'.jpg'
    f2=r+u+'.jpg'
    shutil.copyfile("./"+f1,"./"+str(2*i)+"/"+f1)
    shutil.copyfile("./"+f2,"./"+str(2*i)+"/"+f2)
    print(f1+" "+f2)
