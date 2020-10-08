import  cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

x = [1597,2091,2433,2329]
y = [170,291,839,1336]
x1 = [2055,2171,1839,1498]
y1 = [1335,930,546,388]


x,y = np.array(x),np.array(y)
x1,y1 = np.array(x1),np.array(y1)
plt.figure(0)
# plt.xlim(0,6)
# plt.ylim(0,6)
plt.scatter(x,y)
plt.scatter(x1,y1)
t = np.linspace(0,1,100)

def cal(p,t):
    T = p[0]*np.power((1-t),3)+3*t*p[1]*np.power((1-t),2)+3*(1-t)*np.power(t,2)*p[2]+\
        np.power(t,3)*p[3]
    return T

t1 = cal(x,t)
t2 = cal(y,t)
plt.plot(t1,t2)
t1 = cal(x1,t)
t2 = cal(y1,t)
plt.plot(t1,t2)

#==============================================================================
# img = cv.imread('E:/dataset/ctw1500/train/text_image/0013.jpg')
img = plt.imread('E:/dataset/ctw1500/train/text_image/0013.jpg')
p0 = [1496,171]
p = [101,0,363,106,546,262,693,441,793,649,847,886,833,1166,558,1164,580,938,537,751,446,592,323,452,189,319,0,218]
d ,tmp= [],[]
for i, n in enumerate(p):
    if i % 2==0:
        p[i] += p0[0]
        tmp = []
        tmp.append(p[i])
    else:
        p[i] += p0[1]
        tmp.append(p[i])
        d.append(tmp)
d =  np.array(d)
# cv.polylines(img,[d],1,(0,255,0),10)
plt.imshow(img)
du = d[0:7,:]
dl = d[7:,:]   
def CalcT(d):
    distanceu, tu = [0.0], []
    distancel,tl = [0.0], []
    p1 = 0
    p2 = 0
    for i,n in enumerate(d):
        if i<7:
            if i == 0:
                p1 = d[i]
                continue
            p2 = d[i]
            tmp = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            tmp = np.sqrt(tmp)
            p1 = p2 
            distanceu.append(tmp)
        else:
            if i == 7:
                p1 = d[i]
                continue
            p2 = d[i]
            tmp = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            tmp = np.sqrt(tmp)
            p1 = p2 
            distancel.append(tmp)
            
    D = sum(distanceu)
    for i,n in enumerate(distanceu):
        tmp = n/D
        if i==0:
            tu.append(tmp)
        else:
            tmp += tu[i-1]
            tu.append(tmp)
    D = sum(distancel)
    for i,n in enumerate(distancel):
        tmp = n/D
        if i==0:
            tl.append(tmp)
        else:
            tmp += tl[i-1]
            tl.append(tmp)
    return tu, tl

def getBernstainPolynomialsMat(tu, tl):
    mu, ml = [], []
    for i,j in zip(tu, tl):
        tmpu, tmpl = [], []
        ti, tj = np.power(1-i,3), np.power(1-j,3)
        tmpu.append(ti)
        tmpl.append(tj)
        ti, tj = 3.0*i*np.power(1-i,2), 3.0*j*np.power(1-j,2)
        tmpu.append(ti)
        tmpl.append(tj)
        ti, tj = 3.0*(1-i)*np.power(i,2), 3.0*(1-j)*np.power(j,2)
        tmpu.append(ti)
        tmpl.append(tj)
        ti, tj = np.power(i,3), np.power(j,3)
        tmpu.append(ti)
        tmpl.append(tj)
        mu.append(tmpu)
        ml.append(tmpl)
    mu, ml = np.mat(mu), np.mat(ml)
    return mu, ml

def getControlPoints(M, d):
    cp = []
    mt = M
    d = np.mat(d)
    M = M.T*M
    d = mt.T * d
    cp = M.I * d
    return cp



tu,tl = CalcT(d)
Mu, Ml = getBernstainPolynomialsMat(tu, tl)
cu = getControlPoints(Mu, du)
cl = getControlPoints(Ml, dl)



a = 0

