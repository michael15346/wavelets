import numpy as np
from scipy.signal import convolve
import imageio.v3 as iio


def downscale(a, Minv, base):

    #Здесь и в апскейлинге добавить сдвиг по координате!!!!! 
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for i in range(a.shape[0]): # range is wrong
        for j in range(a.shape[1]):
            x = Minv @ np.array([i, j])
            xmin = min(xmin, x[0])
            xmax = max(xmax, x[0])
            ymin = min(ymin, x[1])
            ymax = max(ymax, x[1])
    ares = np.zeros((int(xmax - xmin + 1), int(ymax - ymin + 1)))
    for i in range(a.shape[0]): # range is wrong
        for j in range(a.shape[1]):
            x = Minv @ np.array([i, j])
            ares[int(x[0] - xmin), int(x[1] - ymin)] = a[i, j]
    base = Minv @ base
    return (ares, base)
    
def upscale(a, M, base):
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for i in range(a.shape[0]): # range is wrong
        for j in range(a.shape[1]):
            x = M @ np.array([i, j])
            xmin = min(xmin, x[0])
            xmax = max(xmax, x[0])
            ymin = min(ymin, x[1])
            ymax = max(ymax, x[1])


    ares = np.zeros((int(xmax - xmin + 1), int(ymax - ymin + 1)))
    for i in range(a.shape[0]): # range is wrong
        for j in range(a.shape[1]):
            x = M @ np.array([i, j])
            ares[int(base[0] + x[0] - xmin), int(base[1] + x[1] - ymin)] = a[i, j]
    base = M @ base
    return (ares, base)

def dwt(a0, h, g, M, level):
    d = list()
    padh_x = h.shape[0] - 1
    padh_y = h.shape[1] - 1
    padg_x = g.shape[0] - 1
    padg_y = g.shape[1] - 1
    base_di = np.array([0,0])
    base_ai = np.array([0,0])
    Minv = np.linalg.inv(M)
    ai = (a0, base_ai)
    for i in range(level):
        (ah, base_ai) = ai
        (ag, _) = ai
        #ah = np.pad(ai, ((padh_x, padh_x), (padh_y, padh_y)), 'edge')
        #ag = np.pad(ai, ((padg_x, padg_x), (padg_y, padg_y)), 'edge')
        print("ag", ag)
        print("ah", ah)
        di = downscale(convolve(ag, g, 'full', method='direct'), Minv, base_di)
        base_di = di[1]
        iio.imwrite(f'di{i}.png', di[0].astype(np.uint8))
        d.append(di)
        ai = downscale(convolve(ah, h, 'full', method='direct'), Minv, base_ai)
        #base_ai = ai[1]
    return (ai, d)


def idwt(an, d, h, g, M):
    padh_x = h.shape[0] - 1
    padh_y = h.shape[1] - 1
    padg_x = g.shape[0] - 1
    padg_y = g.shape[1] - 1

    ai = an

    for (di, base_di) in d:
        (ah, base_ai) = ai #np.pad(ai, ((padh_x, padh_x), (padh_y, padh_y)), 'edge')
        (ag, _) = ai #np.pad(ai, ((padg_x, padg_x), (padg_y, padg_y)), 'edge')
        iio.imwrite('ai.png', (ag / 10).astype(np.uint8))
        iio.imwrite('aiup.png', (upscale(ag,M, base_ai))[0].astype(np.uint8))
        #(di, base_di) = upscale(di, M, base_di)
        di = convolve(di, g, 'full', method='direct')


        (ai, base_ai) = upscale(ah, M, base_ai)
        #d.append(di)
        ai = convolve(ai, h, 'full', method='direct')# + di, base_ai)
        di = np.pad(di, ((int(base_di[0] - base_ai[0]), int(ai.shape[0] - di.shape[0] - base_di[0] + base_ai[0])), (int(base_di[1] - base_ai[1]), int(ai.shape[1] - di.shape[1] - base_di[1] + base_ai[1]))), 'constant')
        print('base_di', base_di)
        print('base_ai', base_ai)
        print('ah.shape', ah.shape)
        print('di.shape', di.shape)
        ai = (ai + di, base_ai)
    return ai

#data = iio.imread('http://upload.wikimedia.org/wikipedia/commons/d/de/Wikipedia_Logo_1.0.png')
#print(data)
data = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

M = np.array([[1, -1], [1,1]])

h = np.array([[0.25], [0.5], [0.25]]) 
g = np.array([[-0.125], [-0.25], [0.75], [-0.25], [-0.125]])
hdual = np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]])
gdual = np.array([[-0.25], [0.5], [-0.25]])

ai, d = dwt(data, h, g, M, 5)
print(ai, d)
a = idwt(ai, d, hdual, gdual, M)
iio.imwrite('image.png', a.astype(np.uint8))

