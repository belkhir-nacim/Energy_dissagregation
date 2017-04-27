import numpy as np
import matplotlib.pyplot as plt
def exo1():
    def fact(n):
        if n<2:
            return 1
        else:
            return n*fact(n-1)
    print(fact(30))

def exo2():
    def sorted(x):
        i = np.argsort(x)
        y = np.asarray(x)[i]
        return y, i
    a = np.random.random_integers(-5,100,10)
    i,j = sorted(a)

def exo3():
    def min_tree(X, minn=np.inf):
        if np.isscalar(X):
            minn = X  if X <minn else minn
            return minn
        else:
            l = []
            for x in X:
                l.append(min_tree(x,minn=minn))
            return np.min(l)

    a = [[1,2,3], [1, 2, 3], [1,[[[[[[-1000]]]]]],  3], [[ [-100,   ],-1 ],100,40],1e4]
    min_tree(a)



def generate_one_hour():
    x = []
    for t in range(1,3601):
        if (t<0) or (t > 900):
            x.append(0)
        else:
            x.append(200 * (1+ np.exp(-t/10)))

    return x

def get_wholeday():
    x = []
    for h in range(24):
        x+=generate_one_hour()
    return x


def get_a():
    x = np.zeros(3600*24)
    return x

def get_b():
    return get_a()+1000


def get_c():
    x = []
    for h in range(24):
        for t in range(1, 3601):
            x.append(1000 * np.cos(2 * np.pi * (t / 3600.)))
    return np.array(x)


def get_d():
    x = []
    for h in range(24):
        for t in range(1, 3601):
            x.append(0+ 15*np.random.rand())
    return np.array(x)


def get_e():
    return get_c() + get_d()

plt.plot(get_e()+get_wholeday())
plt.show()