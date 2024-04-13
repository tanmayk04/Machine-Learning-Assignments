from matplotlib import pyplot as ptt
import numpy as np
import math

# function for calculating Logically weighted Linear regression
def ev_lwlr(g,find):
	v1=0
	for ip in range(len(g)):
		fo=(g[ip]-find[ip])**2
		fv=math.sqrt(fo)
		v1+=fv
	val=fv/len(g)
	return val

# main function
def main():
	print('START Q2_AB\n')
	# to import dataset
	with open('train_dat.txt', 'r') as f:
			ip_data = f.readlines()
			lgts=[]
			for i in ip_data:
				res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
				lgts.append(res)
	ip=[]
	for i in lgts:
		lgt=[]
		for j in i:
			lgt.append(float(j))
		ip.append(lgt)
	gphx,gphy=[],[]
	for i in ip:
		gphx.append(i[0])
		gphy.append(i[1])
	# after getting datas

	vs=[]
	e=[]
	for i in range(len(gphx)):
		v=[]
		w,x,y,z=[],[],[],[]
		for j in range(len(gphy)):
			v1=math.pow((gphx[j]-gphx[i]),2)
			a2=math.pow(0.204,2)
			a3=a2*2
			res=np.exp(-v1/a3)
			x.append(res)
		for k in range(len(x)):
			q=gphx[k]*x[k]
			w.append(q)
			z.append([1,q])
			y.append([gphy[k]*x[k]])
		z=np.matrix(z)
		y=np.mat(y)
		rt=np.linalg.pinv(z)
		result=np.dot(rt,y)
		v1=np.array(result)
		v=[]
		for i in v1:
			v.append(i[0])
		vs.append(v)
	u=[]
	for m in range(len(vs)):
		
		t=vs[m][0]+vs[m][1]*gphx[m]
		u.append(t)
	e=ev_lwlr(gphx,u)
	ptt.scatter(gphx,u,marker='.',label= 'solved')
	ptt.scatter(gphx,gphy,marker='.',label= 'given')
	ptt.legend()
	ptt.show()
	print(e)
	print('END Q2_AB\n')


if __name__ == "__main__":
    main()
