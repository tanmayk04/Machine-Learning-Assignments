from matplotlib import pyplot as plt
import numpy as np
import math

# function for calculating Logically weighted Linear regression
def ev_lwlr(g,find):
	for x in range(len(g)):
		fo=(g[x]-find[x])**2
	val=fo/len(g)
	return val
# main function
def main():
	print('START Q2_C\n')
	e=[]
	# to import dataset
	with open('train_dat.txt', 'r') as f:
		ip_data = f.readlines()
		lgts=[]
		for i in ip_data:
			res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
			lgts.append(res)
	X=[]
	for i in lgts:
		lgt=[]
		for j in i:
			lgt.append(float(j))
		X.append(lgt)
	gphx,gphy=[],[]
	for i in X:
		gphx.append(i[0])
		gphy.append(i[1])
	# after getting datas

	with open('test_dat.txt', 'r') as f:
		ip_data = f.readlines()
		lgts=[]
		for i in ip_data:
			res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
			lgts.append(res)
	test=[]
	for i in lgts:
		lgt=[]
		for j in i:
			lgt.append(float(j))
		test.append(lgt)
	tstY,tstY=[],[]
	for i in test:
		tstY.append(i[0])
		tstY.append(i[1])

	lsWr=10000
	vs=[]
	for i in range(len(tstY)):
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
		t=vs[m][0]+vs[m][1]*tstY[m]
		u.append(t)
	e=ev_lwlr(tstY,u)
	plt.scatter(tstY,u,marker='.',label= 'predicted')
	plt.scatter(tstY,tstY,marker='.',label= 'original')
	plt.show()
	print('the error:',e)
	print('END Q2_C\n')


if __name__ == "__main__":
    main()
    