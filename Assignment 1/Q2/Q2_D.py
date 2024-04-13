from matplotlib import pyplot as plt
import numpy as np
import math

# function for calculating Logically weighted Linear regression
def ev_lwlr(g,find):
	v1=0
	for x in range(len(g)):
		fo=(g[x]-find[x])**2
		fv=math.sqrt(fo)
		v1+=fv
	val=fv/len(g)
	return val

# main function
def main():
	print('START Q2_D\n')
	# to import dataset
	with open('train_dat.txt', 'r') as f:
		ip_data = f.readlines()
		lgts=[]
		for i in ip_data:
			res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
			lgts.append(res)
	ip=[]
	for i in range(20):
		lgt=[]
		for j in lgts[i]:
			lgt.append(float(j))
		ip.append(lgt)
	gnX,gnY=[],[]
	for i in ip:
		gnX.append(i[0])
		gnY.append(i[1])
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
	tstX,tstY=[],[]
	for i in test:
		tstX.append(i[0])
		tstY.append(i[1])
	# part a2
	vs=[]
	for i in range(len(gnX)):
		v=[]
		Xwgt,wgt,tp2,tp1=[],[],[],[]
		for j in range(len(gnY)):
			t1=math.pow((gnX[j]-gnX[i]),2)
			t2=math.pow(0.204,2)
			t3=t2*2
			res=np.exp(-t1/t3)
			wgt.append(res)
			# print(res)
		for k in range(len(wgt)):
			tp=gnX[k]*wgt[k]
			Xwgt.append(tp)
			tp1.append([1,tp])
			tp2.append([gnY[k]*wgt[k]])
		tp1=np.matrix(tp1)
		tp2=np.mat(tp2)
		itp1=np.linalg.pinv(tp1)
		result=np.dot(itp1,tp2)
		v1=np.array(result)
		v=[]
		for i in v1:
			v.append(i[0])
		# print(v)
		vs.append(v)
	u=[]
	for m in range(len(vs)):
		t=vs[m][0]+vs[m][1]*gnX[m]
		u.append(t)
	plt.scatter(gnX,u,marker='.',label= 'pred')
	plt.scatter(gnX,gnY,marker='.',label= 'given')
	plt.show()
	# partc a3
	lsWr=10000
	vs=[]
	for i in range(len(tstX)):
		v=[]
		w,x,y,z=[],[],[],[]
		for j in range(len(gnY)):
			v1=math.pow((gnX[j]-gnX[i]),2)
			a2=math.pow(0.204,2)
			a3=a2*2
			res=np.exp(-v1/a3)
			x.append(res)
		for k in range(len(x)):
			q=gnX[k]*x[k]
			w.append(q)
			z.append([1,q])
			y.append([gnY[k]*x[k]])
		z=np.matrix(z)
		y=np.mat(y)
		rt=np.linalg.pinv(z)
		result=np.dot(rt,y)
		v1=np.array(result)
		v=[]
		for i in v1:
			v.append(i[0])
		vs.append(v)
		v=[]
		for i in v1:
			v.append(i[0])
		# print(v)
		vs.append(v)
	u=[]
	for m in range(len(vs)):
		t=vs[m][0]+vs[m][1]*gnX[m]
		u.append(t)
	e=ev_lwlr(tstY,u)
	# plt.scatter(tstY,u,marker='.',label= 'pred')
	# plt.scatter(tstX,tstY,marker='.',label= 'given')
	# plt.show()
	print('the error:',e)
	print('END Q2_D\n')


if __name__ == "__main__":
    main()
    