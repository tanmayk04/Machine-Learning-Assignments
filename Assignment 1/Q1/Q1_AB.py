from matplotlib import pyplot as plt
import numpy as np
import math

def main():
	print('START Q1_AB\n')
	k=int(input('please enter the value of k(freq. increment)'))
	d=int(input('please enter the value of d(function depth)'))
	# to import dataset
	with open('train_dat.txt', 'r') as f:
			ip_data = f.readlines()
			lgt=[]
			for i in ip_data:
				res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
				lgt.append(res)
	ip=[]
	for i in lgt:
		lgt=[]
		for j in i:
			lgt.append(float(j))
		ip.append(lgt)
	gphx,gphy=[],[]
	for i in ip:
		gphx.append(i[0])
		gphy.append(i[1])

	# Q1:part-a
	for m in range(k):
		for n in range(d):
			h=m
			g=[]
			for j in range(len(gphx)):
				pt2=[]
				lgt3=[]
				for i in gphx:
					lgt3.append([i])
				lgt3=np.mat(lgt3)
				for j in range(len(gphx)):
					pt=[]
					pt.append(1)
					for i in range(1,n+1):
						res=math.pow(math.sin(i*gphy[j]*h),2)
						pt.append(res)
					pt2.append(pt)
				pt2=np.mat(pt2)
				inpt2=np.linalg.pinv(pt2)
				c=np.dot(inpt2,lgt3)
				arr=c.tolist()
				g=[]
				for i in arr:
					g.append(i)
			pred=[]
			for j in range(len(gphx)):
				res=g[0][0]
				for q in range(1,len(g)):
					res+=float(g[q][0])*float(math.pow((math.sin(q*lgt*gphx[j])),2))
				pred.append(res)
			arr=np.array(pred)
			z=arr.tolist()
			
# Q2: part-b
	for lgt in range(1,11):
		for m in range(0,7):
			h=lgt
			g=[]
			for j in range(len(gphx)):
				pt2=[]
				lgt3=[]
				for i in range(len(gphx)):
					lgt3.append([gphx[i]])
				lgt3=np.mat(lgt3)
				for j in range(len(gphx)):
					pt=[]
					pt.append(1)
					for i in range(1,m+1):
						res=math.pow(math.sin(i*gphy[j]*h),2)
						pt.append(res)
					pt2.append(pt)
				pt2=np.mat(pt2)
				inpt2=np.linalg.pinv(pt2)
				c=np.dot(inpt2,lgt3)
				arr=c.tolist()
				g=[]
				for i in arr:
					g.append(i)
			pred=[]
			for j in range(len(gphx)):
				res=g[0][0]
				for q in range(1,len(g)):
					res+=float(g[q][0])*float(math.pow((math.sin(q*lgt*gphx[j])),2))
				pred.append(res)
			arr=np.array(pred)
			z=arr.tolist()
			preds=[]
			for i in z:
				preds.append(i)
			plt.scatter(gphx,preds,marker='.',label= m)
			plt.legend()
		plt.show()
	print('END Q1_AB\n')


if __name__ == "__main__":
    main()