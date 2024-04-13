from matplotlib import pyplot as plt
import numpy as np
import math

# function for calculating Linear regression
def ev_reg(g,find):
	v1=0
	for ip in range(len(g)):
		fo=(g[ip]-find[ip])**2
		fv=math.sqrt(fo)
		v1+=fv
	val=fv/len(g)
	return val

# main function
def main():
	print('START Q1_C\n')
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
		
	with open('test_dat.txt', 'r') as f:
		ip_data = f.readlines()
		lgt=[]
		for i in ip_data:
			res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
			lgt.append(res)

	test=[]
	for i in lgt:
		lgt=[]
		for j in i:
			lgt.append(float(j))
		test.append(lgt)
	tx,ty=[],[]
	for i in ip:
		tx.append(i[0])
		ty.append(i[1])
	p=[]
	lomin=20000
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
			for j in range(len(tx)):
				res=g[0][0]
				for q in range(1,len(g)):
					res+=float(g[q][0])*float(math.pow((math.sin(q*lgt*tx[j])),2))
				pred.append(res)
			arr=np.array(pred)
			z=arr.tolist()
			preds=[]
			for i in z:
				preds.append(i)
			
			u=ev_reg(gphy,preds)
			lomin=min(u,lomin)
			p.append([lgt,m,u])
	mnrLK=[]
	for j in range(len(p)):
		if lomin==p[j][2]:
			mnrLK=p[j]
	
	print(p)
	print(mnrLK)
	print(f'minimum eror is at k={mnrLK[0]}, fo={mnrLK[1]}, error={mnrLK[2]}')

	print('END Q1_C\n')


if __name__ == "__main__":
    main()
    