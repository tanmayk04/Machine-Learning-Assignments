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
	print('START Q1_D\n')
	# to import dataset
	with open('train_dat.txt', 'r') as f:
		ip_data = f.readlines()
		ls=[]
		for i in ip_data:
			res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
			ls.append(res)

	ip=[]
	for i in ls:
		lgt=[]
		for j in i:
			lgt.append(float(j))
		ip.append(lgt)
	gphx,gphy=[],[]
	for i in range(20):
		gphx.append(ip[i][0])
		gphy.append(ip[i][1])
	p=[]
	lsMin=20000
	# part b
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
			
	with open('test_dat.txt', 'r') as f:
		ip_data = f.readlines()
		ls=[]
		for i in ip_data:
			res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
			ls.append(res)

	test=[]
	for i in ls:
		lgt=[]
		for j in i:
			lgt.append(float(j))
		test.append(lgt)
	testX,testY=[],[]
	for i in ip:
		testX.append(i[0])
		testY.append(i[1])
	p=[]
	lsMin=20000
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
			
			u=ev_reg(gphy,preds)
			lsMin=min(u,lsMin)
			p.append([lgt,m,u])
			
	res=[]
	for j in range(len(p)):
		if lsMin==p[j][2]:
			res=p[j]
	for i in p:
		print('for k='+str(p[0])+' fo='+str(p[1])+', error:'+str(p[2]))
	print(res)
	print(f'minimum eror is at k={res[0]}, fo={res[1]}, error={res[2]}')



	print('END Q1_D\n')


if __name__ == "__main__":
    main()