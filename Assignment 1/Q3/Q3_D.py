import numpy as npm
import math as mt
import matplotlib.pyplot as pt


def main():
	print('START Q3_D\n')
	
	al= 0.01
	# to import dataset
	with open('Q3_data.csv', 'r') as f:
			ip_data = f.readlines()
			lgts=[]
			for i in ip_data:
				res=i.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
				lgts.append(res)
	ip=[]
	for i in lgts:
		lgt=[]
		for j in i:
			lgt.append(j)
		ip.append(lgt)
	ht, wt,gen=[],[],[]
	counter=0
	for i in ip:
		ht.append(float(i[0]))
		wt.append(float(i[1]))
		gen.append(i[3])
	
	X = npm.asarray([ht, wt]).T
	Y = npm.asarray(gen)
	incr=[]
	
	srt = npm.zeros(X.shape[1])
	for j in range(30):
		r= 0
		for i in range(len(X)):
			#calculating probability by transforming it to a sigmoid function
			arv = (1/(1+(mt.exp(-npm.dot(srt, X[i])))))
			anum = 1 if Y[i] == 'wt' else 0
			if arv >= 0.5:
				if anum == 1:
					r += 1
			if arv<=0.5:
				if anum==0:
					r += 1
			t1=(anum-arv)
			srt += al * t1 * X[i]
	incr.append(r)
	h1,w1,c1=[],[],[]
	for i in range(len(ht)):
		#turning the results to numbers and retaining the thrashhold value of 0.5
		res=(1/(1+(mt.exp(-npm.dot(srt,[ht[i],wt[i]])))))
		#If the value is greater than 0.5, make it male and set its value to 1.
		h1.append(ht[i])
		w1.append(wt[i])
		if res>=0.5:
			c1.append('M')
		#If the value is less than 0.5, make it male and set its value to 0.
		else:
			c1.append('F')
	pt.figure(figsize=(5,9), dpi=100)
	xes = pt.axes(projection ='3d')
	for j in range(len(ht)):
		if c1[j]=='M':
			xes.plot(ht[j],wt[j],'.b')
		else:
			xes.plot(ht[j],wt[j],'.g')

	arrpt=[]
	for i in range(len(ht)):
		rcv=[]
		g=[]
		for j in range(len(ht)):
			if i!=j:
			#turning the results to numbers and retaining the thrashhold value of 0.5
				rcv.append([ht[j],ht[j]])
				g.append(gen[j])

		X = npm.asarray(rcv)
		Y = npm.asarray(g)
		r= 0
		srt = npm.zeros(X.shape[1])
		for j in range(30):
			for i in range(len(X)):
				#calculating probability by transforming it to a sigmoid function
				arv =  (1/(1+(mt.exp(-npm.dot(srt, X[i])))))
				anum = 1 if Y[i] == 'wt' else 0
				if arv >= 0.5:
					if anum == 1:
						r += 1
				if arv<=0.5:
					if anum==0:
						r += 1
				# 
				t1=(anum-arv)
				srt += al * t1 * X[i]
		
		c1=[]
		for k in range(len(X)):
			res=(1/(1+(mt.exp(-npm.dot(srt, X[k])))))
			#If the value is greater than 0.5, make it male and set its value to 1.
			if res>=0.5:
				c1.append('M')
		#If the value is less than 0.5, make it male and set its value to 0.
			else:
				c1.append('F')
		w=0
		for z in range(len(c1)):
			if c1[z]!=Y[z]:
				w+=1
		arrpt.append([srt.tolist(),w/len(c1)])
		mw=mt.inf
		rqrd=[]
		for j in range(len(arrpt)):
			if mw>arrpt[j][1]:
				mw=arrpt[j][1]
				rqrd=arrpt[j]
		rqrdpt1=[]
	lst=[]
	for x in range(len(ht)):
		lst.append([ht[x],wt[x]])
	rlst=npm.asarray(lst)
	print('minimum error:',mw)
	for x in range(len(rlst)):
		t1=(1/(1+(mt.exp(-npm.dot(rqrd[0], rlst[x])))))
		if t1>=0.5:
			res=1
		else:
			res=0
		rqrdpt1.append([rlst[x].tolist(),res])
	xes = pt.axes(projection ='3d')
	print(arrpt)
	for z in range(len(rqrdpt1)):
		if rqrdpt1[z][1]==1:
			xes.plot(rqrdpt1[z][0][0],rqrdpt1[z][0][1],'.b')
		else:
			xes.plot(rqrdpt1[z][0][0],rqrdpt1[z][0][1],'.y')
	print('minimum error:',incr[0]/len(rqrdpt1))
	print('END Q3_D\n')
	

if __name__ == "__main__":
    main()
    