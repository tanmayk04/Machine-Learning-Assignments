from matplotlib import pyplot as plt
import numpy as np
import math as mt

# main function
def main():
	print('START Q3_AB\n')
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

	ht, wt, u,g=[],[],[],[]
	# Separate each parameter so that it can be used in different ways
	for i in ip:
		ht.append(float(i[0]))
		wt.append(float(i[1]))
		u.append(float(i[2]))
		g.append(i[3])
	X = np.asarray([ht, wt, u]).T
	Y = np.asarray(g)
	r= 0
	srt = np.zeros(X.shape[1])
	for j in range(30):
		for i in range(len(X)):
			#calculating probability by transforming it to a sigmoid function
			arv = (1/(1+(mt.exp(-np.dot(srt, X[i])))))
			anum = 1 if Y[i] == 'wt' else 0
			if arv >= 0.5:
				if anum == 1:
					r += 1
			if arv<=0.5:
				if anum==0:
					r += 1
			
			t1=(anum-arv)
			srt += al * t1 * X[i]
	
	h1,w1,a1,c1=[],[],[],[]
	for i in range(len(ht)):
		#turning the data to numbers and keeping the thrashhold value at 0.5
		res=(1/(1+(mt.exp(-np.dot(srt,[ht[i],wt[i],u[i]])))))
		#If the value is greater than 0.5, make it male and set the value to 1.
		h1.append(ht[i])
		w1.append(wt[i])
		a1.append(u[i])
		if res>=0.5:
			c1.append('M')
		#If the value is less than 0.5, it is male and its value is 0.
		else:
			c1.append('F')
	plt.figure(figsize=(5,9), dpi=100)
	xes = plt.axes(projection ='3d')
	for j in range(len(ht)):
		if c1[j]=='M':
			xes.plot(ht[j],wt[j],u[j],'.b')
		else:
			xes.plot(ht[j],wt[j],u[j],'.g')
	plt.show()
	print('END Q3_AB\n')

if __name__ == "__main__":
    main()
