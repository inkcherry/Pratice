import numpy as np

a1 = np.array([[1,2,3,4],[2,2,2,3]],dtype=np.float32)  
print(a1)  
b=a1.reshape(1,8)
print(b)  
c=a1.reshape(8,1)
print(c) 

d=a1.reshape(4,2)
print(d) 


e=a1.reshape(1,-1)
print(e)

f=a1.reshape(2,-1)
print(f)

print("数据类型",type(a1))           #打印数组数据类型  
print("数组元素数据类型：",a1.dtype) #打印数组元素数据类型  
print("数组元素总数：",a1.size)      #打印数组尺寸，即数组元素总数  
print("数组形状：",a1.shape)         #打印数组形状  
print("数组的维度数目",a1.ndim)      #打印数组的维度数目 
