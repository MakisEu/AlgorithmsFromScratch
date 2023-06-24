from numpy import * 
import os

def compute_error(points,b,m):
  totalerror=0
  for i in range(len(points)):
    x=points[i,0]
    y=points[i,1]
    totalerror+=pow((y-(m*x+b)),2)
  totalerror=totalerror/float(len(points))
  return totalerror
def step_gradient(b_current,m_current,points,learning_rate):
  b_grad=0
  m_grad=0
  N=len(points)
  for i in range(N):
    x=points[i,0]
    y=points[i,1]
    #compute partial derivative with respect to b and m
    b_grad+=-(y-(m_current*x+b_current))
    m_grad+=-x*(y-(m_current*x+b_current))
  b_grad=2*b_grad/float(N)
  m_grad=2*m_grad/float(N)
  new_b = b_current-(learning_rate*b_grad)
  new_m = m_current-(learning_rate*m_grad)
  return [new_b,new_m]
def gradient_descent_runner(points,start_b=0,start_m=0,learning_rate=0.0001,iter_num=1000):
  b=start_b
  m=start_m
  for i in range(iter_num):
    b,m=step_gradient(b,m,array(points),learning_rate)
  return [b,m]
def run():
  #get data
  path=os.path.dirname(os.path.realpath(__file__))
  path=path.replace("/src",'/data')
  points=genfromtxt(path+"/"+'LR.csv',delimiter=",")
  #define HyperParams
  #learning_rate=0.0001
  #in_b=0
  #in_m=0
  #iter_num=1000
  #Train model
  [b,m]=gradient_descent_runner(points)
  print("{0}x+{1}=y,(error{2})".format(m,b,compute_error(points,b,m)))
if __name__=="__main__":
  run()
