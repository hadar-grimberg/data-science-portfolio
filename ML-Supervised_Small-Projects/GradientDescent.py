
import numpy as np

X=np.array([[31,22],[22,21],[40,37],[26,25]])
y=np.array([2,3,8,12])


def normal_euqations(X,y):
    return np.linalg.inv(X.T@X)@X.T@y


def check_normal_equations(X,y):
    teta=normal_euqations(X,y)
    print('Regression Coefficients: ',teta)
    print('predicted y:', X@teta)
    print('loss :',((X@teta-y)**2).mean())

# 3D plane crossing (0,0,0)
check_normal_equations(X,y)

# 3rd feature: constant
X1=np.ones((4,3))
X1[:,:-1]=X
check_normal_equations(X1,y)

# 3rd feature: x1-x2
X2=X1.copy()
X2[:,2]=X[:,0]-X[:,1]
check_normal_equations(X2,y)

# 3rd feature: (x1-x2)^2
X3=X1.copy()
X3[:,2]=(X[:,0]-X[:,1])**2
check_normal_equations(X3,y)

# 3rd feature: (x1-x2)^2, 4th feature constant
X4=np.ones((4,4))
X4[:,:-1]=X3
check_normal_equations(X4,y)


###########################
#### Gradient Descent #####
###########################

x=np.array([0,1,2],dtype=np.float32)
y=np.array([1,3,7],dtype=np.float32)
X=np.c_[np.ones_like(x),x,x**2]

# regression works nicely, but this is not part of the task
check_normal_equations(X,y)

start = np.array([2,2,0],dtype=np.float32)

def prediction(X,tetas):
    return X@tetas

def mse_loss(y_predicted,y_real):
    return 0.5*((y_predicted-y_real)**2).mean()

def mse_loss_grad(y_predicted,y_real,X):
    return (y_predicted-y_real)@X/len(X)
  
    
def run_gradient_descent(X,y,start,learning_rate,epochs):
    t = start.copy()
    for epoch in range(epochs):
        y_predicted = X[:,0]*t[0] + X[:,1]*t[1] + X[:,2]*t[2]
        loss = mse_loss(y_predicted,y)
        grad_t0 = (y_predicted-y)*1
        grad_t1 = (y_predicted-y)*x
        grad_t2 = (y_predicted-y)*(x**2)
        t[0] -= learning_rate*grad_t0.mean()
        t[1] -= learning_rate*grad_t1.mean()
        t[2] -= learning_rate*grad_t2.mean()
    return t,loss
        
t,loss = run_gradient_descent(X,y,start,0.1,1000)    

def run_gradient_descent_matrices(X,y,start,learning_rate,epochs):
    t=start.copy()  
    for epoch in range(epochs):
        y_predicted=prediction(X,t)
        loss = mse_loss(y_predicted,y)
        grad = mse_loss_grad(y_predicted,y,X)
        t -= learning_rate*grad
        if epoch%100==0:
            print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
    return t

t = run_gradient_descent_matrices(X,y,start,0.1,1000)


def run_momentum_gradient_descent(X,y,start,learning_rate,momentum_decay,epochs):
    t=start.copy()
    v=np.zeros_like(start)
    for epoch in range(epochs):
        y_predicted = prediction(X,t)
        grad = mse_loss_grad(y_predicted,y,X)
        v = momentum_decay*v - learning_rate*grad
        t += v
        if epoch%100==0:
            loss = mse_loss(y_predicted,y)
            print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
    return t

t = run_momentum_gradient_descent(X,y,start,0.01,0.9,1000)


def run_nesterov_momentum_gradient_descent(X,y,start,rate,momentum_decay,epochs):
    t=start.copy()
    v=np.zeros_like(start)
    for epoch in range(epochs):
        y_predicted_nestrov = prediction(X,t+momentum_decay*v)
        grad=mse_loss_grad(y_predicted_nestrov,y,X)
        v=momentum_decay*v - rate*grad
        t+=v
        if epoch%100==0:
            y_predicted = prediction(X,t)
            loss=mse_loss(y_predicted,y)
            print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
    return t   
        
t = run_nesterov_momentum_gradient_descent(X,y,start,0.01,0.9,1000)



