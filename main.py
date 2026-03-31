import numpy as np
import csv
import matplotlib.pyplot as plt

# Importing The dataset
rows=[]

f= open("data/housing.csv","r")
reader=csv.DictReader(f)
for row in reader:
    try:
        income=float(row['median_income'])
        price=float(row['median_house_value'])
        rows.append((income,price))
    except:
        pass
f.close()

data=np.array(rows,dtype=float)
X= data[:,0] 
Y=data[:,1]


#Train and Test Split
np.random.seed(42)
indices=np.random.permutation(len(X))
split = int(0.8*len(X))
X_Train = X[indices[:split]]
Y_Train = Y[indices[:split]]
X_Test = X[indices[split:]]
Y_Test = Y[indices[split:]]

#Standardization
X_mean = np.mean(X_Train)
X_std = np.std(X_Train)
Y_mean = np.mean(Y_Train)
Y_std = np.std(Y_Train)

x_Train_norm = (X_Train - X_mean)/X_std
x_Test_norm = (X_Test - X_mean)/X_std
Y_Train_norm = (Y_Train - X_mean)/X_std
Y_Test_norm = (Y_Test - X_mean)/X_std


#Initialize The Hyperparameter
weight=0
bias=0
learning_rate=0.01
epochs=1000

#Model Functions
def pred(X,w,b):
    return w*X+b

def loss(Y_pred,Y_true):

    n=len(Y_true)
    error=Y_pred-Y_true
    return (1/n)*np.sum(error**2)

def gradient(X,Y_true,Y_pred):
    n=len(Y_true)
    error=Y_pred-Y_true
    m_grad=(1/n)*np.dot(error,X)
    b_grad=(1/n)*np.sum(error)
    return m_grad,b_grad

def r_squared(Y_true,Y_pred):
    ss_res=np.sum((Y_true-Y_pred)**2)
    ss_tot=np.sum((Y_true-np.mean(Y_true))**2)
    return 1-(ss_res/ss_tot)

losses=[]
R2=[]


#Training process
for epoch in range(epochs):
    y_pred= pred(x_Train_norm,weight,bias)
    lose=loss(y_pred,Y_Train_norm)
    dm,db=gradient(x_Train_norm,Y_Train_norm,y_pred)
    r2=r_squared(Y_Train_norm,y_pred)

    losses.append(lose)
    R2.append(r2)

    #Updation
    weight=weight-(learning_rate*dm)
    bias=bias-(learning_rate*db)

#Testing Process
y_test=pred(x_Test_norm,weight,bias)
t_lose=loss(y_test,x_Test_norm)
r2=r_squared(Y_Test_norm,y_test)

#Denormalization
m_real=weight*(Y_std/X_std)
b_real=bias*Y_std+Y_mean-m_real*X_mean


#Deliverables
print(f"Learned Eqaution y={m_real:.2f}*X_train_norm+{b_real:.2f}")
print(f"Train R²={R2[-1]:.4f}  Test R²={r2:.4f}")

plt.figure(figsize=(8,5))
plt.plot(losses, color='blue', linewidth=2)
plt.title("Loss vs Epoch (Convergence Graph)")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (Loss)")
plt.grid(True, alpha=0.3)
plt.show()