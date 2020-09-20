import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
pr = PolynomialFeatures(degree=2, include_bias=False)

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
url = "imports-85.data"		#temp DNS error on URL
df = pd.read_csv(url, header = None)

headers = ['symboling','normalised-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
df.columns = headers


#x = df.drop(['price'],axis=1)
x = df['engine-size','num-of-doors']
y = df['price']

steps = [('scaler',StandardScaler()), ('svm',SVC())]
pipe = Pipeline(steps)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

lr = lm.fit(x,y)							#fit linear regression model
yhat = lm.predict(x)						#prediction using above model

scores = cross_val_score(lr, x, y, cv=3)
np.mean(scores)


#parameters = {'SVM__C':[0.001,0.1,10,100,10e5],'SVM__gamma':[0.1,0.01]}

#grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

#grid.fit(x_train, y_train)

#print 'score = %3.2f' %(grid.score(x_test,y_test))
#print grid.best_params_


"""
input = [('scale',StandardScaler()), ('polynomial',PolynomialFeatures(degree=2)), ('model',LinearRegression())]
pipe=Pipeline(input)

pipe.train(X['horsepower','curb-weight','engine-size','highway-mpg'],y)
yhat = pipe.predict(X[['horsepower','curb-weight','engine-size','highway-mpg']])




def numeric(header):		#converts column 'header' to numeric values (to weed out blanks)
	df[header] = pd.to_numeric(df[header], errors = 'coerce')
	return(df[header])
def normalise(header):		#normalise columns
	df[header] = (df[header] - df[header].mean())/df[header].std()
	df[header] = df[header].fillna(df[header].mean())
def dummy(header):		#convert categorical variables to dummy variables (0s or 1s)
	pd.get_dummies(df[header])
def plot_corr_matrix(data,attr,fig_no):	#prep for heatmapping correlation matrix
	correlations = data_basic.corr()
	fig = plt.figure(fig_no)
	ax = fig.add_subplot(111)
	ax.set_title('Correlation Matrix')
	ax.set_xticklabels(['']+attr)
	ax.set_yticklabels(['']+attr)
	cax = ax.matshow(correlations,vmax=1,vmin=-1)
	fig.colorbar(cax)
	plt.show()

numeric('price')
df.dropna(subset=['price'], axis=0, inplace = True) #axis=0 drops row, 1 drops column. Inplace rewrites into df as if using df = this

numeric('normalised-losses')
mean_normloss = df['normalised-losses'].mean()
df['normalised-losses'].fillna(mean_normloss, inplace=True)

x4 = df['engine-size']			#map to variable before normalising for regression plots

numeric('num-of-doors')
numeric('num-of-cylinders')
numeric('bore')
numeric('stroke')
numeric('horsepower')
numeric('peak-rpm')
normalise('symboling')				#alternatively
normalise('normalised-losses')			#use StandardScaler from sklearn.preprocessing
normalise('num-of-doors')			#
normalise('length')				#StandardScaler().fit(x_data[['a','b',...]])
normalise('width')				#x_scale=StandardScaler().transform(x_data[[..
normalise('height')
normalise('curb-weight')
normalise('num-of-cylinders')
normalise('engine-size')
normalise('bore')
normalise('stroke')
normalise('compression-ratio')
normalise('horsepower')
normalise('peak-rpm')
normalise('city-mpg')
normalise('highway-mpg')
dummy('make')
dummy('fuel-type')
dummy('aspiration')
dummy('body-style')
dummy('drive-wheels')
dummy('engine-location')
dummy('engine-type')
dummy('fuel-system')

bins_price = np.linspace(min(df['price']),max(df['price']),4)	#separate price into 3 bins, equally spaced with 4 lines |x|x|x|
bin_names_price = ['Low','Medium','High']			#name each bin group
df['price-binned'] = pd.cut(df['price'],bins_price,labels=bin_names_price,include_lowest=True)								#cut price into the bins

drive_wheels_counts = df['drive-wheels'].value_counts()					#count number in each category
drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'}, inplace = True)
drive_wheels_counts.index.name = 'drive-wheels'

df_test1 = df[['drive-wheels','body-style','price']]
df_grp1 = df_test1.groupby(['drive-wheels','body-style'], as_index=False).mean()	#create table split between groups, showing average values
df_pivot1 = df_grp1.pivot(index = 'drive-wheels', columns = 'body-style')		#pivot to have one variable in columns

##x4 = df['engine-size']			#mapped to variable prior to being normalised
y4 = df['price']

pearson_coef2, p_value2 = stats.pearsonr(df['horsepower'], y4)	#Pearson correlation test


f1 = plt.figure(1)			#need to use figures else show() only shows one plot
plt.boxplot(df['price'])
f2 = plt.figure(2)
plt.scatter(df['price'],df['engine-size'])
f3 = plt.figure(3)			#heatmap
plt.pcolor(df_pivot1, cmap='RdBu')
plt.yticks(np.arange(0.5, len(df_pivot1.index), 1), df_pivot1.index)
plt.xticks(np.arange(0.5, len(df_pivot1.columns), 1), df_pivot1.columns)
plt.colorbar()
f4 = plt.figure(4)
plt.plot(x4,y4,'yo',x4,np.poly1d(np.polyfit(x4,y4,1))(x4),'--k')	#scatter with regression line
plt.show()				#opens windows to show plots in

data_basic = df.loc[:,headers]		#
plot_corr_matrix(data_basic,headers,3)	#correlation heatmap

f5 = plt.figure(5)
sns.regplot(x = 'highway-mpg', y = 'price', data = df)
plt.ylim(0,)
f6 = plt.figure(6)
sns.residplot(df['highway-mpg'], df['price'])		#if residuals are scattered randomly, error is due to randomness, otherwise linear regression is not accurate
plt.show()


df_anova1 = df[['make','price']]
grouped_anova1 = df_anova1.groupby(['make'])
anova_results1 = stats.f_oneway(grouped_anova1.get_group('honda')['price'], grouped_anova1.get_group('jaguar')['price'])



f7 = plt.figure(7)							#plot prediction vs actual values
ax7 = sns.distplot(df['price'], hist=False, color='r', label='Actual value')	#
sns.distplot(ylmhat, hist=False, color='b', label='Fitted values', ax=ax7)	#
plt.show()


f8 = np.polyfit(x4, y4, 3)
p8 = np.poly1d(f8)
x_poly = pr.fit_trasnform(x[['horsepower','curb-weight'])

print (df[['price','price-binned']].head(15))
#print ('Coefficients: \n', ylmrm.coef_)		# \n gives new line within 'text'
#print ('Intercept: \n', ylmrm.intercept_)
print(p8)

df.to_csv('auto.csv') #in ChromeOS, no C:\ drive to export to, so using default directory
"""



#print (df.dtypes)		#shows type, e.g. int64
#print (df['normalised-losses'].describe(include='all')) #provides descriptive stats, remove include='all' for numeric stats only



