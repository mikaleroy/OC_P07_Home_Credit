# IMPORTS
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

import numpy as np
import pandas as pd 

init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()

# loading message
print('Loading functions ...')


######### Functions for model experiment notebooks

# Plot graphical confusion matrix
def plot_confusion_matrix(y, y_pred, axe, suptitle=''):
    size = 16
    # confusion matrix with counts
    c_m = confusion_matrix(y,y_pred)
    # confusion matrix with percentage from true classes
    c_m_norm = confusion_matrix(y,y_pred,normalize='true')
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_count = c_m.flatten()
    group_perc  = c_m_norm.flatten()
    # compose labels
    labels = ['{}\n{}\n{:.2%}'.format(v1, v2, v3) for v1, v2, v3 in zip(group_names, group_count, group_perc)]
    labels = np.asarray(labels).reshape(2,2)
    # plot pimped maxtrix
    sns.heatmap(c_m,
                annot=labels,
                annot_kws={"size": 20},
                fmt='',
                cmap = 'Blues', #plt.cm.RdYlBu_r,
                cbar=False,
                ax=axe
               )
    if suptitle != None :
        axe.set_title(suptitle)
    axe.set_xlabel('Predicted',fontsize=size) ; axe.set_ylabel('Original',fontsize=size)
    axe.set_yticklabels(['Repayed','Default'],rotation = 90,va="center",fontsize=size-3)
    axe.set_xticklabels(['Repayed','Default'], fontsize=size-3)

# Cross validated score    
def cv_score(estimator,X,y,scorer,cv):
    score = cross_validate(estimator,
                           X,
                           y,
                           scoring=scorer,
                           n_jobs=4,
                           cv=cv)
    return {'COST' :score['test_COST'].mean(), 'AUC' : score['test_AUC'].mean()}
    
# Evaluate a fitted estiator on  X_train, y_train, X_test, y_test
from neptune.new.types import File
from sklearn.metrics import roc_auc_score
def evaluate_estimator(estimator, X_train, y_train, X_test, y_test, scorer, test_scorer,cv=3, log=(False,'','')):
    ''' Evaluate a fitted estimator '''
    
    # Predictions
    train_predict = estimator.predict(X_train)
    test_predict = estimator.predict(X_test)
        
    # Setup plot grid
    fig,axe = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle('Predict Labels',fontsize=18)
    
    # Confusion matrix on train
    plot_confusion_matrix(y_train, train_predict, axe[0], suptitle='Entrainnement')

    # Confusion matrix on test
    plot_confusion_matrix(y_test,test_predict, axe[1], suptitle='Validation')

    # Classification report on train
    print('TRAIN\n',classification_report(y_train,train_predict))
    # Cross validate score on train
    train_scores = cv_score(estimator, X_train, y_train,scorer,cv)
    print(' Scores  on train : {}'.format(train_scores))
    
    # Classification report on test
    print('TEST\n',classification_report(y_test,test_predict))
    # Cross validate score on test
    test_scores1 = cv_score(estimator,X_test, y_test,scorer,cv)
    test_scores = {'COST' : test_scorer(y_test, test_predict),
#                    'AUC'  : roc_auc_score(y_test, test_predict)
                  }
    print(' Scores  on test : {}'.format(test_scores))
       
    if log[0]:
        log[1][log[2]+ 'Train scores'].log(train_scores)
        log[1][log[2]+ 'Test scores'].log(test_scores)
        log[1][log[2]+ 'Confusion matrix'].log(File.as_image(fig))
#         return fig

# probabilities by target distribution plot
def proba_distributions(y,predicted_probas,log=(False,'','')):
    size = 16
    fig, ax = plt.subplots(1,1)
    ax.set_title('Distributions vs Classes',fontsize=size)
    ax.set_xlabel('Predicted probabilities',fontsize=size-3)             
    sns.histplot(predicted_probas[y ==1][:,1],bins=100, stat='density', kde=True, alpha=.25,color='red',ax=ax)
    sns.histplot(predicted_probas[y ==0][:,1],bins=1000, stat='density', kde=True, alpha=.25,color='blue',ax=ax) 
    if log[0]:
        log[1][log[2]+' probas distributions'].log(File.as_image(fig))



# Feature importance plot
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()


def feature_importance_plot(model_coeffs, X, log=(False,'','')):
    feature_imp = pd.DataFrame(sorted(zip(model_coeffs, X.columns)),
                               columns=['Value','Feature'])
    features_df = feature_imp.sort_values(by="Value", ascending=False)
#     display(features_df)
    
    # Feature importance Plot
    data1 = features_df.head(20)
    data = [go.Bar(x=data1.sort_values(by='Value')['Value'] ,
                   y = data1.sort_values(by='Value')['Feature'],
                   orientation = 'h',
                   marker = dict(color = 'rgba(43, 13, 150, 0.6)',
                                 line = dict(color = 'rgba(43, 13, 150, 1.0)',
                                             width = 1.5
                                            )
                                )
                  )
           ]
    layout = go.Layout(autosize=False,
                       width=1300,
                       height=700,
                       title = "Top 20 important features",
                       xaxis=dict(title='Importance value'),
                       yaxis=dict(automargin=True),
                       bargap=0.4
                      )
    fig = go.Figure(data = data, layout=layout)
    fig.layout.template = 'seaborn'
    py.iplot(fig)
    if log[0]:
        fig.write_image('fi.png')
        log[1][log[2]+' Feature importances'].log(File('fi.png'))



# function to optimize decision threshold
# for each threshold try, looks on predict changes
# apply penalties and compute penalty score
# tries til find best threshold 
from scipy.optimize import minimize_scalar

def optimize_threshold(estimator,
                       X,
                       y,
                       tp_cost: int = 0,
                       tn_cost: int = 0,
                       fp_cost: int = 0,
                       fn_cost: int = 0,
                       model_name: str = ''
                      ):
    
    
    
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    # cost = []
    # seuil = []
    
    # internal loss function
    def calculate_loss(x,*args):
        actual, pred_probas = args    
        predicted = (pred_probas > x).astype('int')
        
        '''internal function to calculate loss '''
        # true positives (1 to 1)
        tp = predicted + actual
        tp = np.where(tp == 2, 1, 0)
        tp = tp.sum()
        
        # true negative (0 to 0)
        tn = predicted + actual
        tn = np.where(tn == 0, 1, 0)
        tn = tn.sum()
        
        # false positive ( 1 to 0)
        fp = (predicted < actual).astype(int)
        fp = np.where(fp == 1, 1, 0)
        fp = fp.sum()
        
        # false negative (0 to 1)
        fn = (predicted > actual).astype(int)
        fn = np.where(fn == 1, 1, 0)
        fn = fn.sum()
        
            
        total_cost = (tp_cost * tp) + (tn_cost * tn) + (fp_cost * fp) + (fn_cost * fn)
        
        # cost.append(total_cost)
        # seuil.append(x)
        return - total_cost
    
    # minimise la fonction score suivant le seuil de décision
    res = minimize_scalar(calculate_loss,args=(y,y_pred_proba), method='golden')
    
    print(res.x)  

    # loop on a small grid to plot cost results
    grid = np.arange(0, 1, 0.01)
    cost = []
    
    for i in grid:
        pred_prob = (y_pred_proba >= i).astype(int)
        cost.append(-calculate_loss(i,*(y,y_pred_proba) ))
     
    optimize_results = pd.DataFrame({'Probability Threshold': grid,
                                         'Cost Function': cost
                                        }
                                       )
    
    # positive_sorted = optimize_results[optimize_results['Probability Threshold'] >= 0].sort_values(by='Probability Threshold')
    positive_sorted = optimize_results
    
    fig = px.line( positive_sorted,
                   x='Probability Threshold',
                   y='Cost Function',
                   line_shape='linear',
                   range_x=(0,1)
                 )
    fig.update_layout(plot_bgcolor='rgb(245,245,245)')
    title = f'{model_name} Probability Threshold Optimization'
    
    # calculate vertical line
    y0 = positive_sorted['Cost Function'].min()
    y1 = positive_sorted['Cost Function'].max()
    x0 = positive_sorted.sort_values(by='Cost Function',
                                      ascending=False
                                     ).iloc[0][0]
    x1 = x0
    t = x0
    fig.add_shape(dict(type='line',
                               x0=x0,
                               y0=y0,
                               x1=x1,
                               y1=y1,
                               line=dict(color='red', width=2)
                              )
                          )
    fig.update_layout( title={'text': title,
                                      'y': 0.95,
                                      'x': 0.45,
                                      'xanchor': 'center',
                                      'yanchor': 'top' 
                                     } 
                              )
    fig.show()
    print(f'Optimized Probability Threshold: {res.x} | Optimized Cost Function: {y1}')   
    
    # return float(t)
    return res.x

# model evaluation with treshold 
def cost_predictions( estimator,X,y,level):
    ''' Evaluate a fitted estimator '''
    # original predicted labels
    labels = estimator.predict(X)
    
    # original predicted proba
    predicted_probas = estimator.predict_proba(X)[:,1]

    # decision
    predicted_labels = (predicted_probas > level).astype('int')
    

    # Evaluate
    # evaluate
    fig,axe=plt.subplots(1, 2, figsize=(13,5))
    fig.suptitle('Avec fonction de coût', fontsize=18)

    # confusion matrix on test with decision function 0.5
    plot_confusion_matrix(y, labels, axe[0], suptitle='Test Predict')

    print('Test predict par défaut (0.5)\n', classification_report(y, labels))

    # confusion matrix on test with function set to threshold
    plot_confusion_matrix(y, predicted_labels, axe[1], suptitle='Test Predict Proba seuil décision {}'.format(level))

    print('Test predict_proba seuil {}\n'.format(level),classification_report(y, predicted_labels))


# Class to add a step to pipeline with decision threshold    
import numpy as np
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    FLOAT_DTYPES,
)
from sklearn.exceptions import NotFittedError

from sklego.base import ProbabilisticClassifier



class Thresholder(BaseEstimator, ClassifierMixin):
    """
    Takes a two class estimator and moves the threshold. This way you might
    design the algorithm to only accept a certain class if the probability
    for it is larger than, say, 90% instead of 50%.

    :param model: the moddel to threshold
    :param threshold: the actual threshold to use
    :param refit: if True, we will always retrain the model even if it is already fitted.
    If False we only refit if the original model isn't fitted.
    """

    def __init__(self, model, threshold: float, refit=False):
        self.model = model
        self.threshold = threshold
        self.refit = refit
        

    def _handle_refit(self, X, y):
        """Only refit when we need to, unless refit=True is present."""
        if self.refit:
            self.estimator_.fit(X, y)
        else:
            try:
                _ = self.estimator_.predict(X[:1])
            except NotFittedError:
                self.estimator_.fit(X, y)


    def fit(self, X, y):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
#         X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.estimator_ = clone(self.model)
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError(
                "The Thresholder meta model only works on classifcation models with .predict_proba."
            )
        self._handle_refit(X, y)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "The Thresholder meta model only works on models with two classes."
            )
        return self



    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
#         check_is_fitted(self, ["classes_", "estimator_"])
#         predicate = self.estimator_.predict_proba(X)[:, 1] >= self.threshold
#         return np.where(predicate, self.classes_[1], self.classes_[0])
        return (self.model.predict_proba(X)[:, 1] > self.threshold).astype('int')



    def predict_proba(self, X):
#         check_is_fitted(self, ["classes_", "estimator_"])
        return self.model.predict_proba(X)



    def score(self, X, y):
        return self.estimator_.score(X, y)
    
    
    
####### EDA FUNCTIONS ################


# Summary with correlation to TARGET and missing values
def numerical_summary(table,tri=True,sens=True):
        # Total Nan
        mis_val = table.isnull().sum()
        
        # Pourcentage de  Nan
        mis_val_percent = table.isnull().sum() / len(table) * 100
        
        # Correlations with TARGET
        correlation = table.corr()['TARGET']
        
        # Data Frame contenant les résultats
        summary_table = pd.concat([correlation,mis_val, mis_val_percent], axis=1)
        
        # Nommage des colonnes
        summary_table = summary_table.rename(columns = {'TARGET' : 'correlations',
                                                        0 : 'valeurs_manquantes',
                                                        1 : '%_total'
                                                       }
                                            )
        
        if tri:
            # Avec tri sur les correlations
            summary_table = summary_table.sort_values('correlations', ascending=sens).round(2)
        else:
            # Sans tri sur les correlations
            summary_table = summary_table.sort_values('correlations', ascending=False).round(2)
        
        # Affichage du nom de la table analysée
        # du nombre de colonnes concernées par des valeurs manquantes
        print (" Data Frame a " + str(table.shape[1]) + " colonnes.\n"      
            "Dont " + str(summary_table.shape[0]) +
              " colonnes contiennent des valeurs manquantes.")
        
        # Retourne la tables contenant les stats par colonnes
        return summary_table
    
    
# Function calculant le score, la pvalue ANOVA et le nombre de valeurs manquantes (Nan) des vars
from sklearn.feature_selection import SelectKBest, f_classif

def categorical_summary(table,target,tri=True,sens=True):
        # Total Nan
        mis_val = table.isnull().sum()
        
        # Pourcentage de  Nan
        mis_val_percent = table.isnull().sum() / len(table) * 100
        
        # Anova with TARGET
        select = SelectKBest(f_classif, k='all').fit(table, target)
        anova = pd.DataFrame([select.scores_,select.pvalues_],
                             columns=table.columns,
                             index=['f_value','p_value']
                            ).T
        
        # Data Frame contenant les résultats
        summary_table = pd.concat([anova, mis_val_percent], axis=1)
        
        # Nommage des colonnes
        summary_table = summary_table.rename(columns = {0 : 'valeurs_manquantes',
                                                        1 : '%_total'
                                                       }
                                            )
        
        if tri:
            # Avec tri sur les correlations
            summary_table = summary_table.sort_values('f_value', ascending=sens).round(2)
        else:
            # Sans tri sur les correlations
            summary_table = summary_table.sort_values('f_value', ascending=False).round(2)
        
        # Affichage du nom de la table analysée
        # du nombre de colonnes concernées par des valeurs manquantes
        print (" Data Frame a " + str(table.shape[1]) + " colonnes."      
            "\n Dont " + str(summary_table.shape[0]) +
              " colonnes contiennent des valeurs manquantes.")
        
        # Retourne la tables contenant les stats par colonnes
        return summary_table
    
# OneHot encode object columns from a dataframe
# encoded df, new_added_columns, df_columns
def one_hot_encoding_dataframe(df):
    '''
    one hot encoding 
    '''
    original_columns = list(df.columns)
    cat_columns=[x for x in df.columns if df[x].dtype == 'object']
    df=pd.get_dummies(df,columns=cat_columns,dummy_na= False)
    new_added_columns=list(set(df.columns).difference(set(original_columns)))
    return df, new_added_columns, df.columns    


# Select numerical features of df (assign df.name before)
# retain those correlated above 'correlation'% with TARGET feature
# discard those with more than 'missing' missing values
def select_numerical(df, correlation=0.03, missing=20):    
    # Get numerical features in df
    summary = numerical_summary(df.select_dtypes('number')) 
    # Get correlation to TARGET
    # retain var with correlation above 3%
    most_corr = summary.loc[abs(summary.correlations) > correlation]
    display('>>>> Most correlated with TARGET from '+str(df.name), most_corr)
    # drop var above 20% of missing values
    most_corr_less_miss = most_corr.loc[most_corr['%_total'] < missing]

    return df[most_corr_less_miss.index]





    
    
# Sort variables by autocorrélation coefficient  
def correlated_features(df , thres : float, kind='pearson'):
    corr_vars = []
    corr_mat = df.corr()
    target = df.columns.get_loc('TARGET')
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if corr_mat.iloc[i,j] > thres :
                corr_vars.append({'feat_1':corr_mat.columns[i],
                               'feat_1 corr to target': corr_mat.iloc[i,target],
                               'feat_2':corr_mat.columns[j],
                               'feat_2 corr to target': corr_mat.iloc[i,target],
                               'feat_1 vs feat_2': corr_mat.iloc[i,j]})
                
    return pd.DataFrame(corr_vars)




# Plot var dist by target value
# 0 in azure
# 1 in seashell
def frame_vs_target(frame,target,title=''):
    num = frame.shape[1]
    colonnes = 3
    lignes = num//colonnes+1
    plt.figure(figsize = (16, 2.5*lignes))
    plt.suptitle(title,fontsize=18, y=1.01)
    # iterate through the new features
    df = frame.join(target)
    for i, col in enumerate(df.columns.drop('TARGET')):
        # create a new subplot for each source
        plt.subplot(lignes, colonnes, i + 1)
        # plot repaid loans
        sns.kdeplot(df.loc[df['TARGET'] == 0, col],color='dodgerblue', label = 'target == 0')
        # plot loans that were not repaid
        sns.kdeplot(df.loc[df['TARGET'] == 1, col],color='firebrick', label = 'target == 1')

        # Label the plots
#             plt.title('Distribution of %s by Target Value' % col)
        plt.xlabel('%s' % col); plt.ylabel('Density');
    plt.gcf().subplots_adjust(top=0.8)
    plt.tight_layout(pad=1, h_pad = 0.5, w_pad=0.5)    
    
    

# Violin plots of all vars in data frame
def multi_violin(df,title=''):
    num = df.shape[1]
    columns = 3
    rows = num//columns+1
    fig=plt.figure(figsize=(4*columns,3*rows))
    plt.suptitle(title,fontsize=18, y=1.01)
    for i,col in enumerate(df.columns.drop('TARGET')):
        # create a new subplot for each source
        plt.subplot(rows, columns, i + 1)
        # plot violins
        sns.violinplot(data=df,
                       x=col,
                       y=["01"]*len(df),
                       saturation=0.50,
                       palette =['azure','seashell'],
                       hue='TARGET',
                       split=True
                      )

        # Label the plots
#             plt.title('Distribution of %s' % col)
        plt.xlabel(''); plt.title('%s' % col); plt.ylabel('Density')

    for ax in fig.axes[1:] : ax.get_legend().remove()

    plt.gcf().subplots_adjust(top=0.8)
    plt.tight_layout(pad=1, h_pad = 0.5, w_pad=0.5)

   


print('.... done.')
