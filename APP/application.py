


# ********** IMPORTS 
import numpy as np
import pandas as pd 

# graphic
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle

# files
import joblib 
# Shap explanations
import shap
# Streamlit
import streamlit as st
import streamlit.components.v1 as components
# Layout
st.set_page_config(page_title='Home Credit',
                   page_icon=None,
                   layout='wide',
                   initial_sidebar_state='auto',
#                    base='light'
                  )
st.set_option('deprecation.showPyplotGlobalUse', False)



# Chargement des données
@st.cache(allow_output_mutation=True)
def load_datas():
    return  joblib.load('APP/features'), joblib.load('df_client'), joblib.load('Neighbors'), joblib.load('model')


# Predictions probabilities
@st.cache
def predict_probabilities():
    return pd.DataFrame(model.predict_proba(feat)[:,1],
                        columns=['proba'],
                        index=feat.index
                       )

# Predicted classes
@st.cache
def predict_classes():
    p_c = (predicted_probas.proba > 0.5).astype(int)
    p_c.name = 'class'
    return p_c

# Function to explain by shap (does not handle votingclassifer natively)
@st.cache
def f(X):
    return model.predict_proba(X)[:,1]

# Gauge for accept/ reject descision display
def gauge(labels=['LOW','HIGH'], colors='jet_r', arrow=1, title=''): 
    # internal functions
    def degree_range(n): 
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang): 
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation
       
    # some sanity checks first
    N = len(labels)
    if arrow > 180: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, 180))
      
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    # begins the plotting   
    fig, ax = plt.subplots()
    ang_range, mid_points = degree_range(N)
    labels = labels[::-1]
    
    # plots the sectors and the arcs
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=1))
    [ax.add_patch(p) for p in patches]

    # set the labels (e.g. 'LOW','MEDIUM',...)
    for mid, lab in zip(mid_points, labels): 
        ax.text(0.34 * np.cos(np.radians(mid)), 0.34 * np.sin(np.radians(mid)),
                lab,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=30,
                rotation = rot_text(mid))

    # set the bottom banner and the title
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    ax.text(0, -0.09, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=90, fontweight='bold')

    # plots the arrow now
    pos = arrow
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.01, head_width=0.03, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    # removes frame and ticks, and makes axis equal and tight
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    

# Shap bar plot to explain client descision vs nearest neighbors    
def bar_voisins(id_row, num_neighbors=10, max_features=9):
    # queriing num_neighbors closest neighbors to sample id_row and display first max_features
    _, ind = tree.query(feat.iloc[id_row,:].values.reshape(1, -1), k=num_neighbors+1) 
    voisins=['client']
    autres = [ 'repayed' if x==0 else 'default' for x in predicted_classes.iloc[ind[0][1:]]]
    if set(autres) != set(['repayed','default']):
        st.write('Not enougth neighbors to compare >> select more.')
        return
    voisins.extend(autres)
    group = feat.iloc[ind[0],:]
    shap_values_voisins = explainer(group)
    shap.plots.bar(shap_values_voisins.cohorts(voisins).mean(0),
                   max_display=max_features,
                   show=False
                  )
    fig=plt.gcf()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    # get current axes
    ax=plt.gca()
    # change bar color according to repayed/default colors
    for i in range(0,max_features,1):
        ax.patches[i].set_facecolor(client_color)
        ax.patches[max_features + i].set_facecolor(no_color)
        ax.patches[2 * max_features + i].set_facecolor(yes_color)
    # set legend colors according to previous step
    ax.get_legend().legendHandles[0].set_color(client_color) 
    ax.get_legend().legendHandles[1].set_color(no_color) 
    ax.get_legend().legendHandles[2].set_color(yes_color) 


# Html container with js enabled to plot force_plot    
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Force plot for one row    
def explain_row(id_row):
    single = feat.iloc[id_row:id_row+1,:]
    shap_values = explainer(single)
    return shap.force_plot(base_value = shap_values.base_values [0],
                           shap_values = shap_values.values[0],
                           features = single,
                            plot_cmap=[no_color,yes_color]
                          )
# Display some client infos
def display_client(id_sk):
    client = df_client.loc[id_sk]
    st.write('Amount goods price    : {:} \u20B9'.format(client.AMT_GOODS_PRICE))
    st.write('Goods price / Loan   : {:.2%}'.format(client.GOODS_PRICE_CREDIT_PER))
    st.write('Percent of working days   : {:.2%}'.format(client.DAYS_WORKING_PER ))
    st.write('Total loans / working days   : {:.2%}'.format(client.AMT_CREDIT_DAYS_EMPLOYED_PERC ))
    st.write('Working since : {} months'.format(client.DAYS_EMPLOYED ))
#     st.write('\n')
    st.write('Work : {}'.format(client.ORGANIZATION_TYPE))
    st.write('Income type : {}'.format(client.NAME_INCOME_TYPE))
    st.write('Occupation : {}'.format(client.OCCUPATION_TYPE))
#     st.write('\n')
    st.write('Age : {} years'.format(client.DAYS_BIRTH))
    st.write('Family status : {}'.format(client.NAME_FAMILY_STATUS))
    st.write('Education type : {}'.format(client.NAME_EDUCATION_TYPE))
    st.write('Housetype mode : {}'.format(client.HOUSETYPE_MODE))
    st.write('Housing type : {}'.format(client.NAME_HOUSING_TYPE))
#     st.write('\n')
    st.write('EMERGENCYSTATE_MODE : {}'.format(client.EMERGENCYSTATE_MODE))
    






# Main 
if __name__ == "__main__":
    feat,df_client,tree,model = load_datas()
#     prep_feat = prep_features()
    predicted_probas = predict_probabilities()
    predicted_classes = predict_classes()

    client_color = '#7FFFD4'
    yes_color = '#008bfb'   #'#007A00'
    no_color =  '#ED1C24'
    threshold=.5
    
    # Create an explainer
    explainer = shap.Explainer(f,feat,link=shap.links.logit)
    
    # Dashboard
    st.title('Home Credit')
    # Upper band
    left_column,mid_column , right_column = st.beta_columns([3,2,9]) #[10,1,15]

    # Upper left frame
    with left_column:
        with st.beta_container():
            # select loan box
            st.subheader('Summary')
            id_sk = st.selectbox('Select loan application',
                                 df_client.index,
                                 index=14
                                )
            id_row = df_client.index.get_loc(id_sk)
            display_client(id_sk)
            
    # Upper mid frame
    with mid_column:
        with st.beta_container():
            st.subheader('Decision')
            st.pyplot(gauge(labels=['Granted', 'Rejected'] ,
                          colors=[yes_color, no_color],
                          arrow=180-predicted_probas.loc[id_sk].proba*100*1.8-(50-threshold*100)*1.8,
                          title='\n {:.2%}'.format(predicted_probas.loc[id_sk].proba)
                          )
                   )

    # Upper right frame        
    with right_column:
        with st.beta_container():    
            st.subheader('Explanations for application')
            k = st.selectbox('Neighbors number',
                             np.arange(4,20,2),
                             index=3,
                             key=None,
                             help='Choose number of neighbors'
                            )
            st.pyplot(bar_voisins(id_row,num_neighbors = k))

    # Lower band        
    with st.beta_container():
        shap.initjs()
        st.subheader('Force plot')
        st_shap(explain_row(id_row))
        
    # for debugging purpose
#     with st.beta_container():
#             st.subheader('Decision')
#             st.write('row:',id_row,
#                      'app:',id_sk,
#                      'proba:',predicted_probas.loc[id_sk].proba,
#                      'thres:',threshold
#                      )

