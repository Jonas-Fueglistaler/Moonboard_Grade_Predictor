#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dnn_app_utils_predictor import *
import numpy as np
from sklearn import preprocessing #feature normalizing
import sys
sys.version

#### Define interactive surface that stores hold location on click in list - ready for prediction

#### Define interactive surface that stores hold location on click in list - ready for prediction

def interactive_selection():
    from ipywidgets import Button, HBox, VBox, widgets, Layout
    from IPython.display import display
    
    ## Generate interactive grid
    d={}
    output = widgets.Output
    #output.clear_output
    global hold_selection
    hold_selection = []
    
    
    #generate hold_key_matrix
    hold_key_matrix = generate_hold_key_matrix()
    
    def on_button_clicked(b):
        b.style.button_color = 'lightgreen'
        global hold_selection
        hold_selection.append(b.description)
        #print(hold_selection)
                    
                        
    for R in range(hold_key_matrix.shape[0]):    
        hold = hold_key_matrix[R,:].tolist() #convert to list, start with "18"
        item = [Button(description=h, layout=Layout(width='45px', height='60%')) for h in hold] #list of buttons
        d['{}{}'.format('H_box',R)] = HBox(item) #store all Hboxes in dictionary
        #define buttons
        for C in range(hold_key_matrix.shape[1]):
            button = item[C] #
            button.on_click(on_button_clicked)
    
    whole_grid = VBox([d['H_box0'], d['H_box1'], d['H_box2'], d['H_box3'], d['H_box4'], d['H_box5'], d['H_box6'], d['H_box7'], d['H_box8'], d['H_box9'], d['H_box10'], d['H_box11'], d['H_box12'], d['H_box13'], d['H_box14'], d['H_box15'], d['H_box16'], d['H_box17']])
    display(whole_grid)
    
    ## generate Termination button
    
    def end_button_clicked(b):
        predict_grade_JF(hold_selection)
    
    end_button = widgets.Button(description='Predict Grade!', button_style='danger')
    end_button.on_click(end_button_clicked)
    display(end_button)
    
    
    
##### generate hold location 'A1' to location (1,1) hash matrix

## generate holds_key matrix
from string import ascii_uppercase
# initialize grid
holds_grid = np.zeros((18,11), dtype=object) #dtype=object for strings with arbitrary length

counter = 0
for L in ascii_uppercase[0:11]:
    counter = counter + 1
    for N in range(1,19):
        holdsID_temp = '{}{}'.format(L,N)
        holds_grid[N-1, counter-1] = holdsID_temp
        
hold_key_matrix = np.flipud(holds_grid) #flip matrix vertically


## DEFINE FUNCTION TO GENERATE HOLD_KEY_MATRIX
def generate_hold_key_matrix():
    import numpy as np

    ## generate holds_key matrix
    from string import ascii_uppercase

    # initialize grid
    holds_grid = np.zeros((18,11), dtype=object) #dtype=object for strings with arbitrary length

    counter = 0
    for L in ascii_uppercase[0:11]:
        counter = counter + 1
        for N in range(1,19):
            holdsID_temp = '{}{}'.format(L,N)
            holds_grid[N-1, counter-1] = holdsID_temp
        
    hold_key_matrix = np.flipud(holds_grid) #flip matrix vertically
    return hold_key_matrix


## Generate grade dictionary

def generate_grade_dictionary():
    grade_dict = {0 : '5+', 1: '6A', 2: '6A+', 3: '6B', 4: '6B+', 5: '6C', 6: '6C+', 7: '7A', 8: '7A+', 9: '7B', 10: '7B+', 11: '7C', 12: '7C+', 13: '8A', 14: '8A+', 15: '8B', 16: '8B+'}
    return grade_dict

def predict_grade_JF(input_holds):
    assert len(input_holds) > 2, print("COME ON, THIS IS TOO HARD AND YOU KNOW IT")
    
    #initialize grid
    problem = np.zeros((18,11))

    # fill with holds
    for i in range(len(input_holds)):
        hold = input_holds[i]
        idx = np.where(hold_key_matrix == hold)
        problem[idx] = 1
    
    #print(problem)
    #flatten to 1D-array
    problem_flat = problem.reshape(198, 1)

    #feature scaling
    problem_scaled = preprocessing.scale(problem_flat)
    
    #load grade dict
    grade_dict = generate_grade_dictionary()
    
    ## LOAD LEARNED PARAMETERS FROM L-LAYER NEURAL NETWORK
    trained_parameters = np.load("trained_parameters.npy", allow_pickle=True)[()]

    #predict and output
    floats, dummy = L_model_forward_regressor(problem_scaled, trained_parameters)
    prediction = int(np.round(floats))
    grade = grade_dict[prediction]
    print("YOUR BOULDER HAS THE GRADE: \t" + str(grade))

