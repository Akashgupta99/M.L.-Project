# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:03:51 2020

@author: AKASH GUPTA
"""

import pandas as pd
import skfuzzy as skf
import numpy as np
from skfuzzy import control as ctrl
import statistics

df = pd.read_csv('Product Requirement Analysis final.csv')
df.head()

review_cam = ctrl.Consequent(np.arange(0, 11, 1), 'reviewc')
review_cam['low'] = skf.trimf(review_cam.universe, [0, 0, 5])
review_cam['medium'] = skf.trimf(review_cam.universe, [0, 5, 10])
review_cam['high'] = skf.trimf(review_cam.universe, [5, 10, 10])
review_scr = ctrl.Consequent(np.arange(0, 11, 1), 'reviewsc')
review_scr['low'] = skf.trimf(review_scr.universe, [0, 0, 5])
review_scr['medium'] = skf.trimf(review_scr.universe, [0, 5, 10])
review_scr['high'] = skf.trimf(review_scr.universe, [5, 10, 10])
review_soft = ctrl.Consequent(np.arange(0, 11, 1), 'reviewso')
review_soft['low'] = skf.trimf(review_soft.universe, [0, 0, 5])
review_soft['medium'] = skf.trimf(review_soft.universe, [0, 5, 10])
review_soft['high'] = skf.trimf(review_soft.universe, [5, 10, 10])
review_bat = ctrl.Consequent(np.arange(0, 11, 1), 'reviewb')
review_bat['low'] = skf.trimf(review_bat.universe, [0, 0, 5])
review_bat['medium'] = skf.trimf(review_bat.universe, [0, 5, 10])
review_bat['high'] = skf.trimf(review_bat.universe, [5, 10, 10])

a=df['camera']
b=df['screen']
c=df['software']
d=df['battery']

camera = ctrl.Antecedent(np.arange(0, 11, 1), 'camera')
screen = ctrl.Antecedent(np.arange(0, 11, 1), 'screen')
software = ctrl.Antecedent(np.arange(0, 11, 1), 'software')
battery = ctrl.Antecedent(np.arange(0, 11, 1), 'battery')

camera['poor'] = skf.trimf(camera.universe, [0, 0, 5])
camera['average'] = skf.trimf(camera.universe, [0, 5, 10])
camera['good'] = skf.trimf(camera.universe, [5, 10, 10])
screen['poor'] = skf.trimf(screen.universe, [0, 0, 5])
screen['average'] = skf.trimf(screen.universe, [0, 5, 10])
screen['good'] = skf.trimf(screen.universe, [5, 10, 10])
software['poor'] = skf.trimf(software.universe, [0, 0, 5])
software['average'] = skf.trimf(software.universe, [0, 5, 10])
software['good'] = skf.trimf(software.universe, [5, 10, 10])
battery['poor'] = skf.trimf(battery.universe, [0, 0, 5])
battery['average'] = skf.trimf(battery.universe, [0, 5, 10])
battery['good'] = skf.trimf(battery.universe, [5, 10, 10])

camera.view()

rule1 = ctrl.Rule(camera['poor'], review_cam['low'])
rule2 = ctrl.Rule(camera['average'], review_cam['medium'])
rule3 = ctrl.Rule(camera['good'], review_cam['high'])
rule4 = ctrl.Rule(screen['poor'], review_scr['low'])
rule5 = ctrl.Rule(screen['average'], review_scr['medium'])
rule6 = ctrl.Rule(screen['good'], review_scr['high'])
rule7 = ctrl.Rule(software['poor'], review_soft['low'])
rule8 = ctrl.Rule(software['average'], review_soft['medium'])
rule9 = ctrl.Rule(software['good'], review_soft['high'])
rule10 = ctrl.Rule(battery['poor'], review_bat['low'])
rule11 = ctrl.Rule(battery['average'], review_bat['medium'])
rule12 = ctrl.Rule(battery['good'], review_bat['high'])

review_camera = ctrl.ControlSystem([rule1, rule2, rule3])
review_screen = ctrl.ControlSystem([rule4, rule5, rule6])
review_software = ctrl.ControlSystem([rule7, rule8, rule9])
review_battery = ctrl.ControlSystem([rule10, rule11, rule12])

reviews_cam = ctrl.ControlSystemSimulation(review_camera)
reviews_scr = ctrl.ControlSystemSimulation(review_screen)
reviews_soft = ctrl.ControlSystemSimulation(review_software)
reviews_bat = ctrl.ControlSystemSimulation(review_battery)

x = df['camera'].to_numpy()
k = statistics.mean(x) 
reviews_cam.input['camera'] = statistics.mean(x)

reviews_cam.compute()

print(reviews_cam.output['reviewc'])
print(x)

review_cam.view(sim=reviews_cam)

x1 = df['screen'].to_numpy()
reviews_scr.input['screen'] = statistics.mean(x1)
reviews_scr.compute()
print(x1)

print(reviews_scr.output['reviewsc'])
review_scr.view(sim=reviews_scr)

x2 = df['software'].to_numpy()
reviews_soft.input['software'] = statistics.mean(x2)
reviews_soft.compute()

print(reviews_soft.output['reviewso'])
review_soft.view(sim=reviews_soft)

x3 = df['battery'].to_numpy()
reviews_bat.input['battery'] = statistics.mean(x3)
reviews_bat.compute()

print(reviews_bat.output['reviewb'])
review_bat.view(sim=reviews_bat)

print("according to the above charts we can see that a better SOFTWARE is what people require in their product.")