from pysr import PySRRegressor,TemplateExpressionSpec
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sympy
import os



x = np.hstack((np.loadtxt('cord_1.txt').reshape(-1, 1),np.loadtxt('cord_2.txt').reshape(-1, 1),np.loadtxt('dinv_1.txt').reshape(-1, 1),np.loadtxt('dinv_2.txt').reshape(-1, 1)))  # 读取坐标数据并调整形状
# Samples of numertcal CDE entries, and the kinematic variables x,y
y1 = x  
y2 = np.loadtxt('cord_1.txt').reshape(-1, 1)
# According to Miles Crammers, y2 is not used in this mission. We keep it for the convenience of formatting PySR's output

template = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x", "y", "dx", "dy"],
    combine="""
        fdx = D(f, 1)(x, y)
        fdy = D(f, 2)(x, y)
        
        abs2(fdx - dx) + abs2(fdy - dy)
    """
)
# Define a template, to solve a 2-variable PDE problem

model = PySRRegressor(
    # 
    procs=40,
    populations=120,
    niterations=500,
    variable_names=["x","y","dx","dy"],
    population_size=120,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log"],
    model_selection="best",
    nested_constraints={
        "log": {"log":0},
    },  
    # In this family we need only log function, and log(log) nestings are forbidden.
    complexity_of_constants = 5,
    elementwise_loss="my_loss(predicted, target) = predicted",
    expression_spec = template,
    maxsize=100,
    turbo=True,
    bumper=True,
    precision=64,
)
model.fit(x, y2)
