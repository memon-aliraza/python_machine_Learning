Dataset and Problem Statement
-----------------------------
Dataset contains data from 50 different companies. 

Problem Statement:

How much in this year company spend on Research and Development, Administrarion and Marketing in different states. Finally what was the profit by the end!

According to the profit, where companies perform batter state wise. If all states are equal then which company perform batter, one who spend more on Marketing or R&D. 

How they assist companies, do they look companies spend more on R&D or Marketing. Which factor yeilds more profit. 

Something like company located in New York, spend less in Marketing and more in R&D yeilds more profit. 

Multiple linear Regression =>   y = b0 + b1x1 + b2x2 + b3x3 + ........
                    
                                y = b0 + b1(R&D) + b2(Admin) + b3(marketing) + b4(state) 

The "state" attribute is the categorical attribute. Therefore, we need use Dummy variables. 

If we have two variables (New York and California) then we can replace state column with:

    New York   = 1, and 
    California = 0.

But when we multiply it with b4 from b4(state), the California becomes 0. But in real it is not the case. Here in this situation it will become the default case and coefficient of California will become included in b0.  

When dealing with dummy variables, always ommit one.


Building a Model
---------------- 
How to select which attributes are going to be the part of our Model.

1.  All-in
    
    Select all independent variables only if you have prior knowledge. You know all independent variables are significiently important. 
    
2.  Backward Elimination

    
