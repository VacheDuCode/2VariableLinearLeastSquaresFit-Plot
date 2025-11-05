# 2VariableLinearLeastSquaresFit&Plot
This repo receives an xlsx file and fits and plots the equations y=ax, y=ax+b to the data. The fitting uses least-squares regression with Gaussian statistics. The uncertainty in each data point can be manually changed to ensure an ideal goodness-of-fit so P=0.5 (see Bevington & Robinson 2012). It's not polished and has some junk code and files...
The data from sampledata Sheet 2 is meant to have an intercept !=0, so y=ax plots are weird - this is expected.
CLASS is the main version, so getdata&fit&plot does not have all the features/fixes that CLASS (I didn't want to maintain both versions).
Effort has been made to make the code as clean as possible, but the job is not done yet (ref "Clean Code" by Martin)
Things to be done/refactor:
1. Breakup big functions into small one-purpose fucntions, one level of abstraction (high, mid, low)
2. Move variable to close to use (see "Clean Code"-Martin for more)
3. De-nest some of the if ... for loops
4. Raise exception if inputs are wrong type
5. Implement weighted average (a=0, b=...)
6. Apply lamda function to...? (where appropriate)
7. Redo iterators so no range(len(xxx))??? not important I think
8. refactoring functions to return a, b, chi_squared, reduced_chi_squared values (This would basically be a whole new program)?? - possible as tuples or dicts...?