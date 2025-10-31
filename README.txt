# 2VariableLinearLeastSquaresFit&Plot
This repo receives an xlsx file and fits and plots the equations y=ax, y=ax+b to the data. The uncertainty in each data point can be manually changed to ensure an ideal goodness-of-fit so P=0.5 (see Bevington & Robinson 2012). It's not polished and has some junk code and files...
The data from sampledata Sheet 2 is meant to have an intercept !=0, so y=ax plots are weird - this is expected.
CLASS is the main version, so getdata&fit&plot does not have all the features/fixes that CLASS (I didn't want to maintain both versions).
Things to be done/refactor:
3. De-nest some of the if ... for loops
4. Implement weighted average (a=0, b=...)
5. Work out implementing fitting Ax^2+b / Ax^2+Bx+C, A/x+B and/or Aexp(Bx+C)+D...(Bev & Rob?)
6. Redo iterators so no range(len(xxx))??? not important I think
7. refactoring functions to return a, b, chi_squared, reduced_chi_squared values (This would basically be a whole new program)?? - possible as tuples or dicts...?