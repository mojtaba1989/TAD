import numpy as np
import pandas as pd
import os
import python_data_fusion as pdf
import matplotlib.pyplot as plt
from scipy import stats


# FILE_NAME = '01-Oct-2019-14-04'
# FILE_NAME = '20-Nov-2019-11-06'
FILE_NAME = "04-Feb-2020-12-37"

CWD_PATH = os.getcwd()
PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-lined-' + FILE_NAME + '.csv')

Data = pd.read_csv(PATH_TO_CSV, sep=',')

PATH_TO_RESULTS = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-cleaned-' + FILE_NAME + '.csv')
columns = ["Vernier", "L", "Lambda", "Lhat", "Lambdahat"]
Results = pd.DataFrame(np.zeros([Data.__len__(), 5]), columns=columns)

center = np.array([[-.1], [.28]])

for INDEX in range(Data.__len__()):
    Results.loc[INDEX, "Vernier"] = np.array(pdf.csvCellReader(Data.loc[INDEX, "vernier"]))

    x_c = np.array(pdf.csvCellReader(Data.loc[INDEX, "Camera X"])) - center[0]
    y_c = np.array(pdf.csvCellReader(Data.loc[INDEX, "Camera Y"])) - center[1]

    y_c = y_c[x_c.argsort()]
    x_c.sort()

    if len(x_c) > 1:
        Results.loc[INDEX, "L"] = np.mean([np.sqrt(x_c[0]**2+y_c[0]**2),
                                           np.sqrt(x_c[-1]**2+y_c[-1]**2)])
        Results.loc[INDEX, "Lambda"] = np.sqrt((x_c[0]-x_c[-1])**2+(y_c[0]-y_c[-1])**2)

    else:
        Results.loc[INDEX, "L"] = np.nan
        Results.loc[INDEX, "Lambda"] = np.nan

    if INDEX >= 1 and ~np.isnan(Results.loc[INDEX, "Lambda"]):
        Results.loc[INDEX, "Lhat"] = (Results.loc[INDEX-1, "Lhat"] * (INDEX)
                                      + Results.loc[INDEX, "L"])/(INDEX+1)
        Results.loc[INDEX, "Lambdahat"] = (Results.loc[INDEX-1, "Lambdahat"] * (INDEX)
                                           + Results.loc[INDEX, "Lambda"])/(INDEX+1)
    elif INDEX == 0:
        Results.loc[INDEX, "Lhat"] = Results.loc[INDEX, "L"]
        Results.loc[INDEX, "Lambdahat"] = Results.loc[INDEX, "Lambda"]

    else:
        Results.loc[INDEX, "Lhat"] = Results.loc[INDEX-1, "Lhat"]
        Results.loc[INDEX, "Lambdahat"] = Results.loc[INDEX-1, "Lambdahat"]

Results.to_csv(os.path.join(CWD_PATH, FILE_NAME, 'Dimension-' + FILE_NAME + '.csv'), index=True)
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(Results["Vernier"], label='True Angle')
line2, = ax1.plot(Results["L"], label='L')
line3, = ax1.plot(Results["Lambda"], label='Lambda')

Results = Results.dropna()

k2, p = stats.normaltest(Results["L"])

print("p = {:g}".format(p))

k2, p = stats.normaltest(Results["Lambda"])
print("p = {:g}".format(p))

# plt.hist(Results["Lambda"])


ax1.legend()
plt.show()
plt.waitforbuttonpress()