import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style="darkgrid")
tips = sns.load_dataset("tips")
tips['index'] = np.arange(tips.shape[0])
# ax = sns.pointplot(x="tip", y="day", data=tips, join=False)
# ax = sns.lineplot(x="tip", y="sex", data=tips)
ax = sns.lineplot(x="index", y="tip",hue="sex", data=tips)

plt.show()
