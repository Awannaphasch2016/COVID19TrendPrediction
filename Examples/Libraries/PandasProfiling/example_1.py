import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.DataFrame(
    np.random.rand(100, 5),
    columns=['a', 'b', 'c', 'd', 'e']
)

profile = ProfileReport(df, title='Pandas Profiling Report')

profile.to_widgets()
profile.to_notebook_iframe()

profile.to_file("your_report.html")

# As a string
json_data = profile.to_json()

# As a file
profile.to_file("your_report.json")

# ==== example 2 
# Change the config when creating the report
profile = df.profile_report(title="Pandas Profiling Report", pool_size=1)

# Change the config after
profile.set_variable("html.minify_html", False)

profile.to_file("output.html")
