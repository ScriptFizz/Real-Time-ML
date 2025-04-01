import pandas as pd

from layouts.input_layout import Input

def convert_input_to_df(input_data: Input) -> pd.DataFrame:
	#Convert input into pandas DataFrame
	data = {}
	for key, value in input_data.dict().items():
		data[key] = [value]
	return pd.DataFrame(data)
