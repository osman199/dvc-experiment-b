from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

from config import Config


Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

load_boston(str(Config.ORIGINAL_DATASET_FILE_PATH))
data = load_boston()
df = pd.DataFrame(data=data['data'], columns = data['feature_names'])
df['MEDV'] = data.target
# print(df.head)
df.to_csv(str(Config.ORIGINAL_DATASET_FILE_PATH), sep = ',', index = False)


df_train, df_test = train_test_split( df, test_size=0.2, random_state=28750 )

#df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH))

df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)
