

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
1 load data: 
1.1 获取data 的类型（csv/xlsx/tsv ...）,若为其他类型，提示用户意外类型；
1.2 获取data的size： 500M：
    如果超过500M， 需分批次加载； 如果小于500M， 直接加载； 如果，超过10G：仅加载前500M；

1.3 data -> dataframe

2 data_基础分析：
2.1 行数/列数：（如果无法完全加载，仅展示加载的列数）
2.2 列名：（如果无法完全加载，仅展示加载的列数） + 每列的数据类型（int； float）  + 数据的分布（离散/连续）
2.3 缺失值分析： 每列缺失值的数量 + 缺失值占比（%）
2.4 重复值分析： 每列重复值的数量 + 重复值占比（%）
2.
假如： 数据是离散的-> 每个数值的频数，占比（%）
如果： 数据是连续的-> 均值、中位数、标准差、最小值、25%、50%、75%、最大值

"""








import os
from pathlib import Path
from typing import Optional, Tuple


import numpy as np


class DataLoad:
    # max data size per batch and data load size allowed 
    allow_data_size_one_batch = "500M"
    max_data_size = "10G"

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.file_size_bytes: Optional[int] = None
        self.file_type: Optional[str] = None   # csv / xlsx / tsv / other

    # --------- 工具方法 ---------
    @staticmethod
    def _parse_size_str(size_str: str) -> int:
        """
        把 '500M' / '10G' 之类的转成byte
        """
        size_str = size_str.strip().upper()
        if size_str.endswith("G"):
            num = float(size_str[:-1])
            return int(num * 1024 ** 3)
        elif size_str.endswith("M"):
            num = float(size_str[:-1])
            return int(num * 1024 ** 2)
        elif size_str.endswith("K"):
            num = float(size_str[:-1])
            return int(num * 1024)
        else:
            # 默认当成字节
            return int(float(size_str))

    # 1.2 GET DATA SIZE
    def _data_size_get(self) -> int:
        """
        返回文件大小（字节），顺便存一下到 self.file_size_bytes
        """
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {self.data_path}")
        size_bytes = path.stat().st_size
        self.file_size_bytes = size_bytes
        return size_bytes

    # 1.1 Get DATA TYPE
    def _data_type_get(self) -> str:
        """
        根据后缀判断文件类型：csv / xlsx / tsv / other
        """
        suffix = Path(self.data_path).suffix.lower()
        if suffix == ".csv":
            file_type = "csv"
        elif suffix in [".xls", ".xlsx"]:
            file_type = "xlsx"
        elif suffix in [".tsv", ".txt"]:
            file_type = "tsv"
        else:
            file_type = "other"

        self.file_type = file_type
        if file_type == "other":
            raise ValueError(f"non expected files: {suffix}，only support csv/xlsx/tsv")
        return file_type
    #this function estimate the trunk size of each batch
    def estimate_chunksize(self, target_size_str="500M", sep=",", sample_rows=100):
        sample_df = pd.read_csv(self.data_path, sep=sep, nrows=sample_rows)
        sample_size = sample_df.memory_usage(deep=True).sum()
        per_row_bytes = sample_size / sample_rows
        target_bytes = self._parse_size_str(target_size_str)
        chunksize= int(target_bytes / per_row_bytes)
        return chunksize







    def _load_csv_or_tsv(
        self,
        sep: str = ",",
    ) -> pd.DataFrame:
        """
        根据文件大小规则加载 csv/tsv 文件：
        - <=500M: 一次性加载
        - 500M~10G: 分批读取，再 concat
        - >10G: 仅加载大约 500M 的数据（通过 sample 估算行数）
        """
        size_bytes = self._data_size_get()
        one_batch_limit = self._parse_size_str(self.allow_data_size_one_batch)
        max_size = self._parse_size_str(self.max_data_size)
        filename = os.path.basename(self.data_path)

        # 1）小于等于 500M，一次性加载
        if size_bytes <= one_batch_limit:
            print(f"[INFO] File: {filename}, size: {size_bytes/1024/1024:.2f} MB, loaded all at once")
            df = pd.read_csv(self.data_path, sep=sep)
            return df

        # 2）500M ~ 10G，分批加载（但是最后还是会 concat 成一个 DataFrame）
        if size_bytes <= max_size:
            print(f"[INFO] File: {filename}, size: {size_bytes/1024/1024:.2f} MB, loaded all at once")
            chunksize = self.estimate_chunksize(target_size_str=self.allow_data_size_one_batch)

            chunks = []
            # 这里 chunksize 可以根据需要调大/调小
            for chunk in pd.read_csv(self.data_path, sep=sep, chunksize=chunksize):
                chunks.append(chunk)
                print(f"[INFO] 已加载 {len(chunks)} 个 chunk，当前总行数：{sum(len(c) for c in chunks)}")
            df = pd.concat(chunks, ignore_index=True)
            return df

        # 3）> 10G，只加载大约 500M
        print(f"[WARN] 文件大小 > {max_size/1024/1024/1024:.1f} GB，仅加载约 500M 数据进行分析。")

        # 先读取一小部分 sample，估算每行大概占多少字节
        sample_rows = 10_000
        sample_df = pd.read_csv(self.data_path, sep=sep, nrows=sample_rows)
        per_row_bytes = sample_df.memory_usage(deep=True).sum() / sample_rows
        target_bytes = one_batch_limit
        n_rows_500M = int(target_bytes / per_row_bytes)

        print(
            f"[INFO] sample 每行约 {per_row_bytes:.1f} 字节，"
            f"预计加载行数 ~ {n_rows_500M:,} 行。"
        )

        df = pd.read_csv(self.data_path, sep=sep, nrows=n_rows_500M)
        return df

    def _load_excel(self) -> pd.DataFrame:
      
        
        size_bytes = self._data_size_get()
        print(f"[INFO] Excel 文件大小 {size_bytes/1024/1024:.2f} MB，loaded it all at once")
        df = pd.read_excel(self.data_path)
        return df

    # 1.3 data -> DataFrame
    def loader(self) -> pd.DataFrame:
        file_type = self._data_type_get()

        if file_type == "csv":
            return self._load_csv_or_tsv(sep=",")
        elif file_type == "tsv":
            # tsv 
            return self._load_csv_or_tsv(sep="\t")
        elif file_type == "xlsx":
            return self._load_excel()
        else:
            
            raise ValueError(f"不支持的文件类型: {file_type}")


class DataBasicAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    # 2.1 行数 / 列数
    def data_size_info(self) -> pd.DataFrame:
        """
        RETURN DataFrame，WITH ROW# AND COL#
        """
        rows, cols = self.data.shape
        info = pd.DataFrame(
            {"rows": [rows], "cols": [cols]},
            index=["data_shape"]
        )
        return info

    # 2.2 列名 + 数据类型 + 离散/连续
    def data_column_info(self) -> pd.DataFrame:
        """
        对每一列输出：
        - column              列名
        - dtype               pandas 的 dtype
        - is_numeric          是否为数值型
        - is_datetime         是否为日期/时间型
        - n_unique            唯一值个数
        - unique_ratio        唯一值占总行数比例

        数值列专用：
        - numeric_type        'discrete' / 'continuous' / None

        非数值列专用：
        - non_numeric_type    'id' / 'categorical' / 'datetime' / 'text' / None

        方便后续分析：
        - is_categorical      是否“适合当类别变量用”（包括 0/1 这种 numeric）
        """
        df = self.data
        n_rows = len(df)

        infos = []
        for col in df.columns:
            s = df[col]
            dtype = s.dtype

            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)

            num_unique = s.nunique(dropna=True)
            unique_ratio = num_unique / n_rows if n_rows > 0 else np.nan

            numeric_type = None
            non_numeric_type = None
            is_categorical = False  # 统一的“类别变量候选”标记

            if is_numeric:
                # ---------- numeric: 判断 discrete / continuous ----------
                if num_unique <= 15 or (not np.isnan(unique_ratio) and unique_ratio < 0.05):
                    numeric_type = "discrete"
                else:
                    numeric_type = "continuous"

                # 对 numeric 里：unique 值很少的，尤其是 0/1、1/2/3 这些，当成类别变量候选
                if numeric_type == "discrete" and num_unique <= 10:
                    is_categorical = True

            else:
                # ---------- 非 numeric ----------
                if is_datetime:
                    non_numeric_type = "datetime"
                else:
                    # 几乎每一行都不一样，且唯一值很多：像 ID / 姓名
                    if (not np.isnan(unique_ratio) and unique_ratio > 0.9) and num_unique > 50:
                        non_numeric_type = "id"
                    # 唯一值数量不多，或者重复很多：典型 categorical
                    elif num_unique <= 50 or (not np.isnan(unique_ratio) and unique_ratio < 0.3):
                        non_numeric_type = "categorical"
                        is_categorical = True
                    else:
                        # 类似长文本、自由填写内容等
                        non_numeric_type = "text"

            infos.append(
                {
                    "column": col,
                    "dtype": str(dtype),
                    "is_numeric": is_numeric,
                    "is_datetime": is_datetime,
                    "n_unique": num_unique,
                    "unique_ratio": unique_ratio,
                    "numeric_type": numeric_type,
                    "non_numeric_type": non_numeric_type,
                    "is_categorical": is_categorical,
                }
            )

        return pd.DataFrame(infos)

    # 2.3 missing info
    def data_missing_info(self) -> pd.DataFrame:
        """
        每列缺失值数量 + 缺失值占比（%）
        """
        df = self.data
        n_rows = len(df)
        missing_count = df.isnull().sum()
        missing_ratio = missing_count / n_rows * 100 if n_rows > 0 else np.nan

        res = pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_ratio(%)": missing_ratio,
            }
        )
        res.index.name = "column"
        return res.reset_index()

    # 2.4 repeating numbers
    def data_duplicate_info(self) -> pd.DataFrame:
        """
        每列重复值数量 + 重复值占比（%）
        这里“重复值数量”定义为: 行数 - 该列唯一值个数
        """
        df = self.data
        n_rows = len(df)
        records = []
        for col in df.columns:
            s = df[col]
            n_unique = s.nunique(dropna=False)
            dup_count = max(n_rows - n_unique, 0)
            dup_ratio = dup_count / n_rows * 100 if n_rows > 0 else np.nan
            records.append(
                {
                    "column": col,
                    "duplicate_count": dup_count,
                    "duplicate_ratio(%)": dup_ratio,
                }
            )
        return pd.DataFrame(records)

    # 2.5 continuous数据统计（均值/中位数/标准差/分位数）
    def data_numeric_info(self) -> pd.DataFrame:
        info = self.data_column_info()
        cont_cols = info[info["numeric_type"] == "continuous"]["column"].tolist()

        if not cont_cols:
            return pd.DataFrame()

        desc = self.data[cont_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        desc = desc.rename_axis("column").reset_index()
        desc = desc.rename(
            columns={
                "mean": "mean",
                "50%": "median",
                "std": "std",
                "min": "min",
                "25%": "25%",
                "75%": "75%",
                "max": "max",
            }
        )
        return desc[["column", "mean", "median", "std", "min", "25%", "75%", "max"]]

    # 2.6 离散型数据统计：频数 + 占比
    def data_categorical_info(self, top_n: int = 20) -> dict:
        """
        对“适合当类别变量用”的列做统计：
        - numeric & numeric_type == 'discrete' 且 unique 值较少的（比如 0/1）
        - 非 numeric & non_numeric_type == 'categorical'

        返回一个 dict: {column_name: DataFrame(value, freq, ratio%)}
        top_n: 只展示前 top_n 个最常见取值，防止爆炸太长
        """
        col_info = self.data_column_info()

        # ✅ 用 is_categorical，而不是 distribution_type
        cat_cols = col_info[col_info["is_categorical"]]["column"].tolist()
        n_rows = len(self.data)

        result = {}
        for col in cat_cols:
            s = self.data[col]
            vc = s.value_counts(dropna=False).head(top_n)
            ratio = vc / n_rows * 100 if n_rows > 0 else np.nan

            df_stat = pd.DataFrame(
                {
                    "value": vc.index,
                    "freq": vc.values,
                    "ratio(%)": ratio.values,
                }
            )
            result[col] = df_stat

        return result


# ================= 使用示例（以 Titanic 为例） =================
if __name__ == "__main__":
    # 1. 加载数据
    train_path = "train.csv"
    test_path = "test.csv"
    gender_path = "gender_submission.csv"

    train_loader = DataLoad(train_path)
    train_df = train_loader.loader()

    test_loader = DataLoad(test_path)
    test_df = test_loader.loader()

    gender_loader = DataLoad(gender_path)
    gender_df = gender_loader.loader()

    print("TRAIN SHAPE:", train_df.shape)
    print("TEST SHAPE:", test_df.shape)
    print("GENDER SHAPE:", gender_df.shape)
    print("-" * 40)

    # 2. 对 train 做基础分析
    analyzer = DataBasicAnalysis(train_df)

    print("\n=== 数据形状 ===")
    print(analyzer.data_size_info())

    print("\n=== 列信息 ===")
    print(analyzer.data_column_info())

    print("\n=== 缺失值分析 ===")
    print(analyzer.data_missing_info())

    print("\n=== 重复值分析 ===")
    print(analyzer.data_duplicate_info())

    print("\n=== 连续型变量统计 ===")
    print(analyzer.data_numeric_info())

    print("\n=== 离散型变量统计（每列 top 20） ===")
    cat_info = analyzer.data_categorical_info(top_n=10)
    for col, df_cat in cat_info.items():
        print(f"\n[Column: {col}]")
        print(df_cat)





"""

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender = pd.read_csv("gender_submission.csv")
subset = train[["PassengerId", "Sex"]]
print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)
print("GENDER SHAPE:", gender.shape)
print("-" * 40)



#missing value
print("missing value")
print(train.isnull().sum())
print("original shape ",train.shape)

#handle missing data by dropping any rows with missing age and embarked
cleantrain=train.dropna(subset=['Age','Embarked'])
print("shape after ckeaning data" ,cleantrain.shape)




# ========== 2. First few rows ==========
print("=== HEAD ===")
print(cleantrain.head(), "\n")

# ========== 3. Basic info (missing, types) ==========
print("=== INFO ===")
print(cleantrain.info(), "\n")

# ========== 4. Summary statistics ==========
print("=== NUMERIC SUMMARY ===")
print(cleantrain.describe(), "\n")

print("=== CATEGORICAL SUMMARY ===")
print(cleantrain.describe(include='object'), "\n")

# ========== 5. Missing values ==========
print("=== MISSING VALUES ===")
print(cleantrain.isnull().sum(), "\n")

# ========== 6. Unique values ==========
print("=== UNIQUE VALUES ===")
print(cleantrain.nunique(), "\n")

# ========== 7. Target distribution ==========
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=cleantrain)
plt.title("Survival Distribution")
plt.show()

# ========== 8. Survival by Sex ==========
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=cleantrain)
plt.title("Survival by Sex")
plt.show()

# ========== 9. Survival by Class ==========
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=cleantrain)
plt.title("Survival by Passenger Class")
plt.show()

# ========== 10. Age distribution ==========
plt.figure(figsize=(6,4))
sns.histplot(cleantrain["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# ========== 11. Fare distribution ==========
plt.figure(figsize=(6,4))
sns.histplot(cleantrain["Fare"], kde=True, bins=30)
plt.title("Fare Distribution")
plt.show()

# ========== 12. Correlation heatmap ==========
plt.figure(figsize=(10,8))
sns.heatmap(cleantrain.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


"""