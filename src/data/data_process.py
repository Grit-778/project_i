

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
    allow_data_size_one_batch = "500M"
    max_data_size = "10G"

    def __init__(self, data_path: str):
        self.data_path = data_path

    @staticmethod
    def _parse_size_str(size_str: str) -> int:
        size_str = size_str.strip().upper()
        if size_str.endswith("G"):
            return int(float(size_str[:-1]) * 1024 ** 3)
        if size_str.endswith("M"):
            return int(float(size_str[:-1]) * 1024 ** 2)
        if size_str.endswith("K"):
            return int(float(size_str[:-1]) * 1024)
        return int(float(size_str))

    def _data_size_get(self) -> int:
        p = Path(self.data_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        return p.stat().st_size

    def _data_type_get(self) -> str:
        suffix = Path(self.data_path).suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix in [".tsv", ".txt"]:
            return "tsv"
        if suffix in [".xls", ".xlsx"]:
            return "xlsx"
        return "other"

    def estimate_chunksize(self, target_size_str="500M", sep=",", sample_rows=1000) -> int:
        sample_df = pd.read_csv(self.data_path, sep=sep, nrows=sample_rows)
        sample_bytes = sample_df.memory_usage(deep=True).sum()
        per_row = sample_bytes / max(sample_rows, 1)
        target_bytes = self._parse_size_str(target_size_str)
        chunksize = max(int(target_bytes / per_row), 1)
        return chunksize

    def _estimate_nrows_for_target_bytes(self, target_bytes: int, read_fn, **read_kwargs) -> int:
        """
        用 sample 估算每行占用字节数，从而估算 nrows，使读取量约等于 target_bytes。
        read_fn: pd.read_csv 或 pd.read_excel
        """
        sample_rows = 5000
        sample_df = read_fn(nrows=sample_rows, **read_kwargs)
        per_row = sample_df.memory_usage(deep=True).sum() / max(sample_rows, 1)
        nrows = max(int(target_bytes / per_row), 1)
        return nrows

    def loader(self, sep: str = ",") -> pd.DataFrame:
        """
       
        - csv/tsv: 支持 chunksize 分块；>10G 只读约 500M
        - xlsx: pd.read_excel 无 chunksize；>500M 时退化为只读约 500M（按 nrows 估算）
        """
        file_type = self._data_type_get()
        if file_type == "other":
            raise ValueError("Unsupported file type. Only csv/tsv/xlsx supported.")

        size_bytes = self._data_size_get()
        one_batch_limit = self._parse_size_str(self.allow_data_size_one_batch)
        max_size = self._parse_size_str(self.max_data_size)
        filename = os.path.basename(self.data_path)

        # ---- 选择读取函数 ----
        if file_type in ["csv", "tsv"]:
            used_sep = "\t" if file_type == "tsv" else sep
            read_fn = lambda **kw: pd.read_csv(self.data_path, sep=used_sep, **kw)
            can_chunk = True
        else:  # xlsx
            read_fn = lambda **kw: pd.read_excel(self.data_path, **kw)
            can_chunk = False

        # ---- <= 500M：直接读 ----
        if size_bytes <= one_batch_limit:
            print(f"[INFO] File: {filename}, size: {size_bytes/1024/1024:.2f} MB, loaded all at once")
            return read_fn()

        # ---- 500M ~ 10G：csv/tsv 可 chunk；xlsx 只能退化为读部分 ----
        if size_bytes <= max_size:
            if can_chunk:
                print(f"[INFO] File: {filename}, size: {size_bytes/1024/1024:.2f} MB, loading in chunks then concat")
                chunksize = self.estimate_chunksize(target_size_str=self.allow_data_size_one_batch,
                                                   sep=("\t" if file_type == "tsv" else sep))
                chunks = []
                for chunk in read_fn(chunksize=chunksize):
                    chunks.append(chunk)
                    print(f"[INFO] loaded {len(chunks)} chunk(s), rows so far: {sum(len(c) for c in chunks)}")
                return pd.concat(chunks, ignore_index=True)
            else:
                # Excel：不能 chunk，只能读“约 500M”行数
                print(f"[WARN] File: {filename} is Excel; pandas read_excel has no chunksize. "
                      f"Falling back to load ~{self.allow_data_size_one_batch} only.")
                nrows = self._estimate_nrows_for_target_bytes(one_batch_limit, read_fn)
                print(f"[INFO] estimated nrows ~ {nrows:,} for ~{self.allow_data_size_one_batch}")
                return read_fn(nrows=nrows)

        # ---- > 10G：只读前 500M（csv/tsv/xlsx 都按 nrows 估算）----
        print(f"[WARN] File: {filename}, size > {self.max_data_size}. Load only ~{self.allow_data_size_one_batch}.")

        nrows = self._estimate_nrows_for_target_bytes(one_batch_limit, read_fn)
        print(f"[INFO] estimated nrows ~ {nrows:,} for ~{self.allow_data_size_one_batch}")
        return read_fn(nrows=nrows)

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
                elif (not np.isnan(unique_ratio) and unique_ratio > 0.9) and num_unique > 50:
                        numeric_type = "id"
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
                    "n_unique": num_unique,
                    "unique_ratio": unique_ratio,
                    "numeric_type": numeric_type,
                    "non_numeric_type": non_numeric_type,
                    "is_categorical": is_categorical,
                }
            )

        df = pd.DataFrame(infos)
        df["unique_ratio"] = df["unique_ratio"].round(2)
        return df


    # 2.3 missing info
    def data_missing_info(self) -> pd.DataFrame:
        """
        每列缺失值数量 + 缺失值占比（%）
        """
        df = self.data
        n_rows = len(df)
        missing_count = df.isnull().sum()
        missing_ratio = (missing_count / n_rows * 100).round(2)if n_rows > 0 else np.nan

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
            dup_ratio = round(dup_count / n_rows * 100, 2) if n_rows > 0 else np.nan

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
        desc = desc.round(2)

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
            ratio = (vc / n_rows * 100).round(2) if n_rows > 0 else np.nan

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
    def run_basic_eda( df: pd.DataFrame, top_n: int = 10, show: bool = True) -> dict:
       
    
   

        """
    对 DataFrame 运行基础 EDA 分析

    Returns:
        dict 包含所有分析结果，方便后续保存 / 导出 / 单元测试
        """
        analyzer = DataBasicAnalysis(df)

        results = {}

        results["data_shape"] = analyzer.data_size_info()
        results["column_info"] = analyzer.data_column_info()
        results["missing_info"] = analyzer.data_missing_info()
        results["duplicate_info"] = analyzer.data_duplicate_info()
        results["numeric_info"] = analyzer.data_numeric_info()
        results["categorical_info"] = analyzer.data_categorical_info(top_n=top_n)

        if show:
            print("\n=== 数据形状 ===")
            print(results["data_shape"])

            print("\n=== 列信息 ===")
            print(results["column_info"])

            print("\n=== 缺失值分析 ===")
            print(results["missing_info"])

            print("\n=== 重复值分析 ===")
            print(results["duplicate_info"])

            print("\n=== 连续型变量统计 ===")
            print(results["numeric_info"])

            print(f"\n=== 离散型变量统计（每列 top {top_n}） ===")
        for col, df_cat in results["categorical_info"].items():
            print(f"\n[Column: {col}]")
            print(df_cat)

        return results

    result = run_basic_eda(train_df)




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