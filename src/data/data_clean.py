from typing import List, Tuple
from pandas import DataFrame

from utils.logger_config import logger
from src.data.data_process import DataLoad, DataBasicAnalysis

class CleanAssistant:
    @staticmethod
    def get_ratio_columns(df: DataFrame) -> Tuple[str]:
        # 获取所有比例列的名称
        # 0 < ratio < 5%
        ratio_less_5 = df[(df['missing_ratio(%)'] > 0) & 
                                   (df['missing_ratio(%)'] < 5)]
        # 5% <= ratio <= 30%
        ratio_5_to_30 = df[(df['missing_ratio(%)'] >= 5) & 
                                    (df['missing_ratio(%)'] <= 30)]
        # ratio > 30%
        ratio_greater_30 = df[df['missing_ratio(%)'] > 30]
        # ratio > 80%
        ratio_greater_80 = df[df['missing_ratio(%)'] > 80]
        # 合并所有比例列
        all_ratio_columns = (ratio_less_5['column'].tolist(), 
                            ratio_5_to_30['column'].tolist(), 
                            ratio_greater_30['column'].tolist(), 
                            ratio_greater_80['column'].tolist())
        return all_ratio_columns

class CleanData:
    def __init__(self, data_path: str):
        data_load = DataLoad(data_path)
        self.raw_data = data_load.loader()
        self.basic_analysis = DataBasicAnalysis(self.raw_data)
        logger.info(f"Data loaded successfully from {data_path}")
    
    def missing_value(self):
        missing_stauts = self.basic_analysis.data_missing_info()
        logger.debug(f"Missing values status: {missing_stauts} {missing_stauts.columns}")
        need_process_columns = CleanAssistant.get_ratio_columns(missing_stauts)
        less_5, from_5_to_30, greater_30, greater_80 = need_process_columns
        logger.info(f"less_5: {less_5}")
        logger.info(f"from_5_to_30: {from_5_to_30}")
        logger.info(f"greater_30: {greater_30}")
        logger.info(f"greater_80: {greater_80}")


    def remove_duplicates(self):
        logger.info("Removing duplicates")
        # 在这里添加去重的代码
        pass

if __name__ == "__main__":
    raw_data_path = 'data/raw/titanic data/train.csv'
    # clean_data = CleanData(raw_data_path)
    # clean_data.missing_value()