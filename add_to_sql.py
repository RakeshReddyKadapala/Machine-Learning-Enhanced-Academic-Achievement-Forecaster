from db import db_connect
import pandas as pd

def add_to_sql(file_name,table_name,column_names):
    db,cursor = db_connect()
    cursor.execute("USE students_performance_analysis")
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} (id INT PRIMARY KEY AUTO_INCREMENT,metric_no VARCHAR(20) NOT NULL, result VARCHAR(100) NOT NULL)")
    data = pd.read_excel(file_name)
    # drop any columns that are not in the column_names list
    data = data[column_names]
    sql = f'INSERT INTO {table_name} (metric_no, result) VALUES '
    for i in range(len(data)):
        sql += f'("{data.iloc[i,0]}", "{data.iloc[i,1]}"),'
    sql = sql[:-1]
    cursor.execute(sql)
    db.commit()
    db.close()


add_to_sql('./background.xlsx' ,'background', ['MATRIC_NO','GOT_STATUS'])
print('background table added')
add_to_sql('./employement.xls' ,'employeement', ['MATRIC_NO','WORKING_STATUS'])
print('employeement table added')
add_to_sql('./result.xlsx' ,'result', ['MATRIC_NO','RESULT_DESCRIPTION'])
print('result table added')