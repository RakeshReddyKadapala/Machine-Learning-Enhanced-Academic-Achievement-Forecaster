import pymysql

def db_connect():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='Rakeshreddy36.',
        port=3306,
        database='students_performance_analysis'
    )
    return connection, connection.cursor()
