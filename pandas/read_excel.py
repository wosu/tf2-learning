import pandas as pd
file='D:\\yoyi\\TA\\TD数据统计.xlsx'
datas = pd.read_excel(file,sheet_name='Sheet2')

print(datas.groupby('感兴趣的行业').count())

with open('../datas/kurun.txt',mode='w',encoding='utf-8') as writer:
    for index, row in datas.iterrows():
        imei_md5 = row[0]
        if row[1] == "金融/服务行业":
            sex = "male"
        else:
            sex = "female"
        writer.write(imei_md5 + "\t" + sex + "\n")