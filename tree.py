# นำเข้าแพ็คเกจที่จำเป็น
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import Image
import graphviz

# โหลดชุดข้อมูลไวน์
wine = pd.read_csv('D:\\!!WORK\\!!WU-ITD\\3\\1-67\\ITD62-364 Business Analytics\\!Group\\data\\kaggle\\input\\furniture-sales-data\\winequality-red.csv', encoding="utf8")

# ทำการแบ่งช่วงของค่าคุณภาพของไวน์เป็นคลาส "bad" และ "good"
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)

# การเข้ารหัสค่าคุณภาพ: Bad = 0, Good = 1
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])

# แยกชุดข้อมูลเป็นตัวแปรคุณลักษณะ (X) และตัวแปรเป้าหมาย (y)
X = wine.drop('quality', axis=1)
y = wine['quality']

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้การปรับมาตรฐาน (Standard scaling)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ใช้ RandomForestClassifier ในการสร้างโมเดล
rfc = RandomForestClassifier(n_estimators=10, random_state=42)  # กำหนดให้มี 10 ต้นไม้
rfc.fit(X_train, y_train)

# สร้างภาพ Decision Tree จาก Random Forest หนึ่งต้นไม้
estimator = rfc.estimators_[0]  # เลือกต้นไม้หนึ่งต้นจาก Random Forest

# สร้างภาพด้วย Graphviz
dot_data = export_graphviz(estimator, out_file=None, 
                           feature_names=wine.columns[:-1],  # คุณสมบัติในข้อมูลไวน์
                           class_names=['Bad', 'Good'],
                           filled=True, rounded=True, 
                           special_characters=True)

# แปลงเป็นกราฟและแสดงผล
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# แสดงภาพ Decision Tree หนึ่งต้นจาก Random Forest
graphviz.Source(dot_data).view()
