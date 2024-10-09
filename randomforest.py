# นำเข้าแพ็คเกจที่จำเป็น
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# โหลดชุดข้อมูลไวน์
wine = pd.read_csv('winequality-red.csv")

custom_palette = sns.color_palette("husl", len(wine['quality'].unique()))
# แสดงความสัมพันธ์ระหว่างคุณภาพของไวน์กับคุณสมบัติต่าง ๆ
features_to_plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                    'sulphates', 'alcohol']

# วนลูปเพื่อสร้างกราฟ bar plot ของแต่ละคุณสมบัติเทียบกับคุณภาพของไวน์
for feature in features_to_plot:
    plt.figure(figsize=(10,6))
    sns.barplot(x='quality', y=feature, data=wine, palette=custom_palette)
    plt.title(f'Relationship between Wine Quality and {feature.capitalize()}')  # ตั้งชื่อกราฟ
    plt.show()

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

# ใช้การปรับมาตรฐาน (Standard scaling) เพื่อให้ได้ผลลัพธ์ที่ดีขึ้น
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # ทำการปรับมาตรฐานกับข้อมูลฝึก
X_test = sc.transform(X_test)  # ทำการปรับมาตรฐานกับข้อมูลทดสอบ

# ใช้ RandomForestClassifier ในการสร้างโมเดล
rfc = RandomForestClassifier(n_estimators=200)  # สร้างโมเดลด้วยต้นไม้การตัดสิน 200 ต้น
rfc.fit(X_train, y_train)  # ฝึกโมเดลกับชุดข้อมูลฝึก
pred_rfc = rfc.predict(X_test)  # ทำนายผลลัพธ์ด้วยชุดข้อมูลทดสอบ

# คำนวณความแม่นยำและแสดงรายงานการจำแนกผลลัพธ์
accuracy = accuracy_score(y_test, pred_rfc)
report = classification_report(y_test, pred_rfc)
conf_matrix = confusion_matrix(y_test, pred_rfc)

# แสดงผลลัพธ์การประเมินโมเดล
print(f"\nAccuracy: {accuracy*100:.2f}%")  # แสดงค่าความแม่นยำในรูปแบบเปอร์เซ็นต์
print("Classification Report:")
print(report)  # แสดงรายงานการจำแนกประเภท

# แสดง Confusion Matrix
print("Confusion Matrix:")
print(conf_matrix)

# แสดง heatmap สำหรับ Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
