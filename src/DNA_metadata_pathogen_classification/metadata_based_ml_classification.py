
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# 1. 데이터 로드
metadata_path = "/content/drive/MyDrive/BIO_NLP/new_task_microbiom/sample_information_from_prep_2086.tsv"
metadata_df = pd.read_csv(metadata_path, sep="\t")

# 2. 사용할 타겟: co_occur_acathamoeba (0.3 초과 여부로 이진 분류)
df = metadata_df[metadata_df["co_occur_acathamoeba"] != 'not applicable'].copy()


df["co_occur_acathamoeba"] = df["co_occur_acathamoeba"].astype(float)
df["label"] = (df["co_occur_acathamoeba"] > 0.3).astype(int)

# 3. 입력 특성 선정
feature_cols = [
    'ph', 'depth', 'elevation', 'latitude', 'longitude',
    'env_biome', 'env_material', 'env_feature',
    'sample_type', 'cur_land_use', 'cur_vegetation', 'drainage_class',
    'geo_loc_name', 'qiita_empo_1', 'qiita_empo_2', 'qiita_empo_3'
]




# 4. 범주형 변수 라벨 인코딩
cat_cols = df[feature_cols].select_dtypes(include='object').columns
df_encoded = df.copy()
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 5. 결측치 제거
df_encoded = df_encoded.dropna(subset=feature_cols + ['label'])

# 6. 학습/테스트 분할
X = df_encoded[feature_cols]
y = df_encoded["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)






# 7. 모델 학습 (불균형 보정)
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 8. 예측 및 평가
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)





import xgboost as xgb
print(xgb.__version__)  # 1.0 이상이어야 GPU 지원


# 7. XGBoost 모델 (GPU 사용)
xgb_clf = xgb.XGBClassifier(
    tree_method='gpu_hist',          # GPU 기반 학습
    predictor='gpu_predictor',       # GPU 예측
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42
)

# 8. 모델 학습
xgb_clf.fit(X_train, y_train)

# 9. 평가
y_pred = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred))




# considering "not applicable"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드
metadata_path = "/content/drive/MyDrive/BIO_NLP/new_task_microbiom/sample_information_from_prep_2086.tsv"
metadata_df = pd.read_csv(metadata_path, sep="\t")

# 2. 이진 분류 라벨 설정: 'not applicable' → 0, 숫자(float) 값 → 1
df = metadata_df.copy()
df["label"] = (df["co_occur_acathamoeba"] != 'not applicable').astype(int)

# 3. 입력 특성 선정
feature_cols = [
    'ph', 'depth', 'elevation', 'latitude', 'longitude',
    'env_biome', 'env_material', 'env_feature',
    'sample_type', 'cur_land_use', 'cur_vegetation', 'drainage_class',
    'geo_loc_name', 'qiita_empo_1', 'qiita_empo_2', 'qiita_empo_3'
]

# 4. 범주형 변수 라벨 인코딩
cat_cols = df[feature_cols].select_dtypes(include='object').columns
df_encoded = df.copy()
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 5. 결측치 제거
df_encoded = df_encoded.dropna(subset=feature_cols + ['label'])

# 6. 학습/테스트 분할
X = df_encoded[feature_cols]
y = df_encoded["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)






# 7. 모델 학습 (불균형 보정)
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 8. 예측 및 평가
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


import xgboost as xgb
print(xgb.__version__)  # 1.0 이상이어야 GPU 지원


# 7. XGBoost 모델 (GPU 사용)
xgb_clf = xgb.XGBClassifier(
    tree_method='gpu_hist',          # GPU 기반 학습
    predictor='gpu_predictor',       # GPU 예측
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42
)

# 8. 모델 학습
xgb_clf.fit(X_train, y_train)

# 9. 평가
y_pred = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred))



