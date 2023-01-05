TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'

n_splits = 5 # 5 product code in training set
drop_features = ['id', 'product_code', 'attribute_0', 'attribute_1']
select_features = ['loading', 'measurement_17', 'missing_3', 'missing_5', 'missing_loading', 'area']