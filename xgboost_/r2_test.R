# load dataframe
df <- read.csv('./data/pred_and_predict_6.csv')

# ectract data
y_test <- df$BDE
y_pred <- df$BDE_pred

# test
cor.test(y_test, y_pred)