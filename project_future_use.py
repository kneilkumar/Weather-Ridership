import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# transport monthly total boardings for each mode: cleaning etc

month_for_merge = ["Jul 2023", "Aug 2023", "Sep 2023","Oct 2023", "Nov 2023", "Dec 2023", "Jan 2024",
               "Feb 2024", "Mar 2024", "Apr 2024", "May 2024", "Jun 2024", "Jul 2024",
               "Aug 2024", "Sep 2024", "Oct 2024", "Nov 2024", "Dec 2024", "Jan 2025",
               "Feb 2025", "Mar 2025", "Apr 2025", "May 2025"]
month_for_merge_cols = [1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11]

total_monthly_24 = pd.read_csv("grand_totals_23-24.csv").drop([0,1,2,3,4,5])
total_monthly_25 = pd.read_csv("grand_totals_24_25.csv").drop([0,1,2,3,4,5])
grand_totals = pd.concat([total_monthly_25, total_monthly_24], ignore_index=True)
grand_totals = grand_totals.rename(columns={"Unnamed: 0":"Months", "Patronage by mode":"Bus", "Unnamed: 2":"Train", "Unnamed: 3":"Ferry","Unnamed: 4":"Grand Total"})
grand_totals = grand_totals.drop(["Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9",
                                  "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14", "Unnamed: 15",
                                  "Unnamed: 16"], axis=1)
grand_totals = grand_totals.iloc[::-1].reset_index(drop=True)
rows, cols = grand_totals.shape
for i in range(0,rows):
    for j in range(0, cols):
        grand_totals.iat[i,j] = grand_totals.iat[i,j].replace(",", "")
grand_totals[["Bus", "Train", "Ferry", "Grand Total"]] = grand_totals[["Bus", "Train", "Ferry", "Grand Total"]].astype(float)

# Exploratory Analysis

fig, ax = plt.subplots()
ax.plot(grand_totals['Months'], grand_totals['Bus'], label='Bus')
ax.plot(grand_totals['Months'], grand_totals['Train'], label='Train')
ax.plot(grand_totals['Months'], grand_totals['Ferry'], label='Ferry')
ax.plot(grand_totals['Months'], grand_totals['Grand Total'], label='Grand Total')
plt.legend(loc='upper right')
plt.xticks(rotation=45)

plt.show()

# Feature Engineering to help train model

feature_df_total = pd.DataFrame({'Months': grand_totals["Months"], 'Total Riders': grand_totals['Grand Total']})
feature_df_bus = pd.DataFrame({'Months': grand_totals["Months"], 'Total Bus Riders': grand_totals['Bus']})
feature_df_train = pd.DataFrame({'Months': grand_totals["Months"], 'Total Train Riders': grand_totals['Train']})
feature_df_ferry = pd.DataFrame({'Months': grand_totals["Months"], 'Total Ferry  Riders': grand_totals['Ferry']})

# Shift Lag so that model can start to use autocorrelation

feature_df_list = [feature_df_total, feature_df_bus, feature_df_train, feature_df_ferry]
rows, cols = grand_totals.shape
for m in range(0, len(feature_df_list)):
    i = 0
    j = 0
    g = 0
    h = 0
    lag_2_vals = []
    lag_13_vals = []
    lag_4_vals = []
    lag_12_vals = []
    for k in range(0,rows):
        if i >= 2:
            lag_2_vals.append(feature_df_list[m].iloc[i-2,1])
        else:
            lag_2_vals.append(np.nan)
        if j >= 4:
            lag_4_vals.append(feature_df_list[m].iloc[j-4,1])
        else:
            lag_4_vals.append(np.nan)
        if g >= 13:
            lag_13_vals.append(feature_df_list[m].iloc[g-13,1])
        else:
            lag_13_vals.append(np.nan)
        if h >= 12:
            lag_12_vals.append(feature_df_list[m].iloc[h-12,1])
        else:
            lag_12_vals.append(np.nan)
        i += 1
        j += 1
        g += 1
        h += 1
    lag_df = pd.DataFrame({'Months': month_for_merge, '2 Month Lag': lag_2_vals, '13 Month Lag': lag_13_vals,
                           '4 Month Lag': lag_4_vals, '12 Month Lag': lag_12_vals})
    feature_df_list[m] = pd.merge(feature_df_list[m], lag_df, on='Months', how='inner')

# Cyclical Features

is_summer = pd.DataFrame({'Months': month_for_merge, 'Summer': [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0]})
is_school_holidays = pd.DataFrame({'Months': month_for_merge,  'School Holidays': [1,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0]})
sin_list = [np.sin(2*np.pi*nums/12) for nums in month_for_merge_cols]
cos_list = [np.cos(2*np.pi*nums/12) for nums in month_for_merge_cols]
month_df = pd.DataFrame({'Months': month_for_merge,'month_nums': month_for_merge_cols, 'sin_vals':sin_list, 'cos_vals':cos_list})

for i in range(0, len(feature_df_list)):
    feature_df_list[i] = pd.merge(feature_df_list[i], is_summer, on='Months', how='inner')
    feature_df_list[i] = pd.merge(feature_df_list[i], is_school_holidays, on='Months', how='inner')
    feature_df_list[i] = pd.merge(feature_df_list[i], month_df,on='Months', how='inner')

# Feature Engineered DataFrames

feature_df_total = feature_df_list[0]
feature_df_bus = feature_df_list[1]
feature_df_train = feature_df_list[2]
feature_df_ferry = feature_df_list[3]

# Engineering Feature validation:

print(feature_df_total[['Total Riders', '2 Month Lag', '4 Month Lag', '13 Month Lag', '12 Month Lag']].corr())

fig_2_month, ax_2_month = plt.subplots()
ax_2_month.scatter(feature_df_total['2 Month Lag'], feature_df_total['Total Riders'])
ax_2_month.set_title("2 Month Lag against Total Ridership")
plt.show()

fig_4_month, ax_4_month = plt.subplots()
ax_4_month.scatter(feature_df_total['4 Month Lag'],feature_df_total['Total Riders'])
ax_4_month.set_title("4 Month Lag against Total Ridership")
plt.show()

fig_13_month, ax_13_month = plt.subplots()
ax_13_month.scatter(feature_df_total['13 Month Lag'],feature_df_total['Total Riders'])
ax_13_month.set_title("13 Month Lag against Total Ridership")
plt.show()

fig_12_month, ax_12_month = plt.subplots()
ax_12_month.scatter(feature_df_total['12 Month Lag'],feature_df_total['Total Riders'])
ax_12_month.set_title("12 Month Lag against Total Ridership")
plt.show()

print(feature_df_total.groupby('Summer')['Total Riders'].mean())
print(feature_df_total.groupby('School Holidays')['Total Riders'].mean())

fig_sin, ax_sin = plt.subplots()
ax_sin.plot(feature_df_total['Months'], feature_df_total['sin_vals'], color="#0000FF")
ax_sin2 = ax_sin.twinx()
ax_sin2.plot(feature_df_total['Months'], feature_df_total['Total Riders'], color="#FF0000")
plt.xticks(rotation=45)
plt.show()

fig_cos, ax_cos = plt.subplots()
ax_cos.plot(feature_df_total['Months'], feature_df_total['cos_vals'], color="#0000FF")
ax_cos2 = ax_cos.twinx()
ax_cos2.plot(feature_df_total['Months'], feature_df_total['Total Riders'], color="#FF0000")
plt.xticks(rotation=45)
plt.show()

# baseline model, still part of feature engineering validation using grand total

X = pd.DataFrame(feature_df_total, columns=['12 Month Lag', 'Summer', 'School Holidays', 'sin_vals', 'cos_vals'])
X['12 Month Lag'] = X['12 Month Lag'].replace(np.nan, 0)
y = pd.DataFrame(feature_df_total, columns=['Total Riders'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

predictive_model = LinearRegression()
predictive_model.fit(X_trained_scaled, y_train)
ridership_predictions = predictive_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, ridership_predictions)
rmse = root_mean_squared_error(y_test, ridership_predictions)
print(f"mae = {mae:.2f}    rmse = {rmse:.2f}")
print(r2_score(y_test, ridership_predictions))
relative_error = np.mean(abs(y_test - ridership_predictions)/y_test)
print(relative_error)

# baseline model works pretty well, adding more features such as a rolling average and then testing to get higher R^2

rolling_avg_list = []
i = 0
j = 3
k = 0
while j != len(feature_df_total['Total Riders'] - 1):
    if k < 3:
        rolling_avg_list.append(0)
        k += 1
        continue
    rolling_avg_list.append(np.mean(feature_df_total['Total Riders'][i:j]))
    i += 1
    j += 1

rolling_avgs = pd.DataFrame({'Months': month_for_merge, '4 Month Rolling Avg': rolling_avg_list})
feature_df_total = pd.merge(feature_df_total, rolling_avgs, on='Months', how='inner')

X_new = pd.DataFrame(feature_df_total, columns=['12 Month Lag', 'Summer', 'School Holidays', 'sin_vals', 'cos_vals', '4 Month Rolling Avg'])
X_new['12 Month Lag'] = X_new['12 Month Lag'].replace(np.nan, 0)
y_new = pd.DataFrame(feature_df_total, columns=['Total Riders'])

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
X_train_new_scaled = scaler.fit_transform(X_train_new)
X_test_new_scaled = scaler.fit_transform(X_test_new)

new_predictive_model = LinearRegression()
new_predictive_model.fit(X_train_new_scaled, y_train_new)
new_ridership_predictions = new_predictive_model.predict(X_test_new_scaled)

mae = mean_absolute_error(y_test_new, new_ridership_predictions)
rmse = root_mean_squared_error(y_test_new, new_ridership_predictions)
print(f"mae = {mae:.2f}    rmse = {rmse:.2f}")
print(f"R^2 = {r2_score(y_test_new, new_ridership_predictions):.2f}")
relative_error = np.mean(abs(y_test_new - new_ridership_predictions)/y_test_new)
print(f"relative error = {relative_error:.2f}")
