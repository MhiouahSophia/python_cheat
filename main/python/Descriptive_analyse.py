import matplotlib.pyplot as plt

#exemple1:
train_by_date = train1.groupby(['year', 'monthOfYear'])['y_true_quantity'].sum().reset_index()
test_by_date = df_test.groupby(['year', 'monthOfYear'])['y_true_quantity'].sum().reset_index()

index_train = train_by_date.monthOfYear.unique()
index_test = test_by_date.monthOfYear.unique()
year = [2017, 2018]
plt.figure(figsize=(10, 10))
plt.bar(index_train, train_by_date.y_true_quantity)
plt.bar(index_test, test_by_date.y_true_quantity)
plt.xlabel('Month', fontsize=5)
plt.ylabel('Number of Booking', fontsize=5)
plt.xticks(index_train, index_train, fontsize=10, rotation=30)
plt.title('Number of booking per month 2017 vs 2018')
plt.legend(year, loc=2)
plt.show()

#exemple2:
ind = np.arange(len(test_alt.y_test))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(20,10))

rects1 = ax.bar(ind - width/2, test_alt.y_test, width,
                label='y_true')
rects2 = ax.bar(ind + width/2, test_alt.y_pred, width,
                label='y_pred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Boooking')
ax.set_title('Alternative')
ax.set_xticks(ind)
ax.set_xticklabels(test_alt.alternative, rotation=90)
ax.legend()