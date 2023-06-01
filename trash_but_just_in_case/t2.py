  list_alpha = np.arange(1/100000, 20, 0.11)
  score_train = np.zeros(len(list_alpha))
  score_test = np.zeros(len(list_alpha))
  recall_test = np.zeros(len(list_alpha))
  precision_test= np.zeros(len(list_alpha))
  count = 0

  for alpha in list_alpha:
    bayes = MultinomialNB(alpha=alpha)
    bayes.fit(data_train, class_train)
    score_train[count] = bayes.score(data_train, class_train)
    score_test[count]= bayes.score(data_test, class_test)
    recall_test[count] = metrics.recall_score(class_test, bayes.predict(data_test))
    precision_test[count] = metrics.precision_score(class_test, bayes.predict(data_test))
    count = count + 1

  matrix = np.matrix(
    np.c_[list_alpha, score_train, score_test, recall_test, precision_test]
  )
  models = pd.DataFrame(
    data=matrix,
    columns=['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision']
  )

  # models.head(n=10)
  # best_index = models['Test Accuracy'].idxmax()
  # print(models.iloc[best_index, :])
