from App import App

def getBarPlot():
  from Models import NaiveBayes, Logistic, RandomForest

  model, class_report, vectorizer = NaiveBayes(True)
  print(NaiveBayesA:=class_report["accuracy"])
  print(NaiveBayesPH:=class_report["ham"]["precision"])
  print(NaiveBayesRH:=class_report["ham"]["recall"])
  print(NaiveBayesPS:=class_report["spam"]["precision"])
  print(NaiveBayesRS:=class_report["spam"]["recall"])
  model, class_report, vectorizer = RandomForest(True)
  print(RandomForestA:=class_report["accuracy"])
  print(RandomForestPH:=class_report["ham"]["precision"])
  print(RandomForestRH:=class_report["ham"]["recall"])
  print(RandomForestPS:=class_report["spam"]["precision"])
  print(RandomForestRS:=class_report["spam"]["recall"])
  model, class_report, vectorizer = Logistic(True)
  print(LogisticA:=class_report["accuracy"])
  print(LogisticPH:=class_report["ham"]["precision"])
  print(LogisticRH:=class_report["ham"]["recall"])
  print(LogisticPS:=class_report["spam"]["precision"])
  print(LogisticRS:=class_report["spam"]["recall"])

  import matplotlib.pyplot as plt
  import numpy as np

  species = ("Logistic", "Naive Bayes", "Random Forest")
  penguin_means = {
    "Accuracy": (float(f"{100*LogisticA}"[:5]), float(f"{100*NaiveBayesA}"[:5]), float(f"{100*RandomForestA}"[:5])),
    "Spam Precision": (float(f"{100*LogisticPS}"[:5]), float(f"{100*NaiveBayesPS}"[:5]), float(f"{100*RandomForestPS}"[:5])),
    "Spam Recall": (float(f"{100*LogisticRS}"[:5]), float(f"{100*NaiveBayesRS}"[:5]), float(f"{100*RandomForestRS}"[:5])),
    "Ham Precision": (float(f"{100*LogisticPH}"[:5]), float(f"{100*NaiveBayesPH}"[:5]), float(f"{100*RandomForestPH}"[:5])),
    "Ham Recall": (float(f"{100*LogisticRH}"[:5]), float(f"{100*NaiveBayesRH}"[:5]), float(f"{100*RandomForestRH}"[:5])),
  }

  x = np.arange(len(species))  # the label locations
  width = 0.15  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')

  for attribute, measurement in penguin_means.items():
      offset = width * multiplier
      rects = ax.bar(x + offset, measurement, width, label=attribute)
      ax.bar_label(rects, padding=3)
      multiplier += 1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Percent (%)')
  ax.set_title('Accuracy, Precision, and Recall Comparison')
  ax.set_xticks(x + width, species)
  ax.legend(loc='upper left', ncols=5)
  ax.set_ylim(50, 110)

  fig.set_figwidth(10)

  plt.show()

def printClassReports():
  from Models import NaiveBayes, Logistic, RandomForest

  model, class_report, vectorizer = NaiveBayes(False)
  print(class_report)
  model, class_report, vectorizer = RandomForest(False)
  print(class_report)
  model, class_report, vectorizer = Logistic(False)
  print(class_report)

def main():
  try:
    app = App()
    app.root.mainloop()

  except Exception as e:
    print(f'Unknown Error Detected\n\n{e}')

if __name__ == '__main__':
  # main()
  printClassReports()
  # getBarPlot()