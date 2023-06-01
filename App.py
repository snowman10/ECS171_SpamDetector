from tkinter import Tk
from Constructor import Constructor

import pandas as pd
import traceback

from Models import NaiveBayes, Logistic, RandomForest, text_preprocess, stemmer

class App(Constructor):
  def __init__(self) -> object:

    # Initialization for GUI Class
    Constructor.__init__(
      self,
      parent=None,
      title="Spam Detection Program",
      grid={"col":2,"row":7},
      geometry="1200x800"
    )

    # Model Selection
    self.model = None
    self.models = []
    self.vectorizer = None
    self.vectorizers = []

    # Home Page Widgets
    self.Labels()
    self.Buttons()

    # Error Management
    self.set_error()

    # Reconfigure Center with Menus
    self.Configure()
    self.Entries()

    # Protocols
    self.setComplexDelete([self.root.quit, self.root.destroy, exit])
    self.root.bind("<Return>", self.Submit)

  def Labels(self) -> None:
    self.BuildLabel(preset="title", key="title", text=f"Spam Detection System")
    self.setLabelGrid("title", column=0, row=0, columnspan=4)

    text = "Logistic Model Statistics\n\n"
    model, stats, vectorizer = Logistic()
    text = text+stats
    self.BuildLabel(key="logistic", text=text)
    self.models.append(model)
    self.model = model
    self.vectorizer = vectorizer
    self.vectorizers.append(vectorizer)

    text = "Naive Bayes Model Statistics\n\n"
    model, stats, vectorizer = NaiveBayes()
    text = text+stats
    self.BuildLabel(key="bayes", text=text)
    self.models.append(model)
    self.vectorizers.append(vectorizer)

    text = "Random Forest Model Statistics\n\n"
    model, stats, vectorizer = RandomForest()
    text = text+stats
    self.BuildLabel(key="rf", text=text)
    self.models.append(model)
    self.vectorizers.append(vectorizer)

    self.BuildLabel(preset="title", key="status", text="Spam Status:")
    self.setLabelGrid(key="status", column=1, row=2)

    self.BuildLabel(preset="title", key="value", text="N/A")
    self.setLabelGrid(key="value", column=1, row=3)

    self.BuildLabel(preset="title", key="model", text="Current Model:\nLogistic")
    self.setLabelGrid(key="model", column=1, row=1)

    self.BuildLabel(key="message", text=f"Message:\n{''.center(100)}")
    self.setLabelGrid(key="message", column=1, row=4, rowspan=2)

    self.setLabelGrid(key="logistic", column=0, row=2)
    self.setLabelGrid(key="bayes", column=0, row=4)
    self.setLabelGrid(key="rf", column=0, row=6)

  def wrapString(self, string, n=10):
    words = string.split()
    partial = [(' '.join(words[i:i+n])) for i in range(0, len(words), n)]
    partial = [a.center(100) for a in partial]
    return '\n'.join(partial)

  def Submit(self, event=None):

    df = pd.DataFrame([{
      "Text": self.getEntry("input")
    }])
    data_text = df["Text"].copy()
    data_text.apply(text_preprocess)
    data_text.apply(stemmer)

    matrix = self.vectorizer.transform(data_text)
    pred = self.model.predict(matrix)

    if pred == [0]:
      self.updateLabelColor("value", "green")
      self.updateLabel("value", "NOT SPAM")
    else:
      self.updateLabelColor("value", "red")
      self.updateLabel("value", "SPAM")

    if self.getEntry('input').split() == []:
      self.updateLabel("message", f"Message:\n{self.getEntry('input').center(100)}")
    else:
      self.updateLabel("message", f"Message:\n{self.wrapString(self.getEntry('input'))}")

  def Buttons(self) -> None:
    self.BuildButton(key="logistic", text="Load Logistic", width=25, height=2)
    self.BuildButton(key="bayes", text="Load Naive Bayes", width=25, height=2)
    self.BuildButton(key="rf", text="Load Random Forest", width=25, height=2)
    self.BuildButton(key="submit", text="Submit", width=25, height=2)

    self.setButtonGrid(key="logistic", column=0, row=1)
    self.setButtonGrid(key="bayes", column=0, row=3)
    self.setButtonGrid(key="rf", column=0, row=5)
    self.setButtonGrid(key="submit", column=1, row=6)

    self.setButtonCmd(key="logistic", function=self.setLogistic, kwargs={})
    self.setButtonCmd(key="bayes", function=self.setBayes, kwargs={})
    self.setButtonCmd(key="rf", function=self.setRF, kwargs={})
    self.setButtonCmd(key="submit", function=self.Submit, kwargs={})

  def setLogistic(self):
    self.model = self.models[0]
    self.vectorizer = self.vectorizers[0]
    self.updateLabel(key="model", text="Current Model:\nLogistic")

  def setBayes(self):
    self.model = self.models[1]
    self.vectorizer = self.vectorizers[1]
    self.updateLabel(key="model", text="Current Model:\nNaive Bayes")

  def setRF(self):
    self.model = self.models[2]
    self.vectorizer = self.vectorizers[2]
    self.updateLabel(key="model", text="Current Model:\nRandom Forest")

  def Entries(self) -> None:
    self.BuildEntry(key="input", width=25, size=18)
    self.setEntryGrid(key="input", column=1, row=5)
    self.setEntryFocus(key="input")

  def callback_error(self, *args) -> None:
    message = 'Generic Error:\n\n'
    message += traceback.format_exc()
    print(message, args)
    exit(1)

  def set_error(self) -> None:
    Tk.report_callback_exception = self.callback_error