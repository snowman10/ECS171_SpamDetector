from gui import Constructor
import tkinter as tk

window = Constructor(grid={"col":5,"row":5})
window.BuildLabel(key="wow", text="Hello")
window.setLabelGrid("wow", row=0, column=0)
window.BuildEntry(key="predict")
window.setEntryGrid("predict", row=1, column=1)

window.root.mainloop()