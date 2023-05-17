from gui import Constructor
import tkinter as tk

window = Constructor(grid={"col":5,"row":5})
window.BuildLabel(preset="title", key="wow", text="ECS 171 Final Project")
window.setLabelGrid("wow", row=0, column=0, columnspan=5)
window.BuildEntry(key="predict", width=20)
window.setEntryGrid("predict", row=3, column=0, columnspan=5)
window.BuildButton(preset="submit", key="submit")
window.setButtonGrid(key="submit", column=0, row=4, columnspan=5)

window.root.mainloop()