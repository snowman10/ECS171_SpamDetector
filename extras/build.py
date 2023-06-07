from Constructor import Constructor

def submitCall(data):
  submit(data())

def submit(data):
  print(x:=data)


  if x in ['spam', 'bad']:
    window.updateLabelColor("output", "red")
    window.updateLabel("output", "SPAM")
  else:
    window.updateLabelColor("output", "green")
    window.updateLabel("output", "NOT SPAM")

window = Constructor(title="ECS 171 Project", grid={"col":1,"row":4})
window.BuildLabel(preset="title", key="wow", text="ECS 171 Final Project")
window.setLabelGrid("wow", row=0, column=0, columnspan=1)
window.BuildEntry(key="predict", width=20)
window.setEntryGrid("predict", column=0, row=2, columnspan=1)
window.BuildLabel(preset="subtitle", key="output", text="N/A", fg='black')
window.setLabelGrid(key="output", column=0, row=1, columnspan=1)
window.BuildButton(preset="submit", key="submit")
window.setButtonGrid(key="submit", column=0, row=3, columnspan=1)
window.setButtonCmd(key="submit", function=submitCall, kwargs={"data":lambda:window.getEntry(key="predict")})
window.root.bind("<Return>", lambda x:submit(data=window.getEntry(key="predict")))

window.root.mainloop()




