from tkinter import Toplevel, Tk
from tkinter import Button, Label, Entry, Text, Frame
from tkinter.ttk import Combobox, Scrollbar

def Error(widget):
  print(f"{widget} Key Error")
  exit(1)

class WindowConfig:

  def Configure(self) -> None:

    # Center
    self.root.update_idletasks()
    frm_width = self.root.winfo_rootx() - self.root.winfo_x()
    win_width = (width:=self.root.winfo_width()) + 2 * frm_width
    titlebar_height = self.root.winfo_rooty() - self.root.winfo_y()
    win_height = (height:=self.root.winfo_height()) + titlebar_height + frm_width
    x = self.root.winfo_screenwidth() // 2 - win_width // 2
    y = self.root.winfo_screenheight() // 2 - win_height // 2
    self.root.geometry(f'{width}x{height}+{x}+{y-10}')
    self.root.deiconify()

    # Grid
    for index in range(self.columns):
      self.root.columnconfigure(index, weight=1)
    for index in range(self.rows):
      self.root.rowconfigure(index, weight=1)

class CustomWidget:

  def _setGrid(self, widget, key, column, row, columnspan, rowspan, padx, pady):
    self.widgets[widget][key].grid(
      column=column, columnspan=columnspan,
      row=row, rowspan=rowspan,
      padx=padx, pady=pady
    )

  def _setState(self, widget, key, state):
    self.widgets[widget][key].configure(
      state="normal" if state==True or state=="normal" else "disabled"
    )

  def _setCommand(self, widget, key, function, kwargs):
    self.widgets[widget][key].configure(
      command=lambda:function(**kwargs)
    )

  def _clearText(self, widget, key, position):
    self.widgets[widget][key].delete(
      first=position, last="end"
    )

  def _updateText(self, widget, key, position, text):
    self.widgets[widget][key].insert(
      position, text
    )

  def _updateLabel(self, widget, key, text):
    self.widgets[widget][key].configure(text=text)

  def _getState(self, widget, key):
    return self.widgets[widget][key]["state"]

  def _getContents(self, widget, key):
    return w.get("1.0", "end-1c") if widget == "entrybox" else (w:=self.widgets[widget][key].get())

  def _setColor(self, widget, key, color):
    self.widgets[widget][key].configure(fg=color)

  def _setFocus(self, widget, key):
    self.widgets[widget][key].focus_set()

class CustomButton(CustomWidget):

  def BuildButton(
    self, preset=None, key="key", text="text", font="Courier", size=16, style="bold",
    width=16, height=1,
    bg="Light Grey", fg="Black",
    padx=0, pady=0
  ):
    if key in self.widgets["buttons"]: Error("Button")

    if preset == "home":
      self.widgets["buttons"][key] = Button(
        master=self.root,
        text=text, font=[font, 20, style],
        width=21, height=4,
        bg=bg, fg=fg,
        padx=padx, pady=pady,
      )
    elif preset == "home1":
      self.widgets["buttons"][key] = Button(
        master=self.root,
        text=text, font=[font, 20, style],
        width=45, height=3,
        bg=bg, fg=fg,
        padx=padx, pady=pady,
      )
    elif preset == "home2":
      self.widgets["buttons"][key] = Button(
        master=self.root,
        text=text, font=[font, 20, style],
        width=40, height=3,
        bg=bg, fg=fg,
        padx=padx, pady=pady,
      )
    elif preset in ["close", "submit"]:
      self.widgets["buttons"][key] = Button(
        master=self.root,
        text=f"{preset[0].upper()}{preset[1:]}", font=[font, size, style],
        width=width, height=height,
        bg=bg, fg=fg,
        padx=padx, pady=pady
      )
    else:
      self.widgets["buttons"][key] = Button(
        master=self.root,
        text=text, font=[font, size, style],
        width=width, height=height,
        bg=bg, fg=fg,
        padx=padx, pady=pady,
      )

  def setButtonGrid(self, key, column, row, columnspan=1, rowspan=1, padx=0, pady=0):
    self._setGrid("buttons", key, column, row, columnspan, rowspan, padx, pady)

  def setButtonStatus(self, key, state):
    self._setState("buttons", key, state)

  def setButtonCmd(self, key, function, kwargs):
    self._setCommand("buttons", key, function, kwargs)

class CustomEntry(CustomWidget):

  def BuildEntry(
    self, preset=None, key="key", text=None, font="Courier", size=16,
    width=16, bg="White", fg="Black", lock=False
  ):
    if key in self.widgets["entries"]: Error("Entry")

    if preset == "something":
      self.widgets["entries"][key] = Entry(
        master=self.root,
        font=[font, size],
        width=width,
        bg=bg, fg=fg
      )
    else:
      self.widgets["entries"][key] = Entry(
        master=self.root,
        font=[font, size],
        width=width,
        bg=bg, fg=fg
      )

    if text: self.widgets["entries"][key].insert(0, text)
    if lock: self.widgets["entries"][key].configure(state="disabled")

  def setEntryGrid(self, key, column, row, columnspan=1, rowspan=1, padx=0, pady=0):
    self._setGrid("entries", key, column, row, columnspan, rowspan, padx, pady)

  def setEntryStatus(self, key, state):
    self._setState("entries", key, state)

  def updateEntry(self, key, text):
    state = self._getState("entries", key)
    self.setEntryStatus(key, True)
    self._clearText("entries", key, 0)
    self._updateText("entries", key, 0, text)
    self.setEntryStatus(key, state)

  def getEntry(self, key):
    return self._getContents("entries", key)

  def setEntryFocus(self, key):
    self._setFocus("entries", key)

class CustomLabel(CustomWidget):

  def BuildLabel(
    self, preset=None, key="key", text="text", font="Courier", size=8, style="bold", fg='black'
  ):
    if key in self.widgets["labels"]: Error("Label")

    if preset == "title":
      self.widgets["labels"][key] = Label(
        master=self.root,
        text=text,
        font=[font, 24, style],
        fg=fg
      )
    elif preset == "subtitle":
      self.widgets["labels"][key] = Label(
        master=self.root,
        text=text,
        font=[font, 16, style],
        fg=fg
      )
    else:
      self.widgets["labels"][key] = Label(
        master=self.root,
        text=text,
        font=[font, size, style],
        fg=fg
      )

  def setLabelGrid(self, key, column, row, columnspan=1, rowspan=1, padx=0, pady=0):
    self._setGrid("labels", key, column, row, columnspan, rowspan, padx, pady)

  def updateLabel(self, key, text):
    self._updateLabel("labels", key, text)

  def updateLabelColor(self, key, color):
    self._setColor("labels", key, color)

class CustomDropdown(CustomWidget):

  def BuildDropdown(
    self, preset=None, key="key", values=None, font="Courier", size=16, style="bold",
    width=16, lock=False, index=None
  ):
    if key in self.widgets["entries"]: Error("Dropdown")

    if preset == "something":
      pass
    else:
      self.widgets["dropdown"][key] = Combobox(
        master=self.root,
        values=values, font=[font, size, style],
        width=width
      )

    if index: self.widgets["dropdown"][key].insert(0, values[index])
    if lock: self.widgets["dropdown"][key].configure(state="disabled")

  def setDropdownGrid(self, key, column, row, columnspan=1, rowspan=1, padx=0, pady=0):
    self._setGrid("dropdown", key, column, row, columnspan, rowspan, padx, pady)

  def getDropdown(self, key):
    return self._getContents("dropdown", key)

class CustomScrollBar(CustomWidget):

  def BuildScrollbar_X(self, key="key", widget_reference=None):
    if f"{key}_scrollx" in self.widgets[widget_reference]: Error(f"Scroll X")

    self.widgets["scrollbars"][f"{key}_xscroll"] = Scrollbar(
      self.widgets["frame"][key],
      command=self.widgets[widget_reference][key].xview,
      orient="horizontal")
    self.widgets[widget_reference][key].configure(xscrollcommand=self.widgets["scrollbars"][f"{key}_xscroll"].set)
    self.widgets["scrollbars"][f"{key}_xscroll"].grid(column=0, row=1, sticky='ew')

  def BuildScrollbar_Y(self, key="key", widget_reference=None):
    if f"{key}_scrolly" in self.widgets[widget_reference]: Error(f"Scroll Y")

    self.widgets["scrollbars"][f"{key}_yscroll"] = Scrollbar(
      self.widgets["frame"][key],
      command=self.widgets[widget_reference][key].yview,
      orient="vertical")
    self.widgets[widget_reference][key].configure(yscrollcommand=self.widgets["scrollbars"][f"{key}_yscroll"].set)
    self.widgets["scrollbars"][f"{key}_yscroll"].grid(column=1, row=0, sticky='ns')

class CustomLargeEntry(CustomScrollBar):

  def BuildLargeEntry(
    self, preset=None, key="key", body=None, font="Courier", size=16, style="normal",
    width=70, bg="White", fg="Black", wrap="word", lock=False
  ):
    if key in self.widgets["entrybox"]: Error("Entry Box")
    if key in self.widgets["frame"]: Error("Frame")

    self.widgets["frame"][key] = Frame(self.root)
    if preset == "xy":
      self.widgets["entrybox"][key] = Text(
        self.widgets["frame"][key],
        width=width,
        font=[font, size, style],
        fg=fg, bg=bg,
        wrap="none"
      )
    else:
      self.widgets["entrybox"][key] = Text(
        self.widgets["frame"][key],
        width=width,
        font=[font, size, style],
        fg=fg, bg=bg,
        wrap=wrap
      )

    if body: self.widgets["entrybox"][key].insert("end", body)
    if lock: self.widgets["entrybox"][key].configure(state="disabled")

    self.BuildScrollbar_Y(key, "entrybox")
    if preset == "xy":
      self.BuildScrollbar_X(key, "entrybox")
    self.widgets["entrybox"][key].grid(column=0, row=0)

  def setLargeEntryGrid(self, key, column, row, columnspan=1, rowspan=1, padx=0, pady=0):
    self._setGrid("frame", key, column, row, columnspan, rowspan, padx, pady)

  def setLargeEntryStatus(self, key, state):
    self._setState("entries", key, state)

  def updateLargeEntry(self, key, text):
    state = self._getState("entries", key)
    self.setEntryStatus(key, True)
    self._clearText("entries", key, 0)
    self._updateText("entries", key, 0, text)
    self.setEntryStatus(key, state)

  def getLargeEntry(self, key):
    self._getContents("dropdown", key)

class PresetWidgets(
  CustomButton,
  CustomEntry,
  CustomDropdown,
  CustomLabel,
  CustomLargeEntry,
): pass

class Constructor(WindowConfig, PresetWidgets):

  def __init__(self, parent=None, title="Default Window Title", grid=None, geometry="500x500") -> object:

    # New Window or Popup
    if parent != None:
      self.root = Toplevel(parent)
      self.root.transient(parent)
    else:
      self.root = Tk()

    # Grid
    self.root.geometry(geometry)
    self.columns = grid["col"]
    self.rows = grid["row"]
    self.Configure()

    # Other Aesthetics
    self.root.resizable(False, False)
    self.root.wm_title(title)

    # Initialize Widgets
    self.resetWidgets()

  def resetWidgets(self):
    self.widgets = {
      "buttons":{},
      "dropdown":{},
      "labels":{},
      "entries":{},
      "entrybox":{},
      "scrollbars":{'x':None,'y':None},
      "frame":{}
    }

  def setDelete(self, function, **kwargs):
    self.root.protocol("WM_DELETE_WINDOW", lambda:function(kwargs))

  def setComplexDelete(self, functions):
    self.root.protocol("WM_DELETE_WINDOW", lambda:[function() for function in functions])

  def hideWindow(self):
    self.root.withdraw()

  def showWindow(self):
    self.root.deiconify()
