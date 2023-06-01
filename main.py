from App import App

def main():
  try:
    app = App()
    app.root.mainloop()

  except Exception as e:
    print(f'Unknown Error Detected\n\n{e}')

if __name__ == '__main__':
  main()