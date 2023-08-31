import tkinter 



if __name__ == "__main__":

    window  = tkinter.Tk()
    window.rowconfigure(1,weight=1)
    window.columnconfigure(1,weight=1)
    window.geometry("1920x1080")
    main_frame      = tkinter.Frame(window) 
    main_frame.rowconfigure(0,weight=1)
    main_frame.rowconfigure(1,weight=20)
    main_frame.columnconfigure(0,weight=3)
    main_frame.columnconfigure(1,weight=15)
    main_frame.grid(row=1,column=1,sticky='nsew')


    title_frame     = tkinter.Label(main_frame,text="Chess Gui")
    view_frame      = tkinter.Frame(main_frame,bg="red")
    input_frame     = tkinter.Frame(main_frame,bg="blue")
    input_frame.columnconfigure(0,weight=2)
    input_frame.columnconfigure(1,weight=1)

    
    n_rows  = 3 
    for _ in range(n_rows):
        input_frame.rowconfigure(1,weigh=1)
        input_frame.rowconfigure(2,weigh=1)
        input_frame.rowconfigure(3,weight=1)

    

    #BUILD BUTTONS 
    buttons         = {"newgame":(tkinter.Button(input_frame,text="New game"),tkinter.Text(input_frame)),
                       "sendmove":(tkinter.Button(input_frame,text="Make Move"),tkinter.Text(input_frame)),
                       "popmove":(tkinter.Button(input_frame,text="Undo Move"),tkinter.Text(input_frame))}
    for i,button in enumerate(buttons):
        buttons[button].grid(column=0,row=i,sticky='ew')
    

    input_frame.grid(row=1,column=0,sticky='nsew')
    view_frame.grid(row=1,column=1,sticky='nsew')
    title_frame.grid(row=0,column=0,columnspan=2)
    window.grid()


    #view_frame
    window.mainloop()