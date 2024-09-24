'''
Created on 08.07.2021

@author: bened
'''
import io
import os
import PySimpleGUI as sg
from PIL import Image
import main as mn
import requests
import time

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

#opens at "Get best matches!", gets the input image, displays results
def open_results(img, method):

    start_time = time.time()

    #set the fields for result images
    layout = [
        [sg.Image(key="-OUTPUT1-"), sg.Image(key="-OUTPUT2-"), sg.Image(key="-OUTPUT3-")],
        [sg.Image(key="-OUTPUT4-"), sg.Image(key="-OUTPUT5-"), sg.Image(key="-OUTPUT6-")],
        [sg.Image(key="-OUTPUT7-"), sg.Image(key="-OUTPUT8-"), sg.Image(key="-OUTPUT9-")],
    ]
    
    #open new window with defined layout, modal -> have to close before doing something in the other window
    #finalize -> auto-fill for images   
    window = sg.Window("Results", layout, modal=True, finalize=True)
    
    #extract the urls of result images for download and display
    link_list = mn.main(img, method)
    url_list = []

    for elem in link_list:
        url_list.append(elem[1])
    
    #open every result image            
    for i, url in enumerate(url_list):
        i = i + 1
        im = Image.open(requests.get(url, stream=True).raw)
        im.thumbnail((300,300))
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        temp_i = str(i) + "-"
        window["-OUTPUT" + temp_i].update(data = bio.getvalue())

    print("Images displayed in --- %s seconds ---" % (time.time() - start_time))
    
    #while window is open
    while True:
        event, values = window.read()
        #standard for closing windows
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

#method for the upload window   
def main():
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image")
        ],
        [
            sg.Button("Run Histogram-Sort"),
            sg.Button("Run Feature-Only-Sort"),
            sg.Button("Run ORB?-Sort")
        ],
    ]
    
    #open window with defined layout
    window = sg.Window("Image Search", layout)
    
    #while window is open
    while True:
        event, values = window.read()
        #standard for closing windows
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        #runs if "Load Image" button is clicked, displays the upload image,
        #necessary before running the search function, otherwise no image data
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((600, 600))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
        #runs if "Get best matches!" button is clicked
        if event == "Run Histogram-Sort":
            #run image search with input file
            open_results(values["-FILE-"], "hist")

        elif event == "Run Feature-Only-Sort":
            #run image search with input file
            open_results(values["-FILE-"], "feature")

        elif event == "Run ORB?-Sort":
            #run image search with input file
            open_results(values["-FILE-"], "orb")

                       
    window.close()
    
if __name__ == "__main__":
    main()
