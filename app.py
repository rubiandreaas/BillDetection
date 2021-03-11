import json,time
from camera import VideoCamera
from flask import Flask, render_template, request, jsonify, Response
import requests
import base64
import cv2


app =Flask(__name__)
output=[]#("message stark","hi")]
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')



def gen(camera):
    found = False
    sound = ''
    showText = ''
    cont = 0
    while True:
        data= camera.get_frame(cont,found,sound)

        frame=data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
    


if __name__=="__main__":
    app.run(debug=True)



