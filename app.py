import json,time
from camera import VideoCamera
from flask import Flask, render_template, request, jsonify, Response
import requests
import base64
import cv2


app=Flask(__name__)
output=[]#("message stark","hi")]
@app.route('/')
def home_page():
    return render_template("IY_Home_page.html",result=output)

@app.route('/camera',methods=['POST'])
def camera():
    cap=cv2.VideoCapture(0)
    while True:
        ret,img=cap.read()
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/cam.png",img)

        # return render_template("camera.html",result=)
        time.sleep(0.1)
        return json.dumps({'status': 'OK', 'result': "static/cam.png"})
        if cv2.waitKey(0) & 0xFF ==ord('q'):
            break
    cap.release()
    # file="/home/ashish/Downloads/THOUGHT.png"
    # with open(file,'rb') as file:
    #     image=base64.encodebytes(file.read())
    #     print(type(image))
    # return json.dumps({'status': 'OK', 'user': user, 'pass': password});
    return json.dumps({'status': 'OK', 'result': "static/cam.png"});



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
    app.run(debug=True)#,host="192.168.43.161")



