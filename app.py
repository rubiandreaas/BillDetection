import json,time
from camera import VideoCamera
from flask import Flask, render_template, request, jsonify, Response
import requests
import base64,cv2


app=Flask(__name__)
output=[]#("message stark","hi")]
@app.route('/')
def home_page():
    return render_template("IY_Home_page.html",result=output)

def gen(camera):
    found = False
    sound = ''
    showText = ''
    cont = 0
    while True:
        frame= camera.get_frame(cont,found,sound)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result',methods=["POST","GET"])
def Result():
    if request.method=="POST":
        print(list(request.form.values()))
        result=list(request.form.values())[0]
        if result.lower()=="restart":
            output.clear()
        else:
            try:
                r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": result})
                print("Bot says, ")
                for i in r.json():
                    bot_message = i['text']
                    print(f"{i['text']}")
                output.extend([("message parker",result),("message stark",bot_message)])
            except:
                output.extend([("message parker", result), ("message stark", "We are unable to process your request at the moment. Please try again...")])

        print(output)
        return render_template("IY_Home_page.html",result=output)

if __name__=="__main__":
    app.run(debug=True)#,host="192.168.43.161")



