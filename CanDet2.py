# Liberary Import
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import webbrowser
import sqlite3
import pyodbc
import os
import sqlite3
import pyodbc
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
import pyodbc
from datetime import date

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U3TLO97\SQL2017TEST;'
                      'Database=Xcenter;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()

coursor2 = conn.cursor()





def LungCancerDetection(i):
    # Relative Path
    img = i
    # Angle given
    print(i)
    SUBMITCTScreen(Tk())
    # image is loaded with imread command
    image1 = cv2.imread('ct1.jpg')
    # to convert the image in grayscale
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # ///////////////////////////////////////////////////////////
    #####Contrast Limited Adaptive Histogram Equalization.#####
    img = cv2.imread('ct_lc_1.jpg', 0)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    cv2.imwrite('ct_lc_1-1.jpg', cl1)
    cv2.imshow("cl1", cl1)
    # //////////////////////////////////////////////////////////////
    img = cv2.imread('ct_lc_2.jpg', 0)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl2 = clahe.apply(img)
    cv2.imwrite('ct_lc_2-1.jpg', cl2)
    cv2.imshow("cl2", cl2)
    # //////////////////////////////////////////////////////////////
    #######Otsu Threshold Method ######
    # image is loaded with imread command
    image1 = cv2.imread('ct_lc_1-1.jpg')
    # to convert the image in grayscale
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # applying Otsu thresholding
    ret1, thresh1 = cv2.threshold(img1, 0.4619, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # the window showing output image
    # with the corresponding thresholding
    # techniques applied to the input image
    cv2.imshow('Otsu Threshold', thresh1)
    # Normalize and threshold image
    Image.fromarray(thresh1).save("res_1.png")
    # /////////////////////////////////////////////////////////
    # image is loaded with imread command
    image2 = cv2.imread('ct_lc_2-1.jpg')
    # to convert the image in grayscale
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # applying Otsu thresholding
    ret2, thresh2 = cv2.threshold(img2, 0.4619, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # the window showing output image
    # with the corresponding thresholding
    # techniques applied to the input image
    cv2.imshow('Otsu Threshold2', thresh2)
    # Normalize and threshold image
    Image.fromarray(thresh2).save("res_2.png")
    # /////////////////////////////////////////////////////////
    # Load image and greyscale it
    im1 = np.array(Image.open("res_1.png").convert('L'))
    # Normalize and threshold image
    im1 = cv2.normalize(im1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    res11, im1 = cv2.threshold(im1, 64, 255, cv2.THRESH_BINARY)
    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im1, None, (0, 0), 255)
    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im1, None, (0, 0), 0)
    # Save result
    Image.fromarray(im1).save("res_3.png")
    # ////////////////////////////////////////////////////////
    # Load image and greyscale it
    im2 = np.array(Image.open("res_2.png").convert('L'))
    # Normalize and threshold image
    im2 = cv2.normalize(im2, None, alpha=20, beta=255, norm_type=cv2.NORM_MINMAX)
    res22, im2 = cv2.threshold(im2, 64, 255, cv2.THRESH_BINARY)
    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im2, None, (0, 0), 255)
    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im2, None, (0, 0), 0)
    # Save result
    Image.fromarray(im2).save("res_4.png")
    # ////////////////////////////////////////////////////////
    img1 = cv2.imread("res_3.png", cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints_sift, descriptors = orb.detectAndCompute(img1, None)
    img1 = cv2.drawKeypoints(img1, keypoints_sift, None)
    cv2.imshow("Image", img1)
    # ////////////////////////////////////////////////////////
    img2 = cv2.imread("res_4.png", cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints_sift2, descriptors2 = orb.detectAndCompute(img2, None)
    img2 = cv2.drawKeypoints(img2, keypoints_sift2, None)
    cv2.imshow("Image2", img2)
    # ////////////////////////////////////////////////////////
    ########Noise removal using Morphological after using thresholding#########
    # Image operation using thresholding
    img = cv2.imread('res_3.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY +
                                cv2.THRESH_OTSU)
    cv2.imshow('image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Noise removal using Morphological
    # closing operation
    kernel = np.ones((50, 0), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=2)

    # Background area using Dialation
    bg = cv2.dilate(closing, kernel, iterations=1)
    cv2.imshow('image', bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret1, fg = cv2.threshold(dist_transform, 0.02
                             * dist_transform.max(), 255, 0)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret1, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(bg, sure_fg)
    cv2.imshow('image', fg)
    Image.fromarray(fg).save("final_1.tif")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # //////////////////////////////////////////////////////
    # Image operation using thresholding
    img2 = cv2.imread('res_4.png')

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret2, thresh2 = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY +
                                  cv2.THRESH_OTSU)
    cv2.imshow('image2', thresh2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Noise removal using Morphological
    # closing operation
    kernel = np.ones((30, 0), np.uint8)
    closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE,
                               kernel, iterations=2)

    # Background area using Dialation
    bg = cv2.dilate(closing, kernel, iterations=1)
    cv2.imshow('image', bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret2, fg2 = cv2.threshold(dist_transform, 0.02
                              * dist_transform.max(), 255, 0)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unk = cv2.subtract(bg, sure_fg)
    cv2.imshow('image', fg2)
    Image.fromarray(fg2).save("final_2.tif")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # //////////////////////////////////////////////////////
    #######ORB(Oriented FAST and Rotated BRIEF) Detector for feature detection######
    img1 = cv2.imread("res_3.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("res_4.png", cv2.IMREAD_GRAYSCALE)
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    ############### Brute Force Matching Algorithm##############
    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    cv2.imshow("Img1", img1)
    cv2.imshow("Img2", img2)
    Image.fromarray(matching_result).save("Matching result.png")
    cv2.imshow("Matching result", matching_result)
    # /////////////////////////////////////////////////////
    ###########matching counter#########
    from numpy import math
    img = cv2.imread("matching result.png")
    cv2.imshow("Matching result", matching_result)
    edges = cv2.Canny(img, 80, 120)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 2, None, 10, 1)
    n = len(lines)
    print("Matching Result =", float(n), "Matches")
    # ////////////////////////////////////////////////////
    ###########SVM For Classification#########
    cr = pd.read_csv('data.csv')
    cr.head(200)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cr.Clump = le.fit_transform(cr.Clump)
    cr.UnifSize = le.fit_transform(cr.UnifSize)
    cr.UnifShape = le.fit_transform(cr.UnifShape)
    cr.MargAdh = le.fit_transform(cr.MargAdh)
    cr.SingEpiSize = le.fit_transform(cr.SingEpiSize)
    cr.BareNuc = le.fit_transform(cr.BareNuc)
    cr.BlandChrom = le.fit_transform(cr.BlandChrom)
    cr.NormNucl = le.fit_transform(cr.NormNucl)
    cr.Mit = le.fit_transform(cr.Mit)
    cr.head(200)
    cr.describe(include='all')
    X = cr.iloc[:, 1:11].values
    Y = cr.iloc[:, -1].values
    print("Cancer data set dimensions : {}".format(cr.shape))
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    from sklearn import svm

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, Y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    from sklearn import metrics

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    y_pred = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    new_input = [[2.12309797, -1.41131072, 2.1111, 5.22, 8.6, 11.22, 1.11, 6.8, 7.5, 6.22]]
    # get prediction for new input
    new_output = clf.predict(new_input)
    # summarize input and output
    print(new_input, new_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def AddNewPatient(destroyPS):
    destroyPS.destroy()
    addPatientsScreen(Tk())


# Patient Window
class patientsScreen:
    def __init__(self, patients):
        self.window = patients
        self.window.title("CanDet")
        self.fullScreenState = True
        self.window.attributes('-fullscreen', self.fullScreenState)
        self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry("%dx%d+0+0" % (self.w, self.h))
        self.window.configure(bg='#2B2D42')
        self.window.bind("<F11>", self.toggleFullScreen)
        self.window.bind("<Escape>", self.quitFullScreen)
        self.picture = PhotoImage(file="resources\\2user-100.png")
        self.pictureLable = Label(bg="#2B2D42", image=self.picture)
        self.pictureLable.grid(column=0, row=0, ipady=10, sticky="W")
        self.userLable = Label(bg="#2B2D42", text="  Mohammed Alaa \n Doctor", font=("Courier", 20), fg="white")
        self.userLable.grid(column=1, row=0, ipady=10, sticky="W")
        self.window.columnconfigure(2, weight=2)
        self.btnHelp = Button(self.window, width=20, height=2, relief=RAISED, bg="#CC062B", fg="white", text="Help",
                              activebackground="#F2C832", activeforeground="white",
                              command=lambda: helpPopUp())
        self.btnHelp.grid(column=5, row=0, ipady=10, sticky="E")
        self.btnSearch = Button(self.window, width=20, height=2, relief=RAISED, bg="#CC062B", fg="white",
                                activebackground="#F2C832", activeforeground="white",
                                text="Search",
                                command=lambda: searchPatientPopUp())
        self.btnSearch.grid(column=4, row=0, ipady=10, sticky="E")
        self.btnRecord = Button(self.window, width=20, height=2, relief=RAISED, bg="#CC062B", fg="white",
                                activebackground="#F2C832", activeforeground="white",
                                text="Record Patient", command=lambda: AddNewPatient(self.window))
        self.btnRecord.grid(column=3, row=0, ipady=10, sticky="E")
        self.PatientRecordTable = Button(self.window, width=100, height=3, bg="#13131c",
                                         text="Pateint name \t \t Age \t \t Phone \t \t Gender \t \t Scan ",
                                         font=(18), state=DISABLED)
        self.PatientRecordTable.place(x=200, y=200)
        self.PatientRecord = Button(self.window, width=100, height=2, bg="#2B2D42",
                                    text="AhmedAli \t \t 45 \t \t 355698555 \t \t Male \t \t 2020-06-25 ",
                                    font=(15), state=DISABLED)
        self.PatientRecord.place(x=200, y=275)
        self.PatientRecord = Button(self.window, width=100, height=2, bg="#CC062B",
                                    text="Hany Mohamed \t \t 65 \t \t 55555885 \t \t Male \t \t 2020-06-25 ",
                                    font=(15), state=DISABLED)
        self.PatientRecord.place(x=200, y=350)
        self.PatientRecord = Button(self.window, width=100, height=2, bg="#2B2D42",
                                    text="AlyHosam \t \t 68 \t \t 35699996 \t \t Male \t \t 2020-06-25 ",
                                    font=(15), state=DISABLED)
        self.PatientRecord.place(x=200, y=425)
        self.PatientRecord = Button(self.window, width=100, height=2, bg="#CC062B",
                                    text="HanyMatwee \t \t 85 \t \t 566555866 \t \t Male \t \t 2020-06-25 ",
                                    font=(15), state=DISABLED)
        self.PatientRecord.place(x=200, y=500)
        self.PatientRecord = Button(self.window, width=100, height=2, bg="#2B2D42",
                                    text="HanyMatwee \t \t 55 \t \t 255698855 \t \t Female\t \t 2020-06-25 ",
                                    font=(15), state=DISABLED)
        self.PatientRecord.place(x=200, y=575)
        self.PatientRecord = Button(self.window, width=100, height=2, bg="#CC062B",
                                    text="HanyMatwee \t \t 59 \t \t 522555566 \t \t Female \t \t 2020-06-25 ",
                                    font=(15), state=DISABLED)
        self.PatientRecord.place(x=200, y=650)



    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.window.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)





# report screen button function
def back(backtopateintscreen):
    backtopateintscreen.destroy()
    patientsScreen(Tk())


# report-1
class report_1Screen:
    def __init__(self, report):
        self.window = report
        self.window.title("CanDet")
        self.fullScreenState = True
        self.window.attributes('-fullscreen', self.fullScreenState)
        self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry("%dx%d+0+0" % (self.w, self.h))
        self.window.configure(bg='#2B2D42')
        self.window.bind("<F11>", self.toggleFullScreen)
        self.window.bind("<Escape>", self.quitFullScreen)
        # self.window.rowconfigure(2, weight=5)
        self.picture = PhotoImage(file="resources\\2user-100.png")
        self.pictureLable = Label(bg="#2B2D42", image=self.picture)
        self.pictureLable.place(x=0, y=0)
        self.userLable = Label(bg="#2B2D42", text="Mohammad Alaa\n Doctor", font=("Courier", 20), fg="white")
        self.userLable.place(x=self.picture.width() + 10, y=10)
        self.report = PhotoImage(file="resources\\candetReport.png")
        self.reportlabel = Label(image=self.report).place(x=390, y=50)
        backbtn = Button(self.window, fg="white", bg="#CC062B", bd=5, width=30, height=2, activebackground="#F2C832",
                         activeforeground="white", text="Back to Pateints table", relief=FLAT,
                         command=lambda: back(self.window))
        backbtn.place(x=5, y=700)
        email = 1
        url = "https://mail.google.com/"

        def openEmail(): webbrowser.open(url, new=email)

        self.printReport = Button(self.window, fg="white", bg="#CC062B", bd=5, text="Print", relief=FLAT, width=20,
                                  height=2, activebackground="#F2C832", activeforeground="white",
                                  command=print("connect to the printer")).place(x=1200, y=630)
        self.email = Button(self.window, fg="white", bg="#CC062B", bd=5, text="E-mail", relief=FLAT, width=20,
                            height=2, activebackground="#F2C832", activeforeground="white",
                            command=openEmail).place(x=1200, y=700)

    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.window.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)


# submit button function
def gotoReport(submitdestroy):
    submitdestroy.destroy()
    report_1Screen(Tk())


# submit window
class SUBMITCTScreen:
    def __init__(self, CT):
        self.window = CT
        self.window.title("CanDet")
        self.fullScreenState = True
        self.window.attributes('-fullscreen', self.fullScreenState)
        self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry("%dx%d+0+0" % (self.w, self.h))
        self.window.configure(bg='#2B2D42')
        self.window.bind("<F11>", self.toggleFullScreen)
        self.window.bind("<Escape>", self.quitFullScreen)
        # self.window.rowconfigure(2, weight=5)
        self.picture = PhotoImage(file="resources\\2user-100.png")
        self.pictureLable = Label(bg="#2B2D42", image=self.picture)
        self.pictureLable.place(x=0, y=0)
        self.userLable = Label(bg="#2B2D42", text="Mohammad Alaa \n Doctor", font=("Courier", 20), fg="white")
        self.userLable.place(x=self.picture.width() + 10, y=10)
        self.ct = PhotoImage(file="resources\\final.png")
        self.ctLabel = Label(bg="#2B2D42", image=self.ct).place(x=450, y=224)
        ###################################
        ##################### the image after the detection process must be here  ##############################
        #################################################
        self.report = Button(self.window, fg="white", bg="#CC062B", bd=5, text="Generate Report", relief=FLAT, width=20,
                             height=2, activebackground="#F2C832", activeforeground="white",
                             command=lambda: gotoReport(self.window)).place(x=1200, y=700)

    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.window.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)


# Upload Image
def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='"pen')
    return filename


def open_img(window):
    # Select the Imagename from a folder
    x = openfilename()
    # opens the image
    img = Image.open(x)
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((444, 353), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    panel = Label(window, image=img)
    # set the image as img
    panel.image = img
    panel.grid(row=2, column=2, rowspan=10)
   # LungCancerDetection(img)

def SubmitNewPateint(submitDestroy,i, p_name, p_age, p_phone, p_gender, p_ct):
    print(p_name)
    print(p_age)
    print(p_phone)
    print(p_gender)
    print(p_ct)
    patient_1_name = p_name
    patient_1_age = p_age
    patient_1_phone = p_phone
    patient_1_gender = p_gender
    patient_1_ct = p_ct
    submitDestroy.destroy()
    LungCancerDetection(i)



# Add new pateint window
def addpatientdab(addPatientS, Fname, Ag, phn, gender, cttybe):
    today = date.today()
    ag = int(Ag)
    phnNom = int(phn)
    if Fname == "" or ag == "" or phn == "" or gender == "" or cttybe == "":
        print("empty")
    else:
        coursor2.execute("INSERT INTO [dbo].[Patient] VALUES('%s', '%s', '%s', '%s', '%s', '%s')"%(Fname, ag, phnNom, gender, cttybe, today))
        coursor2.commit()
        coursor2.close()
        conn.close()
        print("ok")
        conn.close()

class addPatientsScreen:
    def __init__(self, add):
        self.window = add
        self.window.title("CanDet")
        self.fullScreenState = True
        self.window.attributes('-fullscreen', self.fullScreenState)
        self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry("%dx%d+0+0" % (self.w, self.h))
        self.window.configure(bg='#2B2D42')
        self.window.bind("<F11>", self.toggleFullScreen)
        self.window.bind("<Escape>", self.quitFullScreen)
        # self.window.rowconfigure(2, weight=5)
        self.picture = PhotoImage(file="resources\\2user-100.png")
        self.pictureLable = Label(bg="#2B2D42", image=self.picture)
        self.pictureLable.grid(column=1, row=0, ipady=10, sticky="W")
        self.userLable = Label(bg="#2B2D42", text="Mohammad Alaa\n Doctor", font=("Courier", 20), fg="white")
        self.userLable.grid(column=2, row=0, ipady=10, sticky="W")
        self.fullNameLable = Label(bg="#2B2D42", text="Full Name", font=("Courier", 14), fg="white")
        self.fullNameLable.grid(column=0, row=1, padx=50, sticky="W")
        self.fullNameData = StringVar()
        self.fullName = Entry(self.window, font=("Courier", 15), relief=FLAT, textvariable=self.fullNameData)
        self.fullName.grid(column=0, row=2, padx=50, pady=25)

        self.ageLable = Label(bg="#2B2D42", text="Age", font=("Courier", 14), fg="white")
        self.ageLable.grid(column=0, row=3, padx=50, sticky="W")
        self.ageData = StringVar()
        self.age = Entry(self.window, font=("Courier", 15), relief=FLAT, textvariable=self.ageData)
        self.age.grid(column=0, row=4, padx=50, pady=25)
        self.phoneLable = Label(bg="#2B2D42", text="Phone", font=("Courier", 14), fg="white")
        self.phoneLable.grid(column=0, row=5, padx=50, sticky="W")
        self.phoneData = StringVar()
        self.phone = Entry(self.window, font=("Courier", 15), relief=FLAT, textvariable=self.phoneData)
        self.phone.grid(column=0, row=6, padx=50, pady=25)
        self.genderLable = Label(bg="#2B2D42", text="Gender", font=("Courier", 14), fg="white")
        self.genderLable.grid(column=0, row=7, padx=50, sticky="W")
        self.genderData = StringVar()
        self.gender = ttk.Combobox(self.window, font=("Courier", 15), textvariable=self.genderData)
        self.gender.config(values=("Male", "Female"))
        self.gender.grid(column=0, row=8, padx=50, pady=25)
        self.typeLable = Label(bg="#2B2D42", text="Type Of CT", font=("Courier", 14), fg="white")
        self.typeLable.grid(column=0, row=9, padx=50, sticky="W")
        self.typeData = StringVar()
        self.type = Entry(self.window, font=("Courier", 15), relief=FLAT, textvariable= self.typeData)
        self.type.grid(column=0, row=10, padx=50, pady=25)
        self.ct = PhotoImage(file="original.png")
        self.uploadCT = Button(self.window, fg="white", bg="#CC062B", bd=5, text="Upload CT", relief=FLAT, width=20,
                               height=2, activebackground="#F2C832", activeforeground="white",
                               command=lambda: open_img(self.window)).grid(column=1, row=11, padx=50, pady=25)
        self.submit = Button(self.window, fg="white", bg="#CC062B", bd=5, text="Submit", relief=FLAT, width=20,
                             height=2, activebackground="#F2C832", activeforeground="white",
                             command=lambda: addpatientdab(self.window, self.fullNameData.get(), self.ageData.get(),
                                                           self.phoneData.get(), self.genderData.get(), self.typeData.get())).grid(column=0,row=11, padx=50,pady=25)

    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.window.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)


# Mail Window
# class mailScreen:
#     def __init__(self, mail):
#         self.window = mail
#         self.window.title("CanDet")
#         self.fullScreenState = True
#         self.window.attributes('-fullscreen', self.fullScreenState)
#         self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
#         self.window.geometry("%dx%d+0+0" % (self.w, self.h))
#         self.window.configure(bg='#2B2D42')
#         self.window.bind("<F11>", self.toggleFullScreen)
#         self.window.bind("<Escape>", self.quitFullScreen)
#
#         self.style = ttk.Style()
#         self.style.theme_create("mail", parent="alt", settings={
#             "TNotebook": {"configure": {"tabposition": 'wn', "background": "#CC062B", "tabmargins": [0, 100, 0, 0]}},
#             "TNotebook.Tab": {
#                 "configure": {"padding": [100, self.window.winfo_screenheight() / 16], "background": "#CC062B",
#                               "foreground": 'white', "font": ("Courier", 18, "bold")},
#                 "map": {"background": [("selected", "#CC062B")], }}})
#         self.style.theme_use("mail")
#
#         self.frameStyle = ttk.Style()
#         self.borderImage = PhotoImage(file="resources\\border_note.png")
#         self.frameStyle.element_create("RoundedFrame", "image", self.borderImage, border=16, sticky="nsew")
#         self.frameStyle.layout("RoundedFrame", [("RoundedFrame", {"sticky": "nsew"})])
#
#         self.note = ttk.Notebook(self.window)
#
#         self.inbox_frame = ttk.Frame(self.note, style="RoundedFrame")
#         self.note.add(self.inbox_frame, text='Inbox')
#
#         self.sent_frame = ttk.Frame(self.note, style="RoundedFrame")
#         self.note.add(self.sent_frame, text='Sent ')
#
#         self.draft_frame = ttk.Frame(self.note, style="RoundedFrame")
#         self.note.add(self.draft_frame, text='Draft')
#
#         self.span_frame = ttk.Frame(self.note, style="RoundedFrame")
#         self.note.add(self.span_frame, text='Span ')
#
#         self.note.pack(expand=1, fill='both', padx=5, pady=5)
#
#         self.style.configure('inner.TNotebook', tabposition='nw')
#         self.style.configure('inner.TNotebook', tabmargins=[2, 5, 2, 0])
#         self.style.configure('inner.TNotebook', background="#2B2D42")
#         self.style.configure('inner.TNotebook.Tab', background="#2B2D42")
#         self.style.configure('inner.TNotebook.Tab', padding=[self.window.winfo_screenheight() / 5.5, 20])
#         self.style.map('inner.TNotebook.Tab', background=[("selected", "#2B2D42")])
#
#         self.inner_inbox = ttk.Notebook(self.inbox_frame, style='inner.TNotebook')
#
#         self.inbox_all = ttk.Frame(self.inner_inbox, style="RoundedFrame")
#         self.inner_inbox.add(self.inbox_all, text='All')
#
#         self.inbox_most_recent = ttk.Frame(self.inner_inbox, style="RoundedFrame")
#         self.inner_inbox.add(self.inbox_most_recent, text='Most Recent')
#
#         self.inbox_unread = ttk.Frame(self.inner_inbox, style="RoundedFrame")
#         self.inner_inbox.add(self.inbox_unread, text='Unread')
#
#         self.inner_inbox.pack(expand=1, fill='both', padx=5, pady=5)
#
#         self.sent_inbox = ttk.Notebook(self.sent_frame, style='inner.TNotebook')
#
#         self.sent_all = ttk.Frame(self.sent_inbox, style="RoundedFrame")
#         self.sent_inbox.add(self.sent_all, text='All')
#
#         self.sent_most_recent = ttk.Frame(self.sent_inbox, style="RoundedFrame")
#         self.sent_inbox.add(self.sent_most_recent, text='Most Recent')
#
#         self.sent_unread = ttk.Frame(self.sent_inbox, style="RoundedFrame")
#         self.sent_inbox.add(self.sent_unread, text='Unread')
#
#         self.sent_inbox.pack(expand=1, fill='both', padx=5, pady=5)
#
#         self.draft_inbox = ttk.Notebook(self.draft_frame, style='inner.TNotebook')
#
#         self.draft_all = ttk.Frame(self.draft_inbox, style="RoundedFrame")
#         self.draft_inbox.add(self.draft_all, text='All')
#
#         self.draft_most_recent = ttk.Frame(self.draft_inbox, style="RoundedFrame")
#         self.draft_inbox.add(self.draft_most_recent, text='Most Recent')
#
#         self.draft_unread = ttk.Frame(self.draft_inbox, style="RoundedFrame")
#         self.draft_inbox.add(self.draft_unread, text='Unread')
#
#         self.draft_inbox.pack(expand=1, fill='both', padx=5, pady=5)
#
#         self.span_inbox = ttk.Notebook(self.span_frame, style='inner.TNotebook')
#
#         self.span_all = ttk.Frame(self.span_inbox, style="RoundedFrame")
#         self.span_inbox.add(self.span_all, text='All')
#
#         self.span_most_recent = ttk.Frame(self.span_inbox, style="RoundedFrame")
#         self.span_inbox.add(self.span_most_recent, text='Most Recent')
#
#         self.span_unread = ttk.Frame(self.span_inbox, style="RoundedFrame")
#         self.span_inbox.add(self.span_unread, text='Unread')
#
#         self.span_inbox.pack(expand=1, fill='both', padx=5, pady=5)
#
#     def toggleFullScreen(self, event):
#         self.fullScreenState = not self.fullScreenState
#         self.window.attributes("-fullscreen", self.fullScreenState)
#
#     def quitFullScreen(self, event):
#         self.fullScreenState = False
#         self.window.attributes("-fullscreen", self.fullScreenState)


# Add Placeholder To Entry

class PlaceholderEntry(Entry):
    def __init__(self, container, placeholder, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.placeholder = placeholder
        self.insert("0", self.placeholder)
        self.config(fg="#d5d5d5")
        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._add_placeholder)

    def _clear_placeholder(self, e):
        if self.get() == self.placeholder:
            self.delete("0", "end")

    def _add_placeholder(self, e):
        if not self.get():
            self.insert("0", self.placeholder)


# Splash Screen Class
class SplashScreen:
    def __init__(self, splash):
        self.window = splash
        self.window.title("Welcome To CanDet")
        self.window.attributes('-fullscreen', True)
        self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry("%dx%d+0+0" % (self.w, self.h))
        self.window.configure(bg='#2B2D42')
        self.textLable = Label(text="Welcome To ", fg='#FFFFFF', bg='#2B2D42')
        self.textLable.config(font=("Courier", 44))
        self.textLable.place(x=self.w / 3.25, y=self.h / 3)
        self.logo = PhotoImage(file="resources\logo.png")
        self.logoLable = Label(bg="#2B2D42", image=self.logo)
        self.logoLable.place(x=self.w / 1.8, y=self.h / 3.2)

# sign up buttons functions
def signUPdb(signUpScreen, fname, lname, phn, eml, passw):
    fName= fname
    lName = lname
    phnnum = phn
    Eml=eml
    Passw=passw
    if fname == "" or lname == "" or phn == "" or eml == "" or passw == "":
        print("empty")

    else:
        cursor.execute("INSERT INTO [dbo].[Staff] VALUES('%s', '%s', '%s', '%s', '%s')"%(fName, lName, Eml, Passw, phnnum))
        cursor.commit()
        cursor.close()
        print("ok")
        signUpScreen.destroy()
        emailVerifyPopUp()


def SignUpValidation():
    tl = Toplevel()
    tl.title("Error")
    tl.geometry("500x250+400+150")
    frame = Frame(tl, bg="#2B2D42")
    frame.pack(fill="both")
    canvas = Canvas(frame, width=750, height=250, bg="#2B2D42")
    canvas.pack(fill="both")
    msgbody1 = Label(canvas, text="Error", font=("Times New Roman", 22, "bold"), bg="#2B2D42", fg="#CC062B")
    msgbody1.place(x=200, y=40)
    msgbody2 = Label(canvas, text="The data you entered may be wrong or empty. \n please check and try again."
                                  "again.", font=("Times New Roman", 18, "bold"), bg="#2B2D42", fg="white")
    msgbody2.place(x=10, y=110)


def CallloginFunc(signupS):
    signupS.destroy()
    logInScreen(Tk())


class signUpScreen:
    def __init__(self, signup):
        self.window = signup
        self.window.title("CanDet")
        self.fullScreenState = True
        self.window.attributes('-fullscreen', self.fullScreenState)
        self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry("%dx%d+0+0" % (self.w, self.h))
        self.window.configure(bg='#2B2D42')
        self.window.bind("<F11>", self.toggleFullScreen)
        self.window.bind("<Escape>", self.quitFullScreen)
        self.lable1 = Label(self.window, text="Sign Up!", fg='#FFFFFF', bg='#2B2D42')
        self.lable1.config(font=("Courier", 44))
        self.lable1.place(x=self.w / 10, y=self.h / 3.5)
        self.lable2 = Label(self.window, text="Welcome to our CanDet!", fg='#FFFFFF', bg='#2B2D42')
        self.lable2.config(font=("Courier", 33))
        self.lable2.place(x=self.w / 10, y=self.h / 2)
        self.frameStyle = ttk.Style()
        self.borderImage = PhotoImage(file="resources\\border.png")
        self.frameStyle.element_create("RoundedFrame", "image", self.borderImage, border=16, sticky="nsew")
        self.frameStyle.layout("RoundedFrame", [("RoundedFrame", {"sticky": "nsew"})])
        self.frame = ttk.Frame(self.window, style="RoundedFrame", width=self.w / 2, height=self.h - 100)
        self.frame.place(x=self.w / 2 + 10, y=50)
        self.fw, self.fh = self.frame.winfo_screenwidth(), self.frame.winfo_screenheight()
        self.firstName = PlaceholderEntry(self.frame, "First Name", font=("Courier", 14), relief=FLAT,
                                          background='#DD061B')
        self.firstName.place(x=self.fw / 6 - 10, y=self.fh / 4 - 30)
        self.lastName = PlaceholderEntry(self.frame, "Last Name", font=("Courier", 14), relief=FLAT,
                                         background='#DD061B')
        self.lastName.place(x=self.fw / 6 - 10, y=self.fh / 3 - 30)
        # self.phoneData = StringVar()
        self.phone = PlaceholderEntry(self.frame, "Phone Number", font=("Courier", 14), relief=FLAT,
                                      background='#DD061B')
        self.phone.place(x=self.fw / 6 - 10, y=self.fh / 2.4 - 30)
        # self.emailData = StringVar()
        self.email = PlaceholderEntry(self.frame, "E-Mail", font=("Courier", 14), relief=FLAT,
                                      background='#DD061B')
        self.email.place(x=self.fw / 6 - 10, y=self.fh / 2 - 30)
        # self.passwordData = StringVar()
        self.password = PlaceholderEntry(self.frame, "Password", show="*", font=("Courier", 14), relief=FLAT,
                                         background='#DD061B')
        self.password.place(x=self.fw / 6 - 10, y=self.fh / 1.72 - 30)

        self.signUp = Button(self.frame, fg="black", bg="white", bd=4, text="Sign Up", relief=FLAT, width=40,
                             height=1, activebackground="#F2C832", activeforeground="white",
                             command=lambda: signUPdb(self.window, self.firstName.get(), self.lastName.get(), self.phone.get(), self.email.get(), self.password.get())).place(x=self.fw / 6 - 10,
                                                                                    y=self.fh / 0.5 - 30)
        self.logIn = Button(self.frame, fg="white", bg="#1B2D42", bd=5, text="Log In ?", relief=FLAT, width=10,
                            height=1, activebackground="#F2C832", activeforeground="white",
                            command=lambda: CallloginFunc(self.window)).place(x=self.fw / 1.3, y=self.fh / 1.22)

    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.window.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)


# login buttons functions
def logindatabase(logInScreen, uname, passw):
    if uname == "" or passw == "":
        print("empty")
    else:
        cursor.execute("SELECT Email,Paswword From Staff")
        for (Email,Paswword) in cursor:
            if uname == Email and passw == Paswword:
                login = True
                logInScreen.destroy()
                patientsScreen(Tk())
                break
            else:
                login = False
                logInWarnPopUp()


    cursor.close()







def logInWarnPopUp():
    tl = Toplevel()
    tl.title("Error")
    tl.geometry("575x250+500+150")
    frame = Frame(tl, bg="#2B2D42")
    frame.pack(fill="both")
    canvas = Canvas(frame, width=575, height=250, bg="#2B2D42")
    canvas.pack(fill="both")
    msgbody1 = Label(canvas, text="Login Error", font=("Times New Roman", 22, "bold"), bg="#2B2D42", fg="#CC062B")
    msgbody1.place(x=200, y=40)
    msgbody2 = Label(canvas, text="The Email or password you entered didn't match our \n records. please check and try "
                                  "again.", font=("Times New Roman", 18, "bold"), bg="#2B2D42", fg="white")
    msgbody2.place(x=10, y=110)


def CallsignUpFunc(logInScreen):
    logInScreen.destroy()
    signUpScreen(Tk())


# Log In Window
class logInScreen:
    def __init__(self, login):
        self.window = login
        self.window.title("CanDet")
        self.fullScreenState = True
        self.window.attributes('-fullscreen', self.fullScreenState)
        self.w, self.h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry("%dx%d+0+0" % (self.w, self.h))
        self.window.configure(bg='#2B2D42')
        self.window.bind("<F11>", self.toggleFullScreen)
        self.window.bind("<Escape>", self.quitFullScreen)
        self.lable1 = Label(self.window, text="Welcome!", fg='#FFFFFF', bg='#2B2D42')
        self.lable1.config(font=("Courier", 44))
        self.lable1.place(x=self.w / 10, y=self.h / 3.5)
        self.frameStyle = ttk.Style()
        self.borderImage = PhotoImage(file="resources\\border.png")
        self.frameStyle.element_create("RoundedFrame", "image", self.borderImage, border=16, sticky="nsew")
        self.frameStyle.layout("RoundedFrame", [("RoundedFrame", {"sticky": "nsew"})])
        self.frame = ttk.Frame(self.window, style="RoundedFrame", width=self.w / 2, height=self.h - 100)
        self.frame.place(x=self.w / 2 + 10, y=50)
        self.fw, self.fh = self.frame.winfo_screenwidth(), self.frame.winfo_screenheight()
        self.logLable = Label(self.frame, text="Log In", fg='white', bg='#CC062B')
        self.logLable.config(font=("Courier", 22))
        self.logLable.place(x=self.fw / 5, y=self.fh / 3 - 200)
        self.logo = PhotoImage(file="resources\\2user-100.png")
        self.logoLable = Label(self.frame, bg="#CC062B", image=self.logo, width=100, height=100)
        self.logoLable.place(x=self.fw / 5, y=self.fh / 3 - 100)
        # self.usernameData = StringVar()
        self.username = PlaceholderEntry(self.frame, "Enter Username", font=("Courier", 19), relief=FLAT,
                                         background='#DD062B')
        self.username.place(x=self.fw / 7 - 10, y=self.fh / 2 - 30)
        # self.passwordData = StringVar()
        self.password = PlaceholderEntry(self.frame, "Enter Password", show="*", font=("Courier", 19), relief=FLAT,
                                         background='#DD062B')
        self.password.place(x=self.fw / 7 - 10, y=self.fh / 1.72 - 30)
        self.logIn = Button(self.frame, fg="black", bg="white", bd=5, text="Log In", relief=FLAT, width=40,
                            height=2, activebackground="#F2C832", activeforeground="white"
                            , command=lambda: logindatabase(self.window, self.username.get(), self.password.get())).place(
            x=self.fw / 7 - 10, y=self.fh / 1.5 - 30)
        self.forgotPassword = Button(self.frame, fg="black", bg="#CC062B", text="Forget Password ?", relief=FLAT,
                                     width=15, height=1, activebackground="#F2C832", activeforeground="white",
                                     command=forgetPasswordWarnPopUp).place(x=self.fw / 7 - 10, y=self.fh / 1.4 - 20)
        self.signUp = Button(self.frame, fg="white", bg="#2B2D42", bd=5, text="Sign Up ?", relief=FLAT, width=10,
                             height=2, activebackground="#F2C832", activeforeground="white",
                             command=lambda: CallsignUpFunc(self.window)).place(x=self.fw / 2.3, y=self.fh / 1.22)

    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.window.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)


# Change Password
# def changePasswordHintPopUp():
#     tl = Toplevel()
#     tl.title("HINT !")
#     tl.geometry("300x300+900+250")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=400, height=450, bg="#2B2D42")
#     canvas.pack(fill="both")
#     msgbody1 = Label(canvas, text="Password Must Contain :", font=("Times New Roman", 14, "bold"), bg="#2B2D42",
#                      fg="#CC062B")
#     msgbody1.place(x=50, y=30)
#     msgbody1 = Label(canvas, text="At least 1 upper case letter :", font=("Times New Roman", 12), bg="#2B2D42",
#                      fg="white")
#     msgbody1.place(x=50, y=100)
#     msgbody1 = Label(canvas, text="At least 1 number (0-9) :", font=("Times New Roman", 12), bg="#2B2D42", fg="white")
#     msgbody1.place(x=50, y=150)
#     msgbody1 = Label(canvas, text="At least 8 character :", font=("Times New Roman", 12), bg="#2B2D42", fg="white")
#     msgbody1.place(x=50, y=200)
#
#
# Password Changed Messages

# def passwordChangedPopUp():
#     tl = Toplevel()
#     tl.title("Success")
#     tl.geometry("400x400+550+100")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=400, height=600, bg="#2B2D42")
#     canvas.pack(fill="both")
#     imgvar = PhotoImage(file="resources\\mark.png")
#     canvas.create_image(200, 100, image=imgvar)
#     canvas.image = imgvar
#     msgbody1 = Label(canvas, text="Password Changed Successfully", font=("Times New Roman", 20, "bold"), bg="#2B2D42",
#                      fg="white")
#     msgbody1.place(x=10, y=imgvar.height() + 50)
#     okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="OK",
#                     activebackground="#F2C832", activeforeground="white",
#                     command=lambda: print("CLICK"))
#     okbttn.place(x=125, y=imgvar.height() + 130)
#
#
# Change Password

# def changePasswordPopUp():
#     tl = Toplevel()
#     tl.title("Change Password")
#     tl.geometry("400x450+550+100")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=400, height=450, bg="#2B2D42")
#     canvas.pack(fill="both")
#     msgbody1 = Label(canvas, text="Change Password", font=("Times New Roman", 18, "bold"), bg="#2B2D42", fg="#CC062B")
#     msgbody1.place(x=90, y=50)
#     currentPassword = PlaceholderEntry(frame, "Current Password", show="*", font=("Courier", 15), relief=FLAT,
#                                        background='#2B2D35').place(x=90, y=150)
#     newPassword = PlaceholderEntry(frame, "New Password", show="*", font=("Courier", 15), relief=FLAT,
#                                    background='#2B2D35').place(x=90, y=225)
#     confirmPassword = PlaceholderEntry(frame, "Confirm Password", show="*", font=("Courier", 15), relief=FLAT,
#                                        background='#2B2D35').place(x=90, y=300)
#     okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text=" Change",
#                     activebackground="#F2C832", activeforeground="white",
#                     command=lambda: print("CLICK"))
#     okbttn.place(x=125, y=300 + 50)
#
#
# Recovery Digit Messages

# def recoveryDigitsPopUp():
#     tl = Toplevel()
#     tl.title("Recovery")
#     tl.geometry("575x300+500+150")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=575, height=300, bg="#2B2D42")
#     canvas.pack(fill="both")
#     # imgvar = PhotoImage(file="resources\\mark.png")
#     # canvas.create_image(200, 100, image=imgvar)
#     # canvas.image = imgvar
#     msgbody1 = Label(canvas, text="Enter 6 digit recovery code", font=("Times New Roman", 22, "bold"), bg="#2B2D42",
#                      fg="white")
#     msgbody1.place(x=130, y=30)
#     # recoveryData = StringVar()
#     recovery = PlaceholderEntry(canvas, "Press your Email", show="*", font=("Courier", 19), relief=FLAT,
#                                 background='#2B2D35').place(x=150, y=130)
#     okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="Enter",
#                     activebackground="#F2C832", activeforeground="white",
#                     command=lambda: print("CLICK"))
#     okbttn.place(x=200, y=200)


# Forget Password Warning Messages
#


def forgetPasswordWarnPopUp():
    tl = Toplevel()
    tl.title("Forget Password ")
    tl.geometry("575x400+500+150")
    frame = Frame(tl, bg="#2B2D42")
    frame.pack(fill="both")
    canvas = Canvas(frame, width=575, height=400, bg="#2B2D42")
    canvas.pack(fill="both")
    msgbody1 = Label(canvas, text="Forget your password", font=("Times New Roman", 22, "bold"), bg="#2B2D42",
                     fg="#CC062B")
    msgbody1.place(x=150, y=40)
    msgbody2 = Label(canvas, text="Don't worry Resting your password is easy.\ntype in Email you registered with."
                                  "again.", font=("Times New Roman", 18, "bold"), bg="#2B2D42", fg="white")
    msgbody2.place(x=70, y=110)
    mail = PlaceholderEntry(canvas, "Enter your E-mail", font=("Courier", 19), relief=FLAT,
                            background='#2B2D42')
    mail.place(x=150, y=200)

    def pwreset(): okbttn.configure(text="Reset e-mail is sent!")

    okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="Send",
                    activebackground="#F2C832", activeforeground="white",
                    command=pwreset)
    okbttn.place(x=200, y=230 + 50)


# Record Data Warning Messages
# def addNewWarnPopUp():
#     tl = Toplevel()
#     tl.title("Error")
#     tl.geometry("750x250+400+150")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=750, height=250, bg="#2B2D42")
#     canvas.pack(fill="both")
#     # imgvar = PhotoImage(file="resources\\mark.png")
#     # canvas.create_image(200, 100, image=imgvar)
#     # canvas.image = imgvar
#     msgbody1 = Label(canvas, text="Error", font=("Times New Roman", 22, "bold"), bg="#2B2D42", fg="#CC062B")
#     msgbody1.place(x=300, y=40)
#     msgbody2 = Label(canvas, text="The data you entered may be wrong. please check and try again."
#                                   "again.", font=("Times New Roman", 18, "bold"), bg="#2B2D42", fg="white")
#     msgbody2.place(x=10, y=110)
#     # lang = Label(frame, text="Few steps to start using \n CanDet sysyem,you need to \n confirm your email address",
#     #              font=("Times New Roman", 20), bg="#2B2D42", fg="white")
#     # lang.place(x=50, y=imgvar.height() + 100 + 100)
#     okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="Ok",
#                     activebackground="#F2C832", activeforeground="white",
#                     command=lambda: print("CLICK"))
#     okbttn.place(x=300, y=130 + 50)
#

# Log In Warning Messages

# def logInWarnPopUp():
#     tl = Toplevel()
#     tl.title("Error")
#     tl.geometry("575x250+500+150")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=575, height=250, bg="#2B2D42")
#     canvas.pack(fill="both")
#     # imgvar = PhotoImage(file="resources\\mark.png")
#     # canvas.create_image(200, 100, image=imgvar)
#     # canvas.image = imgvar
#     msgbody1 = Label(canvas, text="Login Error", font=("Times New Roman", 22, "bold"), bg="#2B2D42", fg="#CC062B")
#     msgbody1.place(x=200, y=40)
#     msgbody2 = Label(canvas, text="The Email or password you entered didn't match our \n records. please check and try "
#                                   "again.", font=("Times New Roman", 18, "bold"), bg="#2B2D42", fg="white")
#     msgbody2.place(x=10, y=110)
#     # lang = Label(frame, text="Few steps to start using \n CanDet sysyem,you need to \n confirm your email address",
#     #              font=("Times New Roman", 20), bg="#2B2D42", fg="white")
#     # lang.place(x=50, y=imgvar.height() + 100 + 100)
#     okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="Ok",
#                     activebackground="#F2C832", activeforeground="white",
#                     command=lambda: print("CLICK"))
#     okbttn.place(x=200, y=130 + 50)


# def SignUpWarnPopUp():
#     tl = Toplevel()
#     tl.title("Error")
#     tl.geometry("575x250+500+150")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=575, height=250, bg="#2B2D42")
#     canvas.pack(fill="both")
#     # imgvar = PhotoImage(file="resources\\mark.png")
#     # canvas.create_image(200, 100, image=imgvar)
#     # canvas.image = imgvar
#     msgbody1 = Label(canvas, text="Sign up Error", font=("Times New Roman", 22, "bold"), bg="#2B2D42", fg="#CC062B")
#     msgbody1.place(x=200, y=40)
#     msgbody2 = Label(canvas, text="All fields need to be filled", font=("Times New Roman", 18, "bold"), bg="#2B2D42", fg="white")
#     msgbody2.place(x=10, y=110)
#     # lang = Label(frame, text="Few steps to start using \n CanDet sysyem,you need to \n confirm your email address",
#     #              font=("Times New Roman", 20), bg="#2B2D42", fg="white")
#     # lang.place(x=50, y=imgvar.height() + 100 + 100)
#     okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="Ok",
#                     activebackground="#F2C832", activeforeground="white",
#                     command=lambda : tl.destroy())
#     okbttn.place(x=200, y=130 + 50)
#
#
# E-Mail Warning Messages

# def emailWarnPopUp():
#     tl = Toplevel()
#     tl.title("Validate")
#     tl.geometry("400x250+550+100")
#     frame = Frame(tl, bg="#2B2D42")
#     frame.pack(fill="both")
#     canvas = Canvas(frame, width=400, height=250, bg="#2B2D42")
#     canvas.pack(fill="both")
#     # imgvar = PhotoImage(file="resources\\mark.png")
#     # canvas.create_image(200, 100, image=imgvar)
#     # canvas.image = imgvar
#     msgbody1 = Label(canvas, text="Please Enter a valid E-MAIL", font=("Times New Roman", 18, "bold"), bg="#2B2D42",
#                      fg="white")
#     msgbody1.place(x=50, y=90)
#     # lang = Label(frame, text="Few steps to start using \n CanDet sysyem,you need to \n confirm your email address",
#     #              font=("Times New Roman", 20), bg="#2B2D42", fg="white")
#     # lang.place(x=50, y=imgvar.height() + 100 + 100)
#     okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="Ok",
#                     activebackground="#F2C832", activeforeground="white",
#                     command=lambda: print("CLICK"))
#     okbttn.place(x=125, y=100 + 50)
#
#
# E-Mail Verify Messages


# Help Screen
def helpPopUp():
    tl = Toplevel()
    tl.title("Help")
    tl.geometry("500x200+500+150")
    frame = Frame(tl, bg="#2B2D42")
    frame.pack(fill="both")
    canvas = Canvas(frame, width=575, height=250, bg="#2B2D42")
    canvas.pack(fill="both")
    msgbody1 = Label(canvas, text="Press on the Record Patient button to add new patient data",
                     font=("Times New Roman", 14), bg="#2B2D42", fg="white")
    msgbody1.place(x=8, y=40)
    msgbody2 = Label(canvas, text="Press on the Search button to reach privious patients data again"
                     , font=("Times New Roman", 14), bg="#2B2D42", fg="white")
    msgbody2.place(x=8, y=110)


# Search Patient
def searchPatientPopUp():
    tl = Toplevel()
    tl.title("Search")
    tl.geometry("300x350+550+100")
    frame = Frame(tl, bg="#2B2D42")
    frame.pack(fill="both")
    canvas = Canvas(frame, width=400, height=450, bg="#2B2D42")
    canvas.pack(fill="both")
    msgbody = Label(canvas, text="-Enter the patient's name:", font=("Times New Roman", 14), bg="#2B2D42",
                    fg="white")
    msgbody.place(x=40, y=50)
    searchData = PlaceholderEntry(frame, "patient's name", font=("Courier", 15), relief=FLAT, background='#2B2D42')
    searchData.place(x=65, y=150)

    def search(): okbttn.configure(text="No such a name!")

    okbttn = Button(frame, width=20, height=2, relief=FLAT, bg="#CC062B", fg="white", text="Go!",
                    activebackground="#F2C832", activeforeground="white", command=search)
    okbttn.place(x=82, y=250)


def emailVerifyPopUp():
    tl = Toplevel()
    tl.title("Verify")
    tl.geometry("400x600+550+100")
    frame = Frame(tl, bg="#2B2D42")
    frame.pack(fill="both")
    canvas = Canvas(frame, width=400, height=600, bg="#2B2D42")
    canvas.pack(fill="both")
    imgvar = PhotoImage(file="resources\\mark.png")
    canvas.create_image(200, 100, image=imgvar)
    canvas.image = imgvar
    msgbody1 = Label(canvas, text="Verify your Email", font=("Times New Roman", 20, "bold"), bg="#2B2D42", fg="#CC062B")
    msgbody1.place(x=100, y=imgvar.height() + 100)
    lang = Label(frame, text="Few steps to start using \n CanDet system,you need to \n confirm your email address",
                 font=("Times New Roman", 20), bg="#2B2D42", fg="white")
    lang.place(x=50, y=imgvar.height() + 100 + 100)


def call_mainroot():
    splashWindow.window.destroy()
    # mailScreen(Tk())
    # signUpScreen(Tk())
    # logInScreen(Tk())
    # report_1Screen(Tk())
    # SUBMITCTScreen(Tk())
    # addNewWarnPopUp()
    # helpPopUp()
    # searchPatientPopUp()
    # patientsScreen(Tk())
    addPatientsScreen(Tk())
    # changePasswordHintPopUp()
    # passwordChangedPopUp()
    # changePasswordPopUp()
    # recoveryDigitsPopUp()
    # forgetPasswordWarnPopUp()
    # logInWarnPopUp()
    # emailWarnPopUp()
    # emailVerifyPopUp()


# Programme
splashWindow = SplashScreen(Tk())
splashWindow.window.after(1, call_mainroot)
splashWindow.window.mainloop()
