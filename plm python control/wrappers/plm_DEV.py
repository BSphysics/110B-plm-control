
import os
scriptDir = os.getcwd()
import sys
sys.path.append(os.path.join(scriptDir,"plm python functions" ))
sys.path.append(os.path.join(scriptDir,"basler python functions" ))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.close('all')
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QPushButton, QLineEdit, QLabel, QFileDialog
)

from PyQt5.QtCore import QTimer, QSettings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')

from generatePhaseTilt import generate_phase_tilt
from HGMode import HG_mode
from ampModPhase import amp_mod_phase
from superPixelSet import super_pixel_set
from superPixelSet_numba import super_pixel_set_numba
from superPixelSetInit import super_pixel_set_init

from superPixelFrames import super_pixel_frames
from saveSuperPixelImages import save_super_pixel_images
from wavefrontCorrection import wavefront_correction
from phaseScanningFrameGenerator import phase_scanning_frame_generator
from ampRampFrameGenerator import amp_ramp_frame_generator
from phaseScanAnalysor import phase_scan_analysor
from grab50Images import grab_50_images
from tiltMapping import tilt_mapping
from overlapOptimiser import overlap_optimiser
from slider import slider

from applyDarkTheme import apply_dark_theme
from applyDarkPlotTheme import apply_dark_plot_theme

from polMeasure import pol_measure
from loadMultibeamData import load_multibeam_data

import ctypes
from PLMController import PLMController 
import time

import nidaqmx

#----------------------------------Basler camera config
from pypylon import pylon
from pypylon import genicam
import cv2
import matplotlib.patches as patches
import collections
from camera_config import camConfig
from basler_centroid import baslerCentroid

#------------ ----------------------Open the camera
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()

if len(devices) == 0:
    print("No camera found!")
    exit(1)

camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
camera.Open()
print("Found Basler ", camera.GetDeviceInfo().GetModelName(), 'camera')

#----------------------------------Config ROI parameters
exposureTime = 2000
timing_data=[]
#-------------------------------------------------------
# camConfig(camera, image_height, image_width, int(offset_x-image_width/2),int(offset_y - image_height/2), exposureTime)
camConfig(camera, 512, 512, 0, 0, exposureTime)
img = np.zeros((512 , 512))
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


cols, rows = 1358 , 800

default_grating_periods = np.array([10,12,30,12])

plm_hologram = []
superPixelFrames = []
imgs = []

boxNames = ['Beam A x grating period', 'Beam A y grating period', 'Beam B x grating period', 'Beam B y grating period' ]
buttonNames = ['Beam A x tilt off', 'Beam A y tilt off', 'Beam B x tilt off', 'Beam B y tilt off' ]
HGboxNames = ['Beam A HG mode (n, m)', 'Beam B HG mode (n,m)']
beamWaistboxNames = ['Beam A waist (x, y)', 'Beam B waist (x,y)']
beamCentreboxNames = ['Beam A centre (x, y)', 'Beam B centre (x,y)']

frames = 70
N = 1358
M = 800

fullpath = r'D:\PLM\plm python control\bin\plmctrl.dll'
numHolograms = 24

# Create PLMController instance
plm = PLMController(frames, cols, rows, fullpath)
# Start the UI
monitor_id = 1
plm.start_ui(monitor_id)

phase_levels = np.array([0.004, 0.017, 0.036, 0.058, 0.085, 0.117, 0.157, 0.217, 0.296, 0.4, 0.5, 0.605, 0.713, 0.82, 0.922, 0.981, 1], dtype=np.float32);
plm.set_lookup_table(np.asfortranarray(phase_levels));

import itertools
arrangements = list(itertools.product([0, 1], repeat=4))
array = np.array(arrangements)
v = np.array([14, 1, 10, 6, 2, 15, 11, 7, 3, 16, 12, 8, 4, 13, 9, 5])-1
# v = np.array([13, 9, 5, 14, 1, 10, 4, 6, 2, 15, 11, 7, 3, 16, 12, 8])-1 # from Jose

phase_map_order = np.fliplr(array[v,:])
plm.set_phase_map((phase_map_order)) 

time.sleep(1)
plm.play()
plm.play()
class InteractiveGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("plm_GUI")
        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feed)
        self.timer.start(100)  # Update every 100ms
        self.camera = camera

    def init_ui(self):
        # apply_dark_theme(self)
        self.camera = camera
        self.zoom_enabled = False  # Tracks zoom state for phase map before it is sent to plm
        self.bitpack_enabled = False # Don't bitpack until ready
        self.img_zoom_enabled = False  # Flag to track whether zoom is active or not
        self.grab_centroid_enabled = False #Only get centroid when button is pressed
        self.hardware_triggering_enabled = False # Default Camera triggering mode is free streaming 
        self.centroid_x = None  # Store x-coordinate of centroid
        self.centroid_y = None  # Store y-coordinate of centroid
        self.zoom_counter = 0 
        self.beam_A_SP_scan_enabled= False
        self.beam_B_SP_scan_enabled= False

        self.clear_beam_A_correction_flag = False
        self.clear_beam_B_correction_flag = False

        self.beam_A_complex_field_measurement_enabled = False 
        self.beam_B_complex_field_measurement_enabled = False

        self.phase_scan_frame_flag = False
        self.polarisation_measurement_flag = False
        self.amp_ramp_frame_flag = False
        self.grab_50_flag = False
        self.task = nidaqmx.Task()
        self.task.do_channels.add_do_chan("Dev1/port0/line0")    
        
        self.xdata = collections.deque(maxlen=200)  
        self.ydata = collections.deque(maxlen=200)

        self.multibeam_flag = False
        self.tilt_mapping_flag = False
        self.overlap_optimiser_flag = False
        self.multibeam_seq_flag = False
        self.slider_flag = False

        
        """Set up the GUI layout and widgets."""
        self.main_layout = QHBoxLayout()
        
        # ======= LEFT COLUMN: Buttons & Inputs =======
        self.control_layout = QVBoxLayout()
        self.control_layout.setSpacing(2)  # Reduce spacing between widgets
        self.control_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # ======= MIDDLE COLUMN (new buttons) =======
        self.middle_layout = QVBoxLayout()
        self.middle_layout.setSpacing(2)  # Reduce spacing between widgets
        self.middle_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        form_layout = QFormLayout()
        form_layout.setSpacing(2)
        
        self.setGeometry(200, 100, 1200, 800)  

        self.labels = []
        self.inputs = []
        self.buttons = []

        self.user_values = [0] * 20 
        self.button_states = [False] * 4  
        
        self.beam_A_correction_data = np.zeros((rows, cols))  # Store beam A correction here
        self.beam_B_correction_data = np.zeros((rows, cols))  # Store beam B correction here

        for i in range(4):
            label = QLabel(boxNames[i])
            input_box = QLineEdit(self)
            input_box.setText(str(default_grating_periods[i]))  
            input_box.setMaximumWidth(50)
            form_layout.addRow(label, input_box)
            self.inputs.append(input_box)

        self.control_layout.addLayout(form_layout)  # Add input fields to left column           

        # Tilt ON/OFF buttons        
        for i in range(4):
            button = QPushButton(buttonNames[i])
            button.setCheckable(True)  # Makes the button toggleable
            button.clicked.connect(lambda _, idx=i: self.toggle_button(idx))
            self.control_layout.addWidget(button)
            self.buttons.append(button)

       
        # Load Beam A phase correction pattern button
        self.beamA_correction_button = QPushButton('Load Beam A phase correction', self)
        self.beamA_correction_button.clicked.connect(self.beam_A_correction)
        self.control_layout.addWidget(self.beamA_correction_button)   
       
        # Load Beam B phase correction pattern button
        self.beamB_correction_button = QPushButton('Load Beam B phase correction', self)
        self.beamB_correction_button.clicked.connect(self.beam_B_correction)
        self.control_layout.addWidget(self.beamB_correction_button)
        
        # HG mode selection input boxes
        form_layout2 = QFormLayout()
        for i in range(2):
            label = QLabel(HGboxNames[i])
            hbox = QHBoxLayout()
            
            input_box1 = QLineEdit(self)
            input_box1.setText("0")
            input_box1.setMaximumWidth(30)

            input_box2 = QLineEdit(self)
            input_box2.setText("0")
            input_box2.setMaximumWidth(30)

            hbox.addWidget(input_box1)
            hbox.addWidget(input_box2)
            self.inputs.append(input_box1)
            self.inputs.append(input_box2)
            
            form_layout2.addRow(label, hbox)

        self.control_layout.addLayout(form_layout2)           
        
        # Beam waist input boxes
        form_layout3 = QFormLayout()
        for i in range(2):
            label = QLabel(beamWaistboxNames[i])

            hbox = QHBoxLayout()
            input_box1 = QLineEdit(self)
            input_box1.setText("900")
            input_box1.setMaximumWidth(50)
            
            input_box2 = QLineEdit(self)
            input_box2.setText("900")
            input_box2.setMaximumWidth(50)
            
            hbox.addWidget(input_box1)
            hbox.addWidget(input_box2)
            self.inputs.append(input_box1)
            self.inputs.append(input_box2)
        
            form_layout3.addRow(label, hbox)

        self.control_layout.addLayout(form_layout3)
        
        # Beam centre input boxes
        form_layout4 = QFormLayout()
        for i in range(2):
            label = QLabel(beamCentreboxNames[i])

            hbox = QHBoxLayout()
            input_box1 = QLineEdit(self)
            input_box1.setText("0")
            input_box1.setMaximumWidth(50)
            
            input_box2 = QLineEdit(self)
            input_box2.setText("0")
            input_box2.setMaximumWidth(50)
            
            hbox.addWidget(input_box1)
            hbox.addWidget(input_box2)
            self.inputs.append(input_box1)
            self.inputs.append(input_box2)
        
            form_layout3.addRow(label, hbox)

        self.control_layout.addLayout(form_layout4)
        
        # Amplitude and phase input boxes
        form_layout5 = QFormLayout()
        beam_settings = [
            ("Beam A Amplitude", "1"),
            ("Beam B Amplitude", "1"),
            ("Beam A global phase (%)", "50"),
            ("Beam B global phase (%)", "0"),
        ]

        for label_text, default_val in beam_settings:
            label = QLabel(label_text)
            input_box = QLineEdit(self)
            input_box.setText(default_val)
            input_box.setMaximumWidth(50)
            self.inputs.append(input_box)
            # self.inputs.append(input_box2)
            form_layout5.addRow(label, input_box)

        self.control_layout.addLayout(form_layout5)
                                
        # Bitpack button
        self.bitpack_button = QPushButton('Bitpack frames', self)
        self.bitpack_button.setStyleSheet("""
            QPushButton {
                background-color: #800080;  /* Purple */
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #9932CC;  /* Lighter purple on hover */
            }
        """)
        self.bitpack_button.clicked.connect(self.bitpack_frames)
        self.control_layout.addWidget(self.bitpack_button)
        
        # Update button
        self.button = QPushButton('Update', self)
        self.button.clicked.connect(self.update_value)
        self.control_layout.addWidget(self.button)
        
        # Zoom phase map toggle button
        self.zoom_button = QPushButton('Phase map Zoom', self)
        self.zoom_button.setCheckable(True)
        self.zoom_button.clicked.connect(self.toggle_zoom)
        self.control_layout.addWidget(self.zoom_button)
        
        # Find centroid of image button 
        self.centroid_button = QPushButton('Grab Centroid', self)
        self.centroid_button.clicked.connect(self.toggle_centroid)
        self.centroid_button.clicked.connect(self.centroid_button.setDisabled)
        self.control_layout.addWidget(self.centroid_button)
        
        # Zoom live image toggle button
        self.img_zoom_button = QPushButton('Image Zoom', self)
        self.img_zoom_button.setCheckable(True)
        self.img_zoom_button.clicked.connect(self.img_toggle_zoom)
        self.control_layout.addWidget(self.img_zoom_button)
            
        # Toggle camera Trigger mode
        self.camera_trigger_mode_button = QPushButton('Enable Hardware trigger', self)
        self.camera_trigger_mode_button.setCheckable(True)
        self.camera_trigger_mode_button.clicked.connect(self.camera_trigger_mode)
        self.control_layout.addWidget(self.camera_trigger_mode_button)

        # Beam A super pixel scan button
        self.beam_A_super_pixel_scan_button = QPushButton('Beam A super pixel scan', self)        
        self.beam_A_super_pixel_scan_button.clicked.connect(self.beam_A_SP_scan)
        self.control_layout.addWidget(self.beam_A_super_pixel_scan_button)

        # Beam B super pixel scan button
        self.beam_B_super_pixel_scan_button = QPushButton('Beam B super pixel scan', self)        
        self.beam_B_super_pixel_scan_button.clicked.connect(self.beam_B_SP_scan)
        self.control_layout.addWidget(self.beam_B_super_pixel_scan_button)

        self.control_layout.addStretch()

        # ===========MIDDLE COLUMN ==========

         # Generate plm frame (24 phase maps) that scan global phase and record images using HW triggering button
        self.global_phase_scan_button = QPushButton('Global phase scan frame', self)
        self.global_phase_scan_button.clicked.connect(self.global_phase_scan)
        self.middle_layout.addWidget(self.global_phase_scan_button)

        # Ramp the amplitudes and flip the phase by pi halfway through - gives rotating lobes when using HG10+HG01 beams and linear polariser
        self.amp_ramp_button = QPushButton('Amplitude ramp frame' , self)
        self.amp_ramp_button.clicked.connect(self.amplitude_ramp_scan)
        self.middle_layout.addWidget(self.amp_ramp_button)

         # Clear Beam A phase correction pattern button
        self.clear_beamA_correction_button = QPushButton('Clear Beam A phase correction', self)
        self.clear_beamA_correction_button.clicked.connect(self.clear_beam_A_correction)
        self.middle_layout.addWidget(self.clear_beamA_correction_button)   

        # Clear Beam B phase correction pattern button
        self.clear_beamB_correction_button = QPushButton('Clear Beam B phase correction', self)
        self.clear_beamB_correction_button.clicked.connect(self.clear_beam_B_correction)
        self.middle_layout.addWidget(self.clear_beamB_correction_button)  

        # Beam A complex field measurement
        self.beamA_complex_field_button = QPushButton('Beam A complex field measurement', self)
        self.beamA_complex_field_button.clicked.connect(self.beam_A_complex_measurement)
        self.middle_layout.addWidget(self.beamA_complex_field_button)  

        # Beam B complex field measurement
        self.beamB_complex_field_button = QPushButton('Beam B complex field measurement', self)
        self.beamB_complex_field_button.clicked.connect(self.beam_B_complex_measurement)
        self.middle_layout.addWidget(self.beamB_complex_field_button) 

        # Polarisation measurement button
        self.polarisation_measurement_button = QPushButton('Polarisation measurement', self)
        self.polarisation_measurement_button.clicked.connect(self.polarisation_measurement)
        self.middle_layout.addWidget(self.polarisation_measurement_button)   

        # Grab 50 images button
        self.grab_50_button = QPushButton('Grab 50 images', self)
        self.grab_50_button.clicked.connect(self.grab_50)
        self.middle_layout.addWidget(self.grab_50_button)  

        # Load multibeam data button
        self.multibeam_button = QPushButton('Load multi-beam parameters', self)
        self.multibeam_button.setStyleSheet("background-color: green; color: white;")
        self.multibeam_button.clicked.connect(self.multibeam)
        self.middle_layout.addWidget(self.multibeam_button)  

        # Run multibeam sequence button
        self.multibeam_seq_button = QPushButton('Run multibeam sequence', self)
        self.multibeam_seq_button.setStyleSheet("background-color: orange; color: white;")
        self.multibeam_seq_button.clicked.connect(self.multibeam_seq)
        self.middle_layout.addWidget(self.multibeam_seq_button) 

        # Run tilt mapping sequence
        self.tilt_map_button = QPushButton('Run tilt map sequence', self)
        self.tilt_map_button.setStyleSheet("background-color: blue; color: white;")
        self.tilt_map_button.clicked.connect(self.tilt_map)
        self.middle_layout.addWidget(self.tilt_map_button) 

        # Run overlap optimiser
        self.overlap_button = QPushButton('Overlap Optimiser', self)
        self.overlap_button.setStyleSheet("background-color: yellow; color: red;")
        self.overlap_button.clicked.connect(self.overlap)
        self.middle_layout.addWidget(self.overlap_button)

        # Slider button 1
        self.slider_button = QPushButton('Slider shift', self)
        self.slider_button.setStyleSheet("background-color: black; color: white;")
        self.slider_button.clicked.connect(self.slider)
        self.middle_layout.addWidget(self.slider_button)
        
        self.middle_layout.addStretch()

        label = QLabel("Camera acquisition time")
        input_box = QLineEdit(self)
        input_box.setText("0.0")
        input_box.setMaximumWidth(50)
        input_box.textChanged.connect(self.update_camera_acquisition_time)

        self.inputs.append(input_box)
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(input_box)
        self.middle_layout.addLayout(hbox)

        self.saveButton = QPushButton("Save and Close")
        self.saveButton.setStyleSheet("background-color: red; color: white;")
        self.saveButton.clicked.connect(self.save_and_close)
        self.middle_layout.addWidget(self.saveButton)

        self.load_settings()
        # ======= RIGHT COLUMN: Figure Windows =======
        self.figure_layout = QGridLayout()
        
        # Figure 1 (Phase Map)
        self.figure1, self.ax1 = plt.subplots(figsize=(6, 4))
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure_layout.addWidget(self.canvas1, 0, 0)  # Top-right
        
        # Figure 2 (Basler Image & Line Plot)
        self.figure2 = plt.figure(figsize=(6, 5))

        self.gs = self.figure2.add_gridspec(1, 2, width_ratios=[2, 1])  # Default: ax2 takes 2/3, ax3 takes 1/3

        self.ax2 = self.figure2.add_subplot(self.gs[0])  # ax2 on the left
        self.ax3 = self.figure2.add_subplot(self.gs[1])  # ax3 on the right
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure_layout.addWidget(self.canvas2, 1, 0)  # Below first figure
        self.line, = self.ax3.plot([], [], 'b-')
        
        # ======= Add Layouts to Main Layout =======
        self.main_layout.addLayout(self.control_layout, 1)  # Buttons take 1 part width
        self.main_layout.addLayout(self.middle_layout, 1)
        self.main_layout.addLayout(self.figure_layout, 6)   # Figures take 4 parts width

        # Set layout and window title
        self.setLayout(self.main_layout)
        self.setWindowTitle('PLM GUI')
        apply_dark_theme(self)
        apply_dark_plot_theme(self)

        QTimer.singleShot(0, self.update_value)

    def load_settings(self):
        for i, widget in enumerate(self.inputs):
            if isinstance(widget, QLineEdit):
                value = self.settings.value(f"input_{i}", "")
                widget.setText(value)
        

    def save_and_close(self):
        for i, widget in enumerate(self.inputs):
            if isinstance(widget, QLineEdit):
                self.settings.setValue(f"input_{i}", widget.text())
                self.close()

    def toggle_button(self, idx):
        """Toggles the button state between True and False."""
        self.button_states[idx] = not self.button_states[idx]        
        
    def toggle_zoom(self):
        """Toggles zoom mode and updates the plot."""
        self.zoom_enabled = not self.zoom_enabled
        self.update_value()
               
    def img_toggle_zoom(self):
        # """Toggles zoom mode and updates the plot."""
        self.img_zoom_enabled = not self.img_zoom_enabled
        # self.update_value()

    def camera_trigger_mode(self):
        # """Toggles between hardware triggering and free streaming the camera."""
        self.hardware_triggering_enabled = not self.hardware_triggering_enabled
        self.update_value()
        
    def bitpack_frames(self):
        self.bitpack_enabled = True
        self.update_value()
         
    def toggle_centroid(self):
        """Enables one-shot centroid grabbing."""
        self.grab_centroid_enabled = True

        
    def beam_A_SP_scan(self):
        self.beam_A_SP_scan_enabled = True
        self.update_value()

    def beam_B_SP_scan(self):
        self.beam_B_SP_scan_enabled = True
        self.update_value()

    def beam_A_complex_measurement(self):
        self.beam_A_complex_field_measurement_enabled = True
        self.update_value()

    def beam_B_complex_measurement(self):
        self.beam_B_complex_field_measurement_enabled = True
        self.update_value()

    def clear_beam_A_correction(self):
        self.clear_beam_A_correction_flag = True
        self.update_value()

    def clear_beam_B_correction(self):
        self.clear_beam_B_correction_flag = True
        self.update_value()

    def global_phase_scan(self):
        self.phase_scan_frame_flag = True
        self.update_value()

    def amplitude_ramp_scan(self):
        self.amp_ramp_frame_flag = True
        self.update_value()

    def polarisation_measurement(self):
        self.polarisation_measurement_flag = True
        self.update_value()

    def grab_50(self):
        self.grab_50_flag = True
        self.update_value()

    def multibeam(self):
        self.multibeam_flag = True
        self.update_value()

    def multibeam_seq(self):
        self.multibeam_seq_flag = True
        self.update_value()

    def tilt_map(self):
        self.tilt_mapping_flag = True
        self.update_value()

    def overlap(self):
        self.overlap_optimiser_flag = True
        self.update_value()

    def slider(self):
        self.slider_flag = True
        self.update_value()

    def update_camera_acquisition_time(self):
        try:
            value = float(self.inputs[20].text())
            if value < 100:
                value = 100
            print(f"Updated camera acquisition time: {value}")
            self.camera.ExposureTimeAbs.SetValue(value)
        except ValueError:
            print("Invalid input — not a float")

    from PyQt5.QtGui import QPixmap

    def save_gui_screenshot(widget, filename):
        pixmap = widget.grab()
        pixmap.save(filename)
     
    def beam_A_correction(self):
        """Open a file dialog for the user to load a file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.txt);;CSV Files (*.csv);;Numpy Files (*.npy)", options=options)
    
        if file_path:
            loaded_data = None
            # Check if the file is a .npy file (NumPy array)
            if file_path.endswith(".npy"):
                loaded_data = np.load(file_path)
            elif file_path.endswith(".csv"):
                loaded_data = np.loadtxt(file_path, delimiter=",")
            elif file_path.endswith(".txt"):
                loaded_data = np.loadtxt(file_path)
            else:
                print("Unsupported file format")
                return
        
        self.beam_A_correction_data = loaded_data
        
    
    def beam_B_correction(self):
        """Open a file dialog for the user to load a file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.txt);;CSV Files (*.csv);;Numpy Files (*.npy)", options=options)
    
        if file_path:
            loaded_data_B = None
            # Check if the file is a .npy file (NumPy array)
            if file_path.endswith(".npy"):
                loaded_data_B = np.load(file_path)
            elif file_path.endswith(".csv"):
                loaded_data_B = np.loadtxt(file_path, delimiter=",")
            elif file_path.endswith(".txt"):
                loaded_data_B = np.loadtxt(file_path)
            else:
                print("Unsupported file format")
                return
        
        self.beam_B_correction_data = loaded_data_B    
    

    def update_value(self):
        
        try:
            self.user_values = [float(box.text()) for box in self.inputs]  # Read input values
        except ValueError:
            return  # Stop if any input is invalid
        
        beamA_phase_tilt = generate_phase_tilt(rows, cols, self.user_values[0], self.user_values[1], self.button_states[0], self.button_states[1]) 
        beamB_phase_tilt = generate_phase_tilt(rows, cols, self.user_values[2], self.user_values[3], self.button_states[2], self.button_states[3])
        
        beamA_HG_phase, beamA_HG_amplitude = HG_mode(cols, rows, self.user_values[4], self.user_values[5], self.user_values[8], self.user_values[9], self.user_values[12], self.user_values[13])
        beamB_HG_phase, beamB_HG_amplitude = HG_mode(cols, rows, self.user_values[6], self.user_values[7], self.user_values[10], self.user_values[11], self.user_values[14], self.user_values[15])
        
        global_amplitudes = np.array([self.user_values[16] , self.user_values[17]])
        global_amplitudes = global_amplitudes/ np.max(global_amplitudes)

        beamA_amplitude = beamA_HG_amplitude * global_amplitudes[0]
        beamB_amplitude = beamB_HG_amplitude * global_amplitudes[1]
                
        global_phase_A = (self.user_values[18]/100)*2*np.pi  
        global_phase_B = (self.user_values[19]/100)*2*np.pi 

        if self.clear_beam_A_correction_flag == True:
            self.beam_A_correction_data = np.zeros_like(beamA_phase_tilt) # clear the correction

        if self.clear_beam_B_correction_flag == True:
            self.beam_B_correction_data = np.zeros_like(beamB_phase_tilt)
 

        beamA_phase = beamA_phase_tilt - self.beam_A_correction_data + beamA_HG_phase + global_phase_A
        beamB_phase = beamB_phase_tilt - self.beam_B_correction_data + beamB_HG_phase + global_phase_B

        beamAcomplex = beamA_amplitude * np.exp(1j * beamA_phase)
        beamBcomplex = beamB_amplitude * np.exp(1j * beamB_phase)


        beamA2_phase_tilt = generate_phase_tilt(rows, cols, 10.22, 12.26, self.button_states[0], self.button_states[1]) 
        beamB2_phase_tilt = generate_phase_tilt(rows, cols, 30.0, 12.0, self.button_states[2], self.button_states[3])

        beamA2_HG_phase, beamA2_HG_amplitude = HG_mode(cols, rows, self.user_values[4], self.user_values[5], self.user_values[8], self.user_values[9], self.user_values[12], self.user_values[13])
        beamB2_HG_phase, beamB2_HG_amplitude = HG_mode(cols, rows, self.user_values[6], self.user_values[7], self.user_values[10], self.user_values[11], self.user_values[14], self.user_values[15])
        
        beamA2_amplitude = beamA2_HG_amplitude * global_amplitudes[0] 
        beamB2_amplitude = beamB2_HG_amplitude * global_amplitudes[1] 
                
        global_phase_A2 = (self.user_values[18]/100)*2*np.pi  
        global_phase_B2 = (self.user_values[19]/100)*2*np.pi 

        if self.clear_beam_A_correction_flag == True:
            self.beam_A_correction_data = np.zeros_like(beamA_phase_tilt) # clear the correction

        if self.clear_beam_B_correction_flag == True:
            self.beam_B_correction_data = np.zeros_like(beamB_phase_tilt)
 

        beamA2_phase = beamA2_phase_tilt - self.beam_A_correction_data + beamA2_HG_phase + global_phase_A2
        beamB2_phase = beamB2_phase_tilt - self.beam_B_correction_data + beamB2_HG_phase + global_phase_B2

        beamA2complex = beamA2_amplitude * np.exp(1j * beamA2_phase)
        beamB2complex = beamB2_amplitude * np.exp(1j * beamB2_phase)

        combinedComplex = beamAcomplex + beamBcomplex # + beamA2complex + beamB2complex
        
        amplitude_modulated_combined_phase = amp_mod_phase(combinedComplex) 
        
        plm_phase_map = (amplitude_modulated_combined_phase + np.pi) / (2*np.pi)

        # self.camera.ExposureTimeAbs.SetValue(self.user_values[20])

        self.clear_beam_A_correction_flag = False
        self.clear_beam_B_correction_flag = False

        if self.polarisation_measurement_flag:
            print('Polarisation measurement using rotating linear polariser')
            pol_measure(self.camera, global_amplitudes, '_Rotating linear polariser_')
            self.polarisation_measurement_flag = False

        if self.grab_50_flag:
            print('Grab and save 50 images using current plm config')
            folder_name, offline_folder_name = grab_50_images(self.camera, 'grab50_test')
            filename = os.path.join(folder_name, 'GUI screenshot.png')
            self.save_gui_screenshot(filename)

            filename = os.path.join(offline_folder_name, 'GUI screenshot.png')
            self.save_gui_screenshot(filename)
            self.grab_50_flag = False

        if self.multibeam_flag:
            plm.pause_ui()
            print('Entering multibeam mode - hold on to your hats!')
            combinedComplex = load_multibeam_data(self)
            amplitude_modulated_combined_phase = amp_mod_phase(combinedComplex) 
            plm_phase_map = (amplitude_modulated_combined_phase + np.pi) / (2*np.pi)
            self.multibeam_flag = False
            plm_frame = np.broadcast_to(plm_phase_map[:, :, None], (M, N, numHolograms)).astype(np.float32)
            plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')
                                        
            plm.bitpack_and_insert_gpu(plm_frame, 1)
            plm.resume_ui()

            plm.set_frame(1)
            
            time.sleep(0.2)
            plm.play()
            plm.play()

        if self.multibeam_seq_flag:
            plm.pause_ui()
            print('Running selected multiBeam squence with HW triggering')
            self.multibeam_seq_flag = False
            plm.resume_ui()
            time.sleep(0.2)
            plm.play()
            plm.play()

        if self.tilt_mapping_flag:
            print('Running tilt map sequence \n')
            tilt_mapping(self, plm, camera)
            self.tilt_mapping_flag = False 

        if self.overlap_optimiser_flag:
            print('Attempting to optimise overlap of Beam B with Beam A')
            overlap_optimiser(self, plm, camera)
            self.overlap_optimiser_flag = False

        if self.amp_ramp_frame_flag:
            plm.pause_ui()
            print('Bitpacking amplitude ramp frame')
            amp_ramp_scan_frame = amp_ramp_frame_generator (beamA_phase, beamB_phase, beamA_HG_amplitude, beamB_HG_amplitude )
            plm.bitpack_and_insert_gpu(amp_ramp_scan_frame, 66)
            plm.resume_ui()
            plm.play()
            plm.play()
            # Here we setup the hardware triggered acquistion at 720 Hz (using PLM triggers) but remember that the Hardware triggered
            #button has to be pressed before pressing the amp ramp button
            time.sleep(0.1) 
            print("Sending TTL to Line 3...")
            self.task.start()  # Basler acquisition start trigger
            self.task.write(True)  
            time.sleep(0.05)  
            self.task.write(False)  
            time.sleep(0.05)  
            self.task.stop()

            grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

            #Hardware triggered acquistion starts here
            if grabResult.GrabSucceeded():
                print("TTL trigger received on Line 03.")
                grabResult.Release() 
            else:
                print("Timeout waiting for the first image.")
                self.camera.StopGrabbing()  
                self.camera.Close()
            
            self.camera.StopGrabbing()
            all_images = np.zeros((800, 128, 128), dtype=np.uint8)

            idx = 0 
            self.camera.MaxNumBuffer.Value = 25
            self.camera.TriggerSource.SetValue("Line1")  
            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerMode.SetValue("On")  # Enable per-frame trigger
            
            self.camera.TriggerActivation.SetValue("RisingEdge")
            image_height = 128
            image_width = 128
            self.camera.Width.SetValue(image_width)
            self.camera.Height.SetValue(image_height)

            offset_x = 208
            offset_y = 208            
            self.camera.OffsetX.SetValue(offset_x)
            self.camera.OffsetY.SetValue(offset_y)

            self.camera.ExposureTimeAbs.SetValue(200)
            self.camera.StartGrabbingMax(800)
            last_frame_number = None

            while self.camera.IsGrabbing():
                grab_start = time.perf_counter_ns()  # Start timing grab
                grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                grab_end = time.perf_counter_ns()
                if grabResult.GrabSucceeded():
                    frame_number = grabResult.GetBlockID()
                    
                    img = grabResult.Array
                    all_images[idx]=img                   
                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                
                grab_time_us = (grab_end - grab_start)
                timing_data.append(grab_time_us)
                
                frame_number = grabResult.GetBlockID()
                
                if last_frame_number is not None and frame_number != last_frame_number + 1:
                    missed_triggers = frame_number - last_frame_number - 1
                    print(f"Warning: Missed {missed_triggers} trigger(s)!")
                        
                last_frame_number = frame_number  # Update last frame number
                grabResult.Release()
                idx=idx+1

                if idx==5:
                    plm.set_frame(66)
                    print('Amplitude ramp started')
                    plm.play()
                    plm.play()

            self.camera_trigger_mode_button.setChecked(False)
            self.hardware_triggering_enabled = False
                       
            print('\nAll images acquired')
            folder_name = save_super_pixel_images(all_images, '_Amp ramp')
            filename = os.path.join(folder_name, 'GUI screenshot.png')
            self.save_gui_screenshot(filename)
            
            print('\nSwitching back to free streaming')
            self.camera.TriggerMode.SetValue("Off")        
            offset_x = 0
            offset_y = 0        
            self.camera.OffsetX.SetValue(offset_x)
            self.camera.OffsetY.SetValue(offset_y)
            image_height = 512
            image_width = 512
            self.camera.Width.SetValue(image_width)
            self.camera.Height.SetValue(image_height)
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            plm.set_frame(0)
            time.sleep(0.5)
            plm.play()
            plm.play()
            self.amp_ramp_frame_flag = False

        if self.phase_scan_frame_flag:
            plm.pause_ui()
            print('Bitpacking phase scan frame')
            plm_phase_scan_frame = phase_scanning_frame_generator (beamA_phase_tilt ,self.beam_A_correction_data , beamA_HG_phase , beamB_phase, beamA_amplitude, beamB_amplitude)
            plm_phase_scan_frame = np.transpose(plm_phase_scan_frame, (1, 0, 2)).copy(order='F')
            
            plm.bitpack_and_insert_gpu(plm_phase_scan_frame, 65)

            plm.resume_ui()
            plm.play()
            plm.play()
            
            time.sleep(0.1)
            print("Sending TTL to Line 3...")
            self.task.start()  # Basler acquisition start trigger
            self.task.write(True)  
            time.sleep(0.05)  
            self.task.write(False)  
            time.sleep(0.05)  
            self.task.stop()

            grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

            #Hardware triggered acquistion starts here
            if grabResult.GrabSucceeded():
                print("TTL trigger received on Line 03.")
                grabResult.Release() 
            else:
                print("Timeout waiting for the first image.")
                self.camera.StopGrabbing()  # Important: Stop grabbing before reconfiguring.
                self.camera.Close()
            
            self.camera.StopGrabbing()
            all_images = np.zeros((800, 128, 128), dtype=np.uint8)
            
            idx = 0 
            self.camera.MaxNumBuffer.Value = 25
            self.camera.TriggerSource.SetValue("Line1")  
            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerMode.SetValue("On")  # Enable per-frame trigger
            
            self.camera.TriggerActivation.SetValue("RisingEdge")
            image_height = 128
            image_width = 128
            self.camera.Width.SetValue(image_width)
            self.camera.Height.SetValue(image_height)

            offset_x = 208
            offset_y = 208            
            self.camera.OffsetX.SetValue(offset_x)
            self.camera.OffsetY.SetValue(offset_y)

            self.camera.ExposureTimeAbs.SetValue(200)
            self.camera.StartGrabbingMax(800)
            last_frame_number = None

            while self.camera.IsGrabbing():
                grab_start = time.perf_counter_ns()  # Start timing grab
                grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                grab_end = time.perf_counter_ns()
                if grabResult.GrabSucceeded():
                    frame_number = grabResult.GetBlockID()
                    
                    img = grabResult.Array
                    all_images[idx]=img                   
                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                
                grab_time_us = (grab_end - grab_start)
                timing_data.append(grab_time_us)
                
                frame_number = grabResult.GetBlockID()
                
                if last_frame_number is not None and frame_number != last_frame_number + 1:
                    missed_triggers = frame_number - last_frame_number - 1
                    print(f"Warning: Missed {missed_triggers} trigger(s)!")
                        
                last_frame_number = frame_number  # Update last frame number
                grabResult.Release()
                idx=idx+1

                if idx==5:
                    plm.set_frame(65)
                    print('Global phase scan started')
                    plm.play()
                    plm.play()

            self.camera_trigger_mode_button.setChecked(False)
            self.hardware_triggering_enabled = False
                       
            print('\nAll images acquired')
            folder_name = save_super_pixel_images(all_images, 'phase scan')
            phase_scan_analysor(all_images)

            print('\nSwitching back to free streaming')
            self.camera.TriggerMode.SetValue("Off")        
            offset_x = 0
            offset_y = 0        
            self.camera.OffsetX.SetValue(offset_x)
            self.camera.OffsetY.SetValue(offset_y)
            image_height = 512
            image_width = 512
            self.camera.Width.SetValue(image_width)
            self.camera.Height.SetValue(image_height)
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            plm.set_frame(0)
            time.sleep(0.5)
            plm.play()
            plm.play()
            self.phase_scan_frame_flag = False
                
        if self.bitpack_enabled:
            plm.pause_ui()        
            for i in range(1):
                print('\nBitpacking frame ' + str(i+1))

                plm_frame = np.broadcast_to(plm_phase_map[:, :, None], (M, N, numHolograms)).astype(np.float32)
                plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')
                                        
                plm.bitpack_and_insert_gpu(plm_frame, i)
                plm.resume_ui()

            plm.set_frame(0)
            self.bitpack_enabled = False
            time.sleep(0.2)
            plm.play()
            plm.play()

        if self.hardware_triggering_enabled:
            print('Hardware triggering enabled')

            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        
            print("Switching to hardware-triggered acquisition...")

            self.countOfImagesToGrab=1800
            # Disable any triggers first
            self.camera.TriggerSource.SetValue("Line1")
            self.camera.TriggerMode.SetValue("Off")
                
            # Select GPIO line 3
            self.camera.LineSelector.Value = "Line3"
            self.camera.LineMode.Value = "Input"
            self.camera.TriggerSource.SetValue("Line3")
            self.camera.TriggerMode.SetValue("Off") 

            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerSource.SetValue("Line3")
            self.camera.TriggerMode.SetValue("On")
            self.camera.TriggerActivation.SetValue("RisingEdge")

            image_height = 128
            image_width = 128
            
            self.camera.Width.SetValue(image_width)
            self.camera.Height.SetValue(image_height)
            
            offset_x = 208
            offset_y = 208
            
            self.camera.OffsetX.SetValue(offset_x)
            self.camera.OffsetY.SetValue(offset_y)

            self.camera.ExposureTimeAbs.SetValue(400)

            # Camera is now waiting for a TTL trigger on Line 3 to start the acquistion 
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(" Waiting for Acquisition Start trigger on Line 3...")
            # grabResult = camera.RetrieveResult(1800000, pylon.TimeoutHandling_ThrowException)


        if self.beam_A_SP_scan_enabled or self.beam_B_SP_scan_enabled or self.beam_A_complex_field_measurement_enabled or self.beam_B_complex_field_measurement_enabled:
            plm.set_frame(60) #choose any empty frame to make sure camera not exposed whilst frames are bitpacked
            time.sleep(1)
            plm.pause_ui()
            if self.beam_A_SP_scan_enabled:
                phase_tilt = (beamA_phase_tilt % (2*np.pi)) / (2*np.pi)
            elif self.beam_B_SP_scan_enabled:
                phase_tilt = (beamB_phase_tilt % (2*np.pi)) / (2*np.pi)
            elif self.beam_A_complex_field_measurement_enabled:
                phase_tilt = plm_phase_map
                print('\n Attempting complex field measurement for Beam A')
            elif self.beam_B_complex_field_measurement_enabled:
                phase_tilt = plm_phase_map
                print('\n Attempting complex field measurement for Beam B')

            phase_tilt = phase_tilt.astype(np.float32)   
            print('\n Compiling super pixel frames - this takes about 15 seconds')         
            sPix_phase, nx_sPix, ny_sPix = super_pixel_set_numba(64, 64, phase_tilt, 4)

            sPix_phase_init = super_pixel_set_init (phase_tilt,sPix_phase)

            SP_frames = super_pixel_frames(sPix_phase_init)

            global superPixelFrames
            superPixelFrames = SP_frames

            for i in range(SP_frames.shape[0]):
                print('\n Bitpacking super Pixel frame ' + str(i+1))
                plm_frame = SP_frames[i , :, :, :]    
                plm_frame = np.asfortranarray(np.transpose(plm_frame, (1, 0, 2)))
                
                plm.bitpack_and_insert_gpu(plm_frame, i)
            plm.resume_ui()
            plm.play()
            plm.play()
            sequence = np.arange(frames, dtype=np.uint8)
            plm.set_frame_sequence(sequence)

            
            time.sleep(0.1)
            print("Sending TTL to Line 3...")
            self.task.start()  # Basler acquisition start trigger
            self.task.write(True)  
            time.sleep(0.05)  
            self.task.write(False)  
            time.sleep(0.05)  
            self.task.stop()

            grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

            #Hardware triggered acquistion starts here
            if grabResult.GrabSucceeded():
                print("TTL trigger received on Line 03.")
                grabResult.Release() 
            else:
                print("Timeout waiting for the first image.")
                self.camera.StopGrabbing()  # Important: Stop grabbing before reconfiguring.
                self.camera.Close()
            
            self.camera.StopGrabbing()
            all_images = np.zeros((self.countOfImagesToGrab, image_height, image_width), dtype=np.uint8)
            
            idx = 0 
            self.camera.MaxNumBuffer.Value = 25
            self.camera.TriggerSource.SetValue("Line1")  
            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerMode.SetValue("On")  # Enable per-frame trigger
            
            self.camera.TriggerActivation.SetValue("RisingEdge")

            self.camera.StartGrabbingMax(self.countOfImagesToGrab)
            time.sleep(0.2)
            last_frame_number = None

            while self.camera.IsGrabbing():
                grab_start = time.perf_counter_ns()  # Start timing grab
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                grab_end = time.perf_counter_ns()
                if grabResult.GrabSucceeded():
                    frame_number = grabResult.GetBlockID()
                    
                    img = grabResult.Array
                    all_images[idx]=img                   
                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                
                grab_time_us = (grab_end - grab_start)
                timing_data.append(grab_time_us)
                
                frame_number = grabResult.GetBlockID()
                
                if last_frame_number is not None and frame_number != last_frame_number + 1:
                    missed_triggers = frame_number - last_frame_number - 1
                    print(f"Warning: Missed {missed_triggers} trigger(s)!")
                        
                last_frame_number = frame_number  # Update last frame number
                grabResult.Release()
                idx=idx+1
                # print(idx)

                if idx==5:
                    plm.start_sequence(frames)
                    print('plm sequence started')

            self.camera_trigger_mode_button.setChecked(False)
            self.hardware_triggering_enabled = False
                       
            print('\nAll images acquired')

            if self.beam_A_SP_scan_enabled:
                saveFolder = save_super_pixel_images(all_images, '_Beam A')
                wavefront_correction(all_images, saveFolder, '_Beam A')

            elif self.beam_B_SP_scan_enabled:
                saveFolder = save_super_pixel_images(all_images, '_Beam B')
                wavefront_correction(all_images, saveFolder, '_Beam B')

            elif self.beam_A_complex_field_measurement_enabled:
                saveFolder = save_super_pixel_images(all_images, '_Beam A complex field')
                wavefront_correction(all_images, saveFolder, '_Beam A complex field')

            elif self.beam_B_complex_field_measurement_enabled:
                saveFolder = save_super_pixel_images(all_images, '_Beam B complex field')
                wavefront_correction(all_images, saveFolder, '_Beam B complex field')
            

            self.beam_A_SP_scan_enabled= False
            self.beam_B_SP_scan_enabled= False
            self.beam_A_complex_field_measurement_enabled = False
            self.beam_B_complex_field_measurement_enabled = False
            print('\nSwitching back to free streaming')
            self.camera.TriggerMode.SetValue("Off")        
            offset_x = 0
            offset_y = 0        
            self.camera.OffsetX.SetValue(offset_x)
            self.camera.OffsetY.SetValue(offset_y)
            image_height = 512
            image_width = 512
            self.camera.Width.SetValue(image_width)
            self.camera.Height.SetValue(image_height)
            self.camera.ExposureTimeAbs.SetValue(200)
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            plm.play()
            plm.play()

        

        if self.zoom_enabled:
            center_x, center_y = 800 // 2, 1358 // 2
            delta = 30
            zoomed_array = plm_phase_map[center_x - delta:center_x + delta, center_y - delta:center_y + delta]  
            self.ax1.clear()
            
            self.ax1.imshow(zoomed_array, cmap='viridis', aspect='equal')
            self.ax1.axis('off')
            
        else:
            self.ax1.clear()
            
            self.ax1.imshow(plm_phase_map  , cmap='viridis', aspect='equal')
            # self.ax1.set_title("PLM hologram")
            self.ax1.axis('off')
        
               
        self.canvas1.draw()
            
        global plm_hologram
        plm_hologram = plm_phase_map # This allows the plm phase map to be analysed after the GUI has closed. 
    
    def update_camera_feed(self):
       #"""Fetches a new frame from the Basler camera and updates the GUI."""
        if not self.hardware_triggering_enabled and self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(50000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                try:
                    if grab_result.PixelType == pylon.PixelType_Mono8:
                        img = grab_result.Array
                    else:
                        print(f"Unsupported pixel format: {grab_result.PixelType}")
                        grab_result.Release()
                        return
                except Exception as e:
                    print("Error reading grab result:", e)
                    grab_result.Release()
                    return
                # img = grab_result.Array               
              
                if self.grab_centroid_enabled: 
                   self.centroid_x, self.centroid_y = baslerCentroid(img, 3, 5)
                   print('XY centroid = ' + str(self.centroid_x) + ' ' + str(self.centroid_y) + ' pix')
                   self.grab_centroid_enabled = False

                if self.img_zoom_enabled and self.centroid_x is not None:
                    zoomed_image_half_size = 40
                    img = img[self.centroid_y - zoomed_image_half_size : self.centroid_y + zoomed_image_half_size ,self.centroid_x - zoomed_image_half_size : self.centroid_x + zoomed_image_half_size]  
                    self.ax2.clear()
                    self.ax2.imshow(img, cmap='gray')  
                    self.ax2.axis('off') 
                    self.ax3.set_visible(True)    
                    self.ax3.set_axis_on()               
                   
                    rectangle_half=10
                   
                    height, width = img.shape
                    rect_x_coord = width // 2 - rectangle_half
                    rect_y_coord = height // 2 - rectangle_half
                   
                    rect_end_x = rect_x_coord + int(2*rectangle_half)
                    rect_end_y = rect_y_coord + int(2*rectangle_half)
                   
                    roiRect = patches.Rectangle((rect_x_coord, rect_y_coord), int(rectangle_half*2), int(rectangle_half*2),  linewidth=1, edgecolor='r', facecolor='none')
                    backgroundRect = patches.Rectangle((int(rect_x_coord-2*rectangle_half), rect_y_coord), int(rectangle_half*2), int(rectangle_half*2), linewidth=1, edgecolor='b', facecolor='none')
                    self.ax2.add_patch(roiRect)
                    self.ax2.add_patch(backgroundRect) 
                  
                    self.ax3.set_xlim(0, 200)
                    self.ax3.set_ylim(10, 2e4) 
                    self.ax3.set_yscale('log')
                   
                    image_sum = float(np.sum(img[rect_y_coord : rect_end_y , rect_x_coord : rect_end_x]))
                    background_sum = float(np.sum(img[rect_y_coord : rect_end_y , rect_x_coord-int(2*rectangle_half) : rect_end_x-int(2*rectangle_half)])) 
                   
                    self.xdata.append(self.zoom_counter)
                    self.ydata.append(image_sum-background_sum)
                    self.ax3.tick_params(axis='x', labelsize=6)  
                    self.ax3.tick_params(axis='y', labelsize=6)                   
                   
                    self.line.set_xdata(range(len(self.xdata)))
                    self.line.set_ydata(self.ydata)

                    self.ax3.relim()
                    self.ax3.autoscale_view()      
                    pixSum=np.array(self.ydata)
                    if len(pixSum) >= 2:
                        mean_val = np.mean(pixSum[-10:-1])
                    else:
                        mean_val = 0
                    self.ax3.set_title(str(np.round(mean_val)), fontsize = 12, fontweight='bold' )
                    self.zoom_counter += 1  
                    self.canvas2.draw() 
                  
                   
                else:
                    self.ax2.clear()
                    self.ax2.imshow(img, cmap='gray')  
                    self.ax2.axis('off')  
                    height, width = img.shape
                    border = patches.Rectangle((-0.5,-0.5), width+1, height+1, linewidth=4, edgecolor = 'magenta', facecolor = 'none')
                    self.ax2.add_patch(border)

                    self.ax3.set_axis_off()  # Turn off the axis (no ticks, no labels)
                    self.ax3.set_visible(False) 
                    
                    self.canvas2.draw()  


                grab_result.Release()

               

    def closeEvent(self, event):
        """Gracefully stops the camera when the window is closed."""
        self.camera.StopGrabbing()
        self.camera.Close()
        event.accept()

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InteractiveGUI()
    window.show()
    app.exec_()
    app.quit()
   
plm.cleanup()

plt.close('all')
matplotlib.use('QtAgg')
