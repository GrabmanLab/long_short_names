#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on July 28, 2026, at 16:40
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'LtmStm'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'age': '',
    'gender': ['', 'man', 'woman', 'nonbinary', 'other not listed'],
    'ethnicity': ['', 'white', 'hispanic or latino/a/e', 'black or african american', 'american indian or alaska native', 'asian', 'native hawaiian or pacific islander', 'arabic or middle eastern',  'other not listed'],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Public\\Documents\\long_short_names\\LtmStm.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('begin_resp') is None:
        # initialise begin_resp
        begin_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='begin_resp',
        )
    if deviceManager.getDevice('cont_resp') is None:
        # initialise cont_resp
        cont_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='cont_resp',
        )
    if deviceManager.getDevice('listA_instrx_resp') is None:
        # initialise listA_instrx_resp
        listA_instrx_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='listA_instrx_resp',
        )
    if deviceManager.getDevice('listB_instrx_1_resp') is None:
        # initialise listB_instrx_1_resp
        listB_instrx_1_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='listB_instrx_1_resp',
        )
    if deviceManager.getDevice('listB_instrx_2_resp') is None:
        # initialise listB_instrx_2_resp
        listB_instrx_2_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='listB_instrx_2_resp',
        )
    if deviceManager.getDevice('prac_resp') is None:
        # initialise prac_resp
        prac_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_resp',
        )
    if deviceManager.getDevice('listB_instrx_4_resp') is None:
        # initialise listB_instrx_4_resp
        listB_instrx_4_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='listB_instrx_4_resp',
        )
    if deviceManager.getDevice('prac_cont') is None:
        # initialise prac_cont
        prac_cont = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_cont',
        )
    if deviceManager.getDevice('prac_cont2') is None:
        # initialise prac_cont2
        prac_cont2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_cont2',
        )
    if deviceManager.getDevice('rdm_instrx_resp') is None:
        # initialise rdm_instrx_resp
        rdm_instrx_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rdm_instrx_resp',
        )
    if deviceManager.getDevice('dots_resp') is None:
        # initialise dots_resp
        dots_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='dots_resp',
        )
    if deviceManager.getDevice('test_resp') is None:
        # initialise test_resp
        test_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='test_resp',
        )
    if deviceManager.getDevice('test_resp_2') is None:
        # initialise test_resp_2
        test_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='test_resp_2',
        )
    if deviceManager.getDevice('test_resp_3') is None:
        # initialise test_resp_3
        test_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='test_resp_3',
        )
    if deviceManager.getDevice('prac_test_resp') is None:
        # initialise prac_test_resp
        prac_test_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_test_resp',
        )
    if deviceManager.getDevice('prac1_test_resp') is None:
        # initialise prac1_test_resp
        prac1_test_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac1_test_resp',
        )
    if deviceManager.getDevice('prac_cont3') is None:
        # initialise prac_cont3
        prac_cont3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_cont3',
        )
    if deviceManager.getDevice('prac2_test_resp') is None:
        # initialise prac2_test_resp
        prac2_test_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac2_test_resp',
        )
    if deviceManager.getDevice('prac_cont4') is None:
        # initialise prac_cont4
        prac_cont4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_cont4',
        )
    if deviceManager.getDevice('test_resp_4') is None:
        # initialise test_resp_4
        test_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='test_resp_4',
        )
    if deviceManager.getDevice('test_on_resp') is None:
        # initialise test_on_resp
        test_on_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='test_on_resp',
        )
    if deviceManager.getDevice('survey_instrx_1_resp') is None:
        # initialise survey_instrx_1_resp
        survey_instrx_1_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='survey_instrx_1_resp',
        )
    if deviceManager.getDevice('survey_instrx_2_resp') is None:
        # initialise survey_instrx_2_resp
        survey_instrx_2_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='survey_instrx_2_resp',
        )
    if deviceManager.getDevice('survey_prac_RT') is None:
        # initialise survey_prac_RT
        survey_prac_RT = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='survey_prac_RT',
        )
    if deviceManager.getDevice('survey_RT') is None:
        # initialise survey_RT
        survey_RT = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='survey_RT',
        )
    if deviceManager.getDevice('end_resp') is None:
        # initialise end_resp
        end_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "start" ---
    begin_experiment = visual.TextStim(win=win, name='begin_experiment',
        text='On the next page you will read the informed consent form. There are two pages total.\n\nPress the spacebar to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    begin_resp = keyboard.Keyboard(deviceName='begin_resp')
    # Run 'Begin Experiment' code from set_trials
    #set corresponding files
    listA_file = f"participant_list/listA/{expInfo['participant']}_listA.csv"
    listB_file = f"participant_list/listB/{expInfo['participant']}_listB.csv"
    test_file  = f"participant_list/test/{expInfo['participant']}_test.csv"
    names_file = f"participant_list/names_list/{expInfo['participant']}_names_list.csv"
    
    import random
    
    if random.random() < 0.5:
        yes_key = 'left'
        no_key = 'right'
    else:
        yes_key = 'right'
        no_key = 'left'
        
    thisExp.addData('yes_key', yes_key)
    thisExp.addData('no_key', no_key)
    
    
    # --- Initialize components for Routine "consent" ---
    # Run 'Begin Experiment' code from load_consent
    import glob
    
    consent_folder = os.path.join(_thisDir,'consent')
    consent_files = glob.glob(os.path.join(consent_folder, "*.png"))
    consent_files.sort()
    
    nForms = len(consent_files)
    consent_img = visual.ImageStim(
        win=win,
        name='consent_img', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=[612/ win.size[1], 792 / win.size[1]],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    consent_cont = visual.Rect(
        win=win, name='consent_cont',
        width=(0.3, 0.10)[0], height=(0.3, 0.10)[1],
        ori=0.0, pos=[0, -792 / win.size[1]/2 - .05], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor=[-1.0000, -1.0000, 0.0902],
        opacity=None, depth=-2.0, interpolate=True)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    yes_continue = visual.TextStim(win=win, name='yes_continue',
        text=None,
        font='Arial',
        pos=[0, -792 / win.size[1]/2 - .05], draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "set_block" ---
    block_text = visual.TextStim(win=win, name='block_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cont_resp = keyboard.Keyboard(deviceName='cont_resp')
    
    # --- Initialize components for Routine "listA_instrx" ---
    listA_instrx_text = visual.TextStim(win=win, name='listA_instrx_text',
        text='You will see a series of faces, each paired to a name. Please pay close attention to both the faces and names, because we will later ask you questions about who you saw!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    listA_instrx_resp = keyboard.Keyboard(deviceName='listA_instrx_resp')
    listA_instrx_cont = visual.TextStim(win=win, name='listA_instrx_cont',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "begin_listA" ---
    ListA_text = visual.TextStim(win=win, name='ListA_text',
        text='Get ready...',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "listA" ---
    fix_crossA = visual.TextStim(win=win, name='fix_crossA',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    lisA_image = visual.ImageStim(
        win=win,
        name='lisA_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(720/ win.size[1],576/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    listA_name = visual.TextStim(win=win, name='listA_name',
        text='',
        font='Arial',
        pos=(0,-0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "listB_instrx_1" ---
    listB_instrx_1_txt = visual.TextStim(win=win, name='listB_instrx_1_txt',
        text='Now we will try something new. You will see faces and names again, just like before. The names are the same. But this time, they are matched with different people.\n\n\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    listB_instrx_1_resp = keyboard.Keyboard(deviceName='listB_instrx_1_resp')
    listB_instrx_1_cont = visual.TextStim(win=win, name='listB_instrx_1_cont',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "listB_instrx_2" ---
    listB_instrx_2_txt = visual.TextStim(win=win, name='listB_instrx_2_txt',
        text="Before each face, you will get an instruction. It tells you who to pay attention to.\n\nIf you see 'NEXT', pay close attention to the person shown right after.\n\nIf you see 'BACK', picture the person from before who had this same name. Focus on seeing their face in your mind.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    listB_instrx_2_resp = keyboard.Keyboard(deviceName='listB_instrx_2_resp')
    listB_instrx_2_cont = visual.TextStim(win=win, name='listB_instrx_2_cont',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "prac_setup" ---
    prac_text = visual.TextStim(win=win, name='prac_text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_resp = keyboard.Keyboard(deviceName='prac_resp')
    # Run 'Begin Experiment' code from prac_YesNo
    practice_correct = []
    #selected 'yes' so pratice trials start at the beginning.
    prac_selected = 999
    
    # --- Initialize components for Routine "listB_instrx_3" ---
    listB_instrx_3_txt = visual.TextStim(win=win, name='listB_instrx_3_txt',
        text='Imagine that you saw these two people before...',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "pracA" ---
    pracA_image = visual.ImageStim(
        win=win,
        name='pracA_image', 
        image='practice_trials/Original_Derpina.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    pracA_name = visual.TextStim(win=win, name='pracA_name',
        text='beyonce',
        font='Arial',
        pos=(0, -0.22), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    pracA_isi = visual.TextStim(win=win, name='pracA_isi',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    pracA_image2 = visual.ImageStim(
        win=win,
        name='pracA_image2', 
        image='practice_trials/D4cxln0-f1c9d0d2-1a40-4de5-b9fd-9551ea38a6ef.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    pracA_name2 = visual.TextStim(win=win, name='pracA_name2',
        text='jay-z',
        font='Arial',
        pos=(0, -0.223), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "listB_instrx_4" ---
    listB_instrx_4_txt = visual.TextStim(win=win, name='listB_instrx_4_txt',
        text="Now, you'll see new people with the same names. Reminder:\n\n'Next' means pay attention to the face on the next screen.\n\n'Back' means picture the person from before who had this same name.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    listB_instrx_4_resp = keyboard.Keyboard(deviceName='listB_instrx_4_resp')
    listB_instrx_4_cont = visual.TextStim(win=win, name='listB_instrx_4_cont',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "pracB1" ---
    pracB_isi = visual.TextStim(win=win, name='pracB_isi',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    pracB_text = visual.TextStim(win=win, name='pracB_text',
        text='back',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    pracB_image = visual.ImageStim(
        win=win,
        name='pracB_image', 
        image='practice_trials/Black_Hair_Derpina.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    pracB_name = visual.TextStim(win=win, name='pracB_name',
        text='beyonce',
        font='Arial',
        pos=(0, -0.22), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    pracB_verify = visual.TextStim(win=win, name='pracB_verify',
        text='In this trial, which face should you be thinking about?',
        font='Arial',
        pos=(0,.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    pracB_left = visual.ImageStim(
        win=win,
        name='pracB_left', 
        image='practice_trials/Original_Derpina.jpg', mask=None, anchor='center',
        ori=0.0, pos=(-.2, -.2), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    pracB_right = visual.ImageStim(
        win=win,
        name='pracB_right', 
        image='practice_trials/Black_Hair_Derpina.jpg', mask=None, anchor='center',
        ori=0.0, pos=(.2, -.2), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    LeftB1_resp = visual.ButtonStim(win, 
        text=None, font='Arvo',
        pos=(-.2, -.2),
        letterHeight=0.03,
        size=(0.3, 0.3), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[0.0000, 0.0000, 0.0000], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=0.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='LeftB1_resp',
        depth=-7
    )
    LeftB1_resp.buttonClock = core.Clock()
    RightB1_resp = visual.ButtonStim(win, 
        text=None, font='Arvo',
        pos=(.2, -.2),
        letterHeight=0.03,
        size=(0.3, 0.3), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[0.0000, 0.0000, 0.0000], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=0.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='RightB1_resp',
        depth=-8
    )
    RightB1_resp.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "pracB1_feedback" ---
    B1_feedback = visual.TextStim(win=win, name='B1_feedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    B1_cont = visual.TextStim(win=win, name='B1_cont',
        text='press the spacebar to continue.',
        font='Arial',
        pos=(0,-.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_cont = keyboard.Keyboard(deviceName='prac_cont')
    
    # --- Initialize components for Routine "pracB2" ---
    pracB2_isi = visual.TextStim(win=win, name='pracB2_isi',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    pracB2_text = visual.TextStim(win=win, name='pracB2_text',
        text='next ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    pracB2_image = visual.ImageStim(
        win=win,
        name='pracB2_image', 
        image='practice_trials/D6f-removebg-preview.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    pracB2_name = visual.TextStim(win=win, name='pracB2_name',
        text='jay-z',
        font='Arial',
        pos=(0, -0.22), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prabB2_verify = visual.TextStim(win=win, name='prabB2_verify',
        text='In this trial, which smiley face should you be thinking about? ',
        font='Arial',
        pos=(0,.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    pracB2_left = visual.ImageStim(
        win=win,
        name='pracB2_left', 
        image='practice_trials/D4cxln0-f1c9d0d2-1a40-4de5-b9fd-9551ea38a6ef.jpg', mask=None, anchor='center',
        ori=0.0, pos=(-.2, -.2), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    pracB2_right = visual.ImageStim(
        win=win,
        name='pracB2_right', 
        image='practice_trials/D6f-removebg-preview.jpg', mask=None, anchor='center',
        ori=0.0, pos=(.2, -.2), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    LeftB2_resp = visual.ButtonStim(win, 
        text=None, font='Arvo',
        pos=(-.2, -.2),
        letterHeight=0.03,
        size=(0.3, 0.3), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[0.0000, 0.0000, 0.0000], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=0.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='LeftB2_resp',
        depth=-7
    )
    LeftB2_resp.buttonClock = core.Clock()
    RightB2_resp = visual.ButtonStim(win, 
        text=None, font='Arvo',
        pos=(.2, -.2),
        letterHeight=0.03,
        size=(0.3, 0.3), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[-1.0000, -1.0000, -1.0000], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=0.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='RightB2_resp',
        depth=-8
    )
    RightB2_resp.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "pracB2_feedback" ---
    B2_feedback = visual.TextStim(win=win, name='B2_feedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    B2_cont = visual.TextStim(win=win, name='B2_cont',
        text='press the spacebar to continue.',
        font='Arial',
        pos=(0,-.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_cont2 = keyboard.Keyboard(deviceName='prac_cont2')
    
    # --- Initialize components for Routine "begin_listB" ---
    ListB_text = visual.TextStim(win=win, name='ListB_text',
        text='Great! We will now show you the real trials...',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "listB" ---
    fix_crossB = visual.TextStim(win=win, name='fix_crossB',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    back_next_text = visual.TextStim(win=win, name='back_next_text',
        text='',
        font='Arial',
        pos=(0,0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    listB_image = visual.ImageStim(
        win=win,
        name='listB_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(720/ win.size[1],576/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    listB_name = visual.TextStim(win=win, name='listB_name',
        text='',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "rdm_instrx" ---
    rdm_instrx_txt = visual.TextStim(win=win, name='rdm_instrx_txt',
        text='Now you will do a new task. You will see a circle of moving dots. Most dots move randomly. But, overall, more dots move LEFT or more dots move RIGHT. Decide: do MORE dots move LEFT or RIGHT?\n\n-Press the LEFT arrow key for LEFT.\n-Press the RIGHT arrow key for RIGHT.\n\nPlease answer as fast and accurately as possible. If not sure, take your best guess.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    rdm_instrx_resp = keyboard.Keyboard(deviceName='rdm_instrx_resp')
    rdm_instrx_cont = visual.TextStim(win=win, name='rdm_instrx_cont',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "begin_rdm" ---
    begin_rdm_txt = visual.TextStim(win=win, name='begin_rdm_txt',
        text='Get ready! Please put your fingers on the left and right arrow keys.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rdm_isi" ---
    isi_rdm = visual.TextStim(win=win, name='isi_rdm',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "distractor_dots" ---
    dots = visual.DotStim(
        win=win, name='dots',
        nDots=100, dotSize=2.0,
        speed=0.01, dir=1.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=0.5, fieldAnchor='center', fieldShape='circle',
        signalDots='different', noiseDots='direction',dotLife=3.0,
        color=[1.0,1.0,1.0], colorSpace='rgb', opacity=None,
        depth=0.0)
    dots_resp = keyboard.Keyboard(deviceName='dots_resp')
    # Run 'Begin Experiment' code from dots_code
    import random
    
    rdm_clock = core.Clock()
    rdm_total_duration = 30
    
    # --- Initialize components for Routine "distractor_feedback" ---
    dots_feedback_text = visual.TextStim(win=win, name='dots_feedback_text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "test_instr_1" ---
    test_instructions = visual.TextStim(win=win, name='test_instructions',
        text='Now you will do a memory test.\n\nOn each trial, you will see a face. This may be a different photo of someone you saw before, at any point in the experiment. Or it may be a person you have never seen at all. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    test_resp = keyboard.Keyboard(deviceName='test_resp')
    test_continue = visual.TextStim(win=win, name='test_continue',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "test_instr_2" ---
    test_instructions_2 = visual.TextStim(win=win, name='test_instructions_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    test_resp_2 = keyboard.Keyboard(deviceName='test_resp_2')
    test_continue_2 = visual.TextStim(win=win, name='test_continue_2',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # Run 'Begin Experiment' code from test_trials_2
    on_instruction_text = (
        f"The photo may look a little different. But ask yourself: did I see this person before?\n\n"
        f"If YES, press the '{yes_key.upper()}' arrow button. Then rate how sure you are. "
        f"Next, type the name that goes with the face, and rate how sure you are about that too.\n\n"
        f"If NO, press the '{no_key.upper()}' arrow button. Then rate how sure you are.\n\n"
        f"For all ratings: 50% means you are just guessing. 100% means you are completely certain"
    )
    
    # --- Initialize components for Routine "test_instr_3" ---
    test_instructions_3 = visual.TextStim(win=win, name='test_instructions_3',
        text="If you remember seeing the person, but don't remember their name, click the 'Don't Remember' button (the box will fill in with 'idk', short for 'I don't know'). Please do your best to remember their names though!",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    test_resp_3 = keyboard.Keyboard(deviceName='test_resp_3')
    test_continue_3 = visual.TextStim(win=win, name='test_continue_3',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "prac_test_setup" ---
    prac_test_txt = visual.TextStim(win=win, name='prac_test_txt',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_test_resp = keyboard.Keyboard(deviceName='prac_test_resp')
    # Run 'Begin Experiment' code from prac_test_YesNo
    practice_test_correct = []
    #selected 'yes' so pratice trials start at the beginning.
    prac_test_selected = 999
    
    # --- Initialize components for Routine "prac_test1" ---
    # Run 'Begin Experiment' code from prac_TestCode
    if yes_key == 'left':
        left_label_text = "Yes\nLEFT "
        right_label_text = "No\nRIGHT"
    else:
        left_label_text = "No\nLEFT "
        right_label_text = "Yes\nRIGHT"
        
    prac1_isi = visual.TextStim(win=win, name='prac1_isi',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_image = visual.ImageStim(
        win=win,
        name='prac_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    prac_OldNew = visual.TextStim(win=win, name='prac_OldNew',
        text='Do you remember seeing this face?',
        font='Arial',
        pos=(0,-.25), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac1_test_resp = keyboard.Keyboard(deviceName='prac1_test_resp')
    prac1_left_resp = visual.TextStim(win=win, name='prac1_left_resp',
        text='',
        font='Arial',
        pos=(-.20, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    prac1_right_resp = visual.TextStim(win=win, name='prac1_right_resp',
        text='',
        font='Arial',
        pos=(.20, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "prac1_confidence" ---
    prac_image2 = visual.ImageStim(
        win=win,
        name='prac_image2', 
        image='practice_trials/c859eac8ff5175847ec6aca0a830261a.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac_question = visual.TextStim(win=win, name='prac_question',
        text='How confident are you?\n\n',
        font='Arial',
        pos=(0, -0.30), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_conf1 = visual.Slider(win=win, name='prac_conf1',
        startValue=None, size=(1.0, 0.05), pos=(0, -.35), units=win.units,
        labels=["50%", "60%", "70%", "80%", "90%", "100%"], ticks=(1, 2, 3, 4, 5, 6), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    prac_conf1_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(.7,-.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac_conf1_cont',
        depth=-3
    )
    prac_conf1_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "prac1_textbox" ---
    prac_image3 = visual.ImageStim(
        win=win,
        name='prac_image3', 
        image='practice_trials/c859eac8ff5175847ec6aca0a830261a.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac_OldTextbox = visual.TextBox2(
         win, text=None, placeholder='', font='Arial',
         ori=0.0, pos=(0, -.3), draggable=False,      letterHeight=0.05,
         size=(.8, 0.8), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='prac_OldTextbox',
         depth=-1, autoLog=True,
    )
    prac_TextboxCont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0.4 ,-0.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac_TextboxCont',
        depth=-2
    )
    prac_TextboxCont.buttonClock = core.Clock()
    prac_idk_button = visual.ButtonStim(win, 
        text="Don't Remember", font='Arvo',
        pos=(0, -.4),
        letterHeight=0.025,
        size=(.3, .1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='grey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac_idk_button',
        depth=-3
    )
    prac_idk_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "prac1_confidence2" ---
    prac_image4 = visual.ImageStim(
        win=win,
        name='prac_image4', 
        image='practice_trials/c859eac8ff5175847ec6aca0a830261a.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    prac_question2 = visual.TextStim(win=win, name='prac_question2',
        text='How confident are you that this name is correct?\n\n',
        font='Arial',
        pos=(0, -0.30), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_conf2 = visual.Slider(win=win, name='prac_conf2',
        startValue=None, size=(1.0, 0.05), pos=(0, -.35), units=win.units,
        labels=["50%", "60%", "70%", "80%", "90%", "100%"], ticks=(1, 2, 3, 4, 5, 6), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    prac_conf2_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(.7,-.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac_conf2_cont',
        depth=-4
    )
    prac_conf2_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "prac1_respFeedback" ---
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac1_cont = visual.TextStim(win=win, name='prac1_cont',
        text='press the spacebar to continue.',
        font='Arial',
        pos=(0,-.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_cont3 = keyboard.Keyboard(deviceName='prac_cont3')
    
    # --- Initialize components for Routine "prac_test2" ---
    # Run 'Begin Experiment' code from prac2_TestCode
    if yes_key == 'left':
        left_label_text = "Yes\nLEFT "
        right_label_text = "No\nRIGHT"
    else:
        left_label_text = "No\nLEFT "
        right_label_text = "Yes\nRIGHT"
        
    prac2_isi = visual.TextStim(win=win, name='prac2_isi',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac2_image = visual.ImageStim(
        win=win,
        name='prac2_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    prac2_OldNew = visual.TextStim(win=win, name='prac2_OldNew',
        text='Do you remember seeing this face?',
        font='Arial',
        pos=(0,-.25), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac2_test_resp = keyboard.Keyboard(deviceName='prac2_test_resp')
    prac2_left_resp = visual.TextStim(win=win, name='prac2_left_resp',
        text='',
        font='Arial',
        pos=(-.20, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    prac2_right_resp = visual.TextStim(win=win, name='prac2_right_resp',
        text='',
        font='Arial',
        pos=(.20, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "prac2_confidence" ---
    prac2_image2 = visual.ImageStim(
        win=win,
        name='prac2_image2', 
        image='practice_trials/Black_Hair_Derpina.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac2_question = visual.TextStim(win=win, name='prac2_question',
        text='How confident are you?\n\n',
        font='Arial',
        pos=(0, -0.30), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac2_conf1 = visual.Slider(win=win, name='prac2_conf1',
        startValue=None, size=(1.0, 0.05), pos=(0, -.35), units=win.units,
        labels=["50%", "60%", "70%", "80%", "90%", "100%"], ticks=(1, 2, 3, 4, 5, 6), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    prac2_conf1_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(.7,-.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac2_conf1_cont',
        depth=-3
    )
    prac2_conf1_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "prac2_textbox" ---
    prac2_image3 = visual.ImageStim(
        win=win,
        name='prac2_image3', 
        image='practice_trials/Black_Hair_Derpina.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac2_OldTextbox = visual.TextBox2(
         win, text=None, placeholder='', font='Arial',
         ori=0.0, pos=(0, -.3), draggable=False,      letterHeight=0.05,
         size=(.8, 0.8), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='prac2_OldTextbox',
         depth=-1, autoLog=True,
    )
    prac2_TextboxCont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0.4 ,-0.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac2_TextboxCont',
        depth=-2
    )
    prac2_TextboxCont.buttonClock = core.Clock()
    prac2_idk_button = visual.ButtonStim(win, 
        text="Don't Remember", font='Arvo',
        pos=(0, -.40),
        letterHeight=0.025,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='grey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac2_idk_button',
        depth=-3
    )
    prac2_idk_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "prac2_confidence2" ---
    prac2_image4 = visual.ImageStim(
        win=win,
        name='prac2_image4', 
        image='practice_trials/Black_Hair_Derpina.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(357/ win.size[1],360/ win.size[1]),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    prac2_question2 = visual.TextStim(win=win, name='prac2_question2',
        text='How confident are you that this name is correct?\n',
        font='Arial',
        pos=(0, -0.30), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac2_conf2 = visual.Slider(win=win, name='prac2_conf2',
        startValue=None, size=(1.0, 0.05), pos=(0, -.35), units=win.units,
        labels=["50%", "60%", "70%", "80%", "90%", "100%"], ticks=(1, 2, 3, 4, 5, 6), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    prac2_conf2_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(.7,-.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='prac2_conf2_cont',
        depth=-4
    )
    prac2_conf2_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "prac2_respFeedback" ---
    feedback_text2 = visual.TextStim(win=win, name='feedback_text2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac2_cont = visual.TextStim(win=win, name='prac2_cont',
        text='press the spacebar to continue.',
        font='Arial',
        pos=(0,-.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_cont4 = keyboard.Keyboard(deviceName='prac_cont4')
    
    # --- Initialize components for Routine "test_instr_4" ---
    test_instructions_4 = visual.TextStim(win=win, name='test_instructions_4',
        text='You will now see the actual trials. Please make sure to put your fingers above the left and right arrow keys before each trial.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    test_resp_4 = keyboard.Keyboard(deviceName='test_resp_4')
    test_continue_4 = visual.TextStim(win=win, name='test_continue_4',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "test_prep" ---
    test_prep_txt = visual.TextStim(win=win, name='test_prep_txt',
        text='Ready?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "isi" ---
    isi_test = visual.TextStim(win=win, name='isi_test',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "test" ---
    # Run 'Begin Experiment' code from TestCode
    if yes_key == 'left':
        left_label_text = "Yes\nLEFT "
        right_label_text = "No\nRIGHT"
    else:
        left_label_text = "No\nLEFT "
        right_label_text = "Yes\nRIGHT"
        
    test_image = visual.ImageStim(
        win=win,
        name='test_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(.3,.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    test_oldnew_txt = visual.TextStim(win=win, name='test_oldnew_txt',
        text='Do you remember seeing this face?',
        font='Arial',
        pos=(0,-.25), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    test_on_resp = keyboard.Keyboard(deviceName='test_on_resp')
    test_left_resp = visual.TextStim(win=win, name='test_left_resp',
        text='',
        font='Arial',
        pos=(-.20, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    test_right_resp = visual.TextStim(win=win, name='test_right_resp',
        text='',
        font='Arial',
        pos=(.20, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "resp_confidence" ---
    test_image2 = visual.ImageStim(
        win=win,
        name='test_image2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(.3, .4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    conf_question1 = visual.TextStim(win=win, name='conf_question1',
        text='How confident are you?\n\n',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    confidence1 = visual.Slider(win=win, name='confidence1',
        startValue=None, size=(1.0, 0.05), pos=(0, -.35), units=win.units,
        labels=["50%", "60%", "70%", "80%", "90%", "100%"], ticks=(1, 2, 3, 4, 5, 6), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    conf1_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(.7,-.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='conf1_cont',
        depth=-3
    )
    conf1_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "textbox" ---
    test_image3 = visual.ImageStim(
        win=win,
        name='test_image3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(.3, .4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    old_textbox = visual.TextBox2(
         win, text=None, placeholder='', font='Arial',
         ori=0.0, pos=(0, -.3), draggable=False,      letterHeight=0.05,
         size=(.8, 0.8), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='old_textbox',
         depth=-1, autoLog=True,
    )
    textbox_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0.4 ,-0.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='textbox_cont',
        depth=-2
    )
    textbox_cont.buttonClock = core.Clock()
    textbox_idk = visual.ButtonStim(win, 
        text="Don't Remember", font='Arvo',
        pos=(0, -.40),
        letterHeight=0.025,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='grey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='textbox_idk',
        depth=-3
    )
    textbox_idk.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "resp_confidence2" ---
    test_image4 = visual.ImageStim(
        win=win,
        name='test_image4', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(.3, .4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    conf_question2 = visual.TextStim(win=win, name='conf_question2',
        text='How confident are you that this name is correct?\n',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    confidence2 = visual.Slider(win=win, name='confidence2',
        startValue=None, size=(1.0, 0.05), pos=(0, -.35), units=win.units,
        labels=["50%", "60%", "70%", "80%", "90%", "100%"], ticks=(1, 2, 3, 4, 5, 6), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    conf2_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(.7,-.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='conf2_cont',
        depth=-4
    )
    conf2_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "survey_instrx_1" ---
    survey_instrx_1_txt = visual.TextStim(win=win, name='survey_instrx_1_txt',
        text="Now you will see names from the experiment, one at a time. For each name, ask yourself: do I know someone with this name? This could be someone in your life, or someone you know of (like a celebrity).\n\nAs soon as someone with that name comes to mind, press 'SPACE' right away. Don't wait or think it over — just respond the moment someone comes to mind.\n\n If no one comes to mind, press 'CTRL' to move onto the next name.\n\n",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    survey_instrx_1_resp = keyboard.Keyboard(deviceName='survey_instrx_1_resp')
    survey_instrx_1_cont = visual.TextStim(win=win, name='survey_instrx_1_cont',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "survey_instrx_2" ---
    survey_instrx_2_txt = visual.TextStim(win=win, name='survey_instrx_2_txt',
        text="If you press 'SPACE,' you will answer two questions using the scales provided:\n\n1. How easily did a person come to mind?\n2. How many people with this name come to mind?\n\nBefore we begin, you will practice with an example name. Remember: Press SPACE as soon as you think of someone with that name!\n",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    survey_instrx_2_resp = keyboard.Keyboard(deviceName='survey_instrx_2_resp')
    survey_instrx_2_cont = visual.TextStim(win=win, name='survey_instrx_2_cont',
        text='Press the spacebar to continue.',
        font='Arial',
        pos=(0, -.40), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "name_prac" ---
    survey_prac_isi = visual.TextStim(win=win, name='survey_prac_isi',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    name_prac_text = visual.TextStim(win=win, name='name_prac_text',
        text='Barack\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    survey_prac_RT = keyboard.Keyboard(deviceName='survey_prac_RT')
    survey_prac_moveon = visual.TextStim(win=win, name='survey_prac_moveon',
        text='Reminder:\n\nPress SPACE if you know someone with that name.\n\nPress CTRL if you do not know anyone with that name.',
        font='Arial',
        pos=(0, -.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "survey_prac" ---
    name_survey_prac = visual.TextStim(win=win, name='name_survey_prac',
        text='',
        font='Arial',
        pos=(0, .3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    HowMany_prac_slider = visual.Slider(win=win, name='HowMany_prac_slider',
        startValue=None, size=(1.0, 0.05), pos=(0, 0.0), units=win.units,
        labels=["0", "1-2", "3-4", "5-6", "6+"], ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    HowEasy_prac_slider = visual.Slider(win=win, name='HowEasy_prac_slider',
        startValue=None, size=(1.0, 0.05), pos=(0, -0.3), units=win.units,
        labels=["not at all", "", "", "", "extremely"], ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    HowEasy_prac_text = visual.TextStim(win=win, name='HowEasy_prac_text',
        text='How easily does someone with this name come to mind?',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    HowMany_prac_text = visual.TextStim(win=win, name='HowMany_prac_text',
        text='How many people come to mind with this name?',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    survey_prac_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0.75, -0.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='survey_prac_cont',
        depth=-6
    )
    survey_prac_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "name_display" ---
    survey_isi = visual.TextStim(win=win, name='survey_isi',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    name_count = visual.TextStim(win=win, name='name_count',
        text='',
        font='Arial',
        pos=(-.7,.4), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    name_text = visual.TextStim(win=win, name='name_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    survey_RT = keyboard.Keyboard(deviceName='survey_RT')
    survey_move_on = visual.TextStim(win=win, name='survey_move_on',
        text='Reminder:\n\nPress SPACE if you know someone with that name.\n\nPress CTRL if you do not know anyone with that name.',
        font='Arial',
        pos=(0, -.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "survey" ---
    name_text2 = visual.TextStim(win=win, name='name_text2',
        text='',
        font='Arial',
        pos=(0, .3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    name_count2 = visual.TextStim(win=win, name='name_count2',
        text='',
        font='Arial',
        pos=(-.7,.4), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    HowMany_slider = visual.Slider(win=win, name='HowMany_slider',
        startValue=None, size=(1.0, 0.05), pos=(0, 0.0), units=win.units,
        labels=["0", "1-2", "3-4", "5-6", "6+"], ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    HowEasy_slider = visual.Slider(win=win, name='HowEasy_slider',
        startValue=None, size=(1.0, 0.05), pos=(0, -0.3), units=win.units,
        labels=["not at all", "", "", "", "extremely"], ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    HowEasy_test = visual.TextStim(win=win, name='HowEasy_test',
        text='How easily does someone with this name come to mind?',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    HowMany_text = visual.TextStim(win=win, name='HowMany_text',
        text='How many people come to mind with this name?',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    survey_cont = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0.75, -0.4),
        letterHeight=0.025,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='survey_cont',
        depth=-7
    )
    survey_cont.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "end" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='Thanks for participating! Please go get your experimenter.\n\nPress the spacebar to end.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_resp = keyboard.Keyboard(deviceName='end_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "start" ---
    # create an object to store info about Routine start
    start = data.Routine(
        name='start',
        components=[begin_experiment, begin_resp],
    )
    start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for begin_resp
    begin_resp.keys = []
    begin_resp.rt = []
    _begin_resp_allKeys = []
    # store start times for start
    start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start.tStart = globalClock.getTime(format='float')
    start.status = STARTED
    start.maxDuration = None
    # keep track of which components have finished
    startComponents = start.components
    for thisComponent in start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start" ---
    start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *begin_experiment* updates
        
        # if begin_experiment is starting this frame...
        if begin_experiment.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            begin_experiment.frameNStart = frameN  # exact frame index
            begin_experiment.tStart = t  # local t and not account for scr refresh
            begin_experiment.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(begin_experiment, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'begin_experiment.started')
            # update status
            begin_experiment.status = STARTED
            begin_experiment.setAutoDraw(True)
        
        # if begin_experiment is active this frame...
        if begin_experiment.status == STARTED:
            # update params
            pass
        
        # *begin_resp* updates
        waitOnFlip = False
        
        # if begin_resp is starting this frame...
        if begin_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            begin_resp.frameNStart = frameN  # exact frame index
            begin_resp.tStart = t  # local t and not account for scr refresh
            begin_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(begin_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'begin_resp.started')
            # update status
            begin_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(begin_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(begin_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if begin_resp.status == STARTED and not waitOnFlip:
            theseKeys = begin_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _begin_resp_allKeys.extend(theseKeys)
            if len(_begin_resp_allKeys):
                begin_resp.keys = _begin_resp_allKeys[-1].name  # just the last key pressed
                begin_resp.rt = _begin_resp_allKeys[-1].rt
                begin_resp.duration = _begin_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=start,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start" ---
    for thisComponent in start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start
    start.tStop = globalClock.getTime(format='float')
    start.tStopRefresh = tThisFlipGlobal
    # check responses
    if begin_resp.keys in ['', [], None]:  # No response was made
        begin_resp.keys = None
    thisExp.addData('begin_resp.keys',begin_resp.keys)
    if begin_resp.keys != None:  # we had a response
        thisExp.addData('begin_resp.rt', begin_resp.rt)
        thisExp.addData('begin_resp.duration', begin_resp.duration)
    thisExp.nextEntry()
    # the Routine "start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    consent_trials = data.TrialHandler2(
        name='consent_trials',
        nReps=nForms, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(consent_trials)  # add the loop to the experiment
    thisConsent_trial = consent_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisConsent_trial.rgb)
    if thisConsent_trial != None:
        for paramName in thisConsent_trial:
            globals()[paramName] = thisConsent_trial[paramName]
    
    for thisConsent_trial in consent_trials:
        consent_trials.status = STARTED
        if hasattr(thisConsent_trial, 'status'):
            thisConsent_trial.status = STARTED
        currentLoop = consent_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisConsent_trial.rgb)
        if thisConsent_trial != None:
            for paramName in thisConsent_trial:
                globals()[paramName] = thisConsent_trial[paramName]
        
        # --- Prepare to start Routine "consent" ---
        # create an object to store info about Routine consent
        consent = data.Routine(
            name='consent',
            components=[consent_img, consent_cont, mouse, yes_continue],
        )
        consent.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from load_consent
        win.mouseVisible = True
        
        # Input the current form
        current_form = consent_files[consent_trials.thisN]
        consent_img.setImage(current_form)
        
        # Get the text to read out
        if consent_trials.thisN < (nForms - 1):
            yes_continue.setText("Next Page")
        else:
            yes_continue.setText("Yes - I agree to participate")
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # store start times for consent
        consent.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        consent.tStart = globalClock.getTime(format='float')
        consent.status = STARTED
        consent.maxDuration = None
        # keep track of which components have finished
        consentComponents = consent.components
        for thisComponent in consent.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "consent" ---
        consent.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisConsent_trial, 'status') and thisConsent_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *consent_img* updates
            
            # if consent_img is starting this frame...
            if consent_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                consent_img.frameNStart = frameN  # exact frame index
                consent_img.tStart = t  # local t and not account for scr refresh
                consent_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(consent_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'consent_img.started')
                # update status
                consent_img.status = STARTED
                consent_img.setAutoDraw(True)
            
            # if consent_img is active this frame...
            if consent_img.status == STARTED:
                # update params
                pass
            
            # *consent_cont* updates
            
            # if consent_cont is starting this frame...
            if consent_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                consent_cont.frameNStart = frameN  # exact frame index
                consent_cont.tStart = t  # local t and not account for scr refresh
                consent_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(consent_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'consent_cont.started')
                # update status
                consent_cont.status = STARTED
                consent_cont.setAutoDraw(True)
            
            # if consent_cont is active this frame...
            if consent_cont.status == STARTED:
                # update params
                pass
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(consent_cont, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse):
                                gotValidClick = True
                                mouse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            mouse.clicked_name.append(None)
                        x, y = mouse.getPos()
                        mouse.x.append(x)
                        mouse.y.append(y)
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # *yes_continue* updates
            
            # if yes_continue is starting this frame...
            if yes_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                yes_continue.frameNStart = frameN  # exact frame index
                yes_continue.tStart = t  # local t and not account for scr refresh
                yes_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yes_continue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yes_continue.started')
                # update status
                yes_continue.status = STARTED
                yes_continue.setAutoDraw(True)
            
            # if yes_continue is active this frame...
            if yes_continue.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=consent,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                consent.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in consent.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "consent" ---
        for thisComponent in consent.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for consent
        consent.tStop = globalClock.getTime(format='float')
        consent.tStopRefresh = tThisFlipGlobal
        # store data for consent_trials (TrialHandler)
        consent_trials.addData('mouse.x', mouse.x)
        consent_trials.addData('mouse.y', mouse.y)
        consent_trials.addData('mouse.leftButton', mouse.leftButton)
        consent_trials.addData('mouse.midButton', mouse.midButton)
        consent_trials.addData('mouse.rightButton', mouse.rightButton)
        consent_trials.addData('mouse.time', mouse.time)
        consent_trials.addData('mouse.clicked_name', mouse.clicked_name)
        # the Routine "consent" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisConsent_trial as finished
        if hasattr(thisConsent_trial, 'status'):
            thisConsent_trial.status = FINISHED
        # if awaiting a pause, pause now
        if consent_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            consent_trials.status = STARTED
    # completed nForms repeats of 'consent_trials'
    consent_trials.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    block_loop = data.TrialHandler2(
        name='block_loop',
        nReps=4.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(block_loop)  # add the loop to the experiment
    thisBlock_loop = block_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
    if thisBlock_loop != None:
        for paramName in thisBlock_loop:
            globals()[paramName] = thisBlock_loop[paramName]
    
    for thisBlock_loop in block_loop:
        block_loop.status = STARTED
        if hasattr(thisBlock_loop, 'status'):
            thisBlock_loop.status = STARTED
        currentLoop = block_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
        if thisBlock_loop != None:
            for paramName in thisBlock_loop:
                globals()[paramName] = thisBlock_loop[paramName]
        
        # --- Prepare to start Routine "set_block" ---
        # create an object to store info about Routine set_block
        set_block = data.Routine(
            name='set_block',
            components=[block_text, cont_resp],
        )
        set_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from blocks
        #update block numbers, show block num text to participant
        currentBlock = block_loop.thisN + 1
        blockText = f"You will now begin block {currentBlock} of 4.\n\nPress the space bar to continue."
        block_text.setText(blockText
        )
        # create starting attributes for cont_resp
        cont_resp.keys = []
        cont_resp.rt = []
        _cont_resp_allKeys = []
        # store start times for set_block
        set_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        set_block.tStart = globalClock.getTime(format='float')
        set_block.status = STARTED
        set_block.maxDuration = None
        # keep track of which components have finished
        set_blockComponents = set_block.components
        for thisComponent in set_block.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "set_block" ---
        set_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_text* updates
            
            # if block_text is starting this frame...
            if block_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_text.frameNStart = frameN  # exact frame index
                block_text.tStart = t  # local t and not account for scr refresh
                block_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_text.started')
                # update status
                block_text.status = STARTED
                block_text.setAutoDraw(True)
            
            # if block_text is active this frame...
            if block_text.status == STARTED:
                # update params
                pass
            
            # *cont_resp* updates
            waitOnFlip = False
            
            # if cont_resp is starting this frame...
            if cont_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cont_resp.frameNStart = frameN  # exact frame index
                cont_resp.tStart = t  # local t and not account for scr refresh
                cont_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cont_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cont_resp.started')
                # update status
                cont_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(cont_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(cont_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if cont_resp.status == STARTED and not waitOnFlip:
                theseKeys = cont_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _cont_resp_allKeys.extend(theseKeys)
                if len(_cont_resp_allKeys):
                    cont_resp.keys = _cont_resp_allKeys[-1].name  # just the last key pressed
                    cont_resp.rt = _cont_resp_allKeys[-1].rt
                    cont_resp.duration = _cont_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=set_block,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                set_block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in set_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "set_block" ---
        for thisComponent in set_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for set_block
        set_block.tStop = globalClock.getTime(format='float')
        set_block.tStopRefresh = tThisFlipGlobal
        # check responses
        if cont_resp.keys in ['', [], None]:  # No response was made
            cont_resp.keys = None
        block_loop.addData('cont_resp.keys',cont_resp.keys)
        if cont_resp.keys != None:  # we had a response
            block_loop.addData('cont_resp.rt', cont_resp.rt)
            block_loop.addData('cont_resp.duration', cont_resp.duration)
        # the Routine "set_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "listA_instrx" ---
        # create an object to store info about Routine listA_instrx
        listA_instrx = data.Routine(
            name='listA_instrx',
            components=[listA_instrx_text, listA_instrx_resp, listA_instrx_cont],
        )
        listA_instrx.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for listA_instrx_resp
        listA_instrx_resp.keys = []
        listA_instrx_resp.rt = []
        _listA_instrx_resp_allKeys = []
        # store start times for listA_instrx
        listA_instrx.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        listA_instrx.tStart = globalClock.getTime(format='float')
        listA_instrx.status = STARTED
        listA_instrx.maxDuration = None
        # keep track of which components have finished
        listA_instrxComponents = listA_instrx.components
        for thisComponent in listA_instrx.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "listA_instrx" ---
        listA_instrx.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *listA_instrx_text* updates
            
            # if listA_instrx_text is starting this frame...
            if listA_instrx_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                listA_instrx_text.frameNStart = frameN  # exact frame index
                listA_instrx_text.tStart = t  # local t and not account for scr refresh
                listA_instrx_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listA_instrx_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listA_instrx_text.started')
                # update status
                listA_instrx_text.status = STARTED
                listA_instrx_text.setAutoDraw(True)
            
            # if listA_instrx_text is active this frame...
            if listA_instrx_text.status == STARTED:
                # update params
                pass
            
            # *listA_instrx_resp* updates
            waitOnFlip = False
            
            # if listA_instrx_resp is starting this frame...
            if listA_instrx_resp.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                listA_instrx_resp.frameNStart = frameN  # exact frame index
                listA_instrx_resp.tStart = t  # local t and not account for scr refresh
                listA_instrx_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listA_instrx_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listA_instrx_resp.started')
                # update status
                listA_instrx_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(listA_instrx_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(listA_instrx_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if listA_instrx_resp.status == STARTED and not waitOnFlip:
                theseKeys = listA_instrx_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _listA_instrx_resp_allKeys.extend(theseKeys)
                if len(_listA_instrx_resp_allKeys):
                    listA_instrx_resp.keys = _listA_instrx_resp_allKeys[-1].name  # just the last key pressed
                    listA_instrx_resp.rt = _listA_instrx_resp_allKeys[-1].rt
                    listA_instrx_resp.duration = _listA_instrx_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *listA_instrx_cont* updates
            
            # if listA_instrx_cont is starting this frame...
            if listA_instrx_cont.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                listA_instrx_cont.frameNStart = frameN  # exact frame index
                listA_instrx_cont.tStart = t  # local t and not account for scr refresh
                listA_instrx_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listA_instrx_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listA_instrx_cont.started')
                # update status
                listA_instrx_cont.status = STARTED
                listA_instrx_cont.setAutoDraw(True)
            
            # if listA_instrx_cont is active this frame...
            if listA_instrx_cont.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=listA_instrx,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                listA_instrx.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in listA_instrx.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "listA_instrx" ---
        for thisComponent in listA_instrx.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for listA_instrx
        listA_instrx.tStop = globalClock.getTime(format='float')
        listA_instrx.tStopRefresh = tThisFlipGlobal
        # check responses
        if listA_instrx_resp.keys in ['', [], None]:  # No response was made
            listA_instrx_resp.keys = None
        block_loop.addData('listA_instrx_resp.keys',listA_instrx_resp.keys)
        if listA_instrx_resp.keys != None:  # we had a response
            block_loop.addData('listA_instrx_resp.rt', listA_instrx_resp.rt)
            block_loop.addData('listA_instrx_resp.duration', listA_instrx_resp.duration)
        # the Routine "listA_instrx" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "begin_listA" ---
        # create an object to store info about Routine begin_listA
        begin_listA = data.Routine(
            name='begin_listA',
            components=[ListA_text],
        )
        begin_listA.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for begin_listA
        begin_listA.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        begin_listA.tStart = globalClock.getTime(format='float')
        begin_listA.status = STARTED
        begin_listA.maxDuration = None
        # keep track of which components have finished
        begin_listAComponents = begin_listA.components
        for thisComponent in begin_listA.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "begin_listA" ---
        begin_listA.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ListA_text* updates
            
            # if ListA_text is starting this frame...
            if ListA_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ListA_text.frameNStart = frameN  # exact frame index
                ListA_text.tStart = t  # local t and not account for scr refresh
                ListA_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ListA_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ListA_text.started')
                # update status
                ListA_text.status = STARTED
                ListA_text.setAutoDraw(True)
            
            # if ListA_text is active this frame...
            if ListA_text.status == STARTED:
                # update params
                pass
            
            # if ListA_text is stopping this frame...
            if ListA_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ListA_text.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    ListA_text.tStop = t  # not accounting for scr refresh
                    ListA_text.tStopRefresh = tThisFlipGlobal  # on global time
                    ListA_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ListA_text.stopped')
                    # update status
                    ListA_text.status = FINISHED
                    ListA_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=begin_listA,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                begin_listA.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in begin_listA.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "begin_listA" ---
        for thisComponent in begin_listA.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for begin_listA
        begin_listA.tStop = globalClock.getTime(format='float')
        begin_listA.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if begin_listA.maxDurationReached:
            routineTimer.addTime(-begin_listA.maxDuration)
        elif begin_listA.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # set up handler to look after randomisation of conditions etc
        listA_loop = data.TrialHandler2(
            name='listA_loop',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(listA_file), 
            seed=None, 
        )
        thisExp.addLoop(listA_loop)  # add the loop to the experiment
        thisListA_loop = listA_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisListA_loop.rgb)
        if thisListA_loop != None:
            for paramName in thisListA_loop:
                globals()[paramName] = thisListA_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisListA_loop in listA_loop:
            listA_loop.status = STARTED
            if hasattr(thisListA_loop, 'status'):
                thisListA_loop.status = STARTED
            currentLoop = listA_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisListA_loop.rgb)
            if thisListA_loop != None:
                for paramName in thisListA_loop:
                    globals()[paramName] = thisListA_loop[paramName]
            
            # --- Prepare to start Routine "listA" ---
            # create an object to store info about Routine listA
            listA = data.Routine(
                name='listA',
                components=[fix_crossA, lisA_image, listA_name],
            )
            listA.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from listA_blocks
            #use corresponding block
            currentBlock = block_loop.thisN + 1
            lisA_image.setImage(img_path)
            listA_name.setText(name)
            # store start times for listA
            listA.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            listA.tStart = globalClock.getTime(format='float')
            listA.status = STARTED
            thisExp.addData('listA.started', listA.tStart)
            listA.maxDuration = None
            # keep track of which components have finished
            listAComponents = listA.components
            for thisComponent in listA.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "listA" ---
            listA.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 6.0:
                # if trial has changed, end Routine now
                if hasattr(thisListA_loop, 'status') and thisListA_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from listA_blocks
                continueRoutine = (block == currentBlock)
                
                # *fix_crossA* updates
                
                # if fix_crossA is starting this frame...
                if fix_crossA.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix_crossA.frameNStart = frameN  # exact frame index
                    fix_crossA.tStart = t  # local t and not account for scr refresh
                    fix_crossA.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix_crossA, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_crossA.started')
                    # update status
                    fix_crossA.status = STARTED
                    fix_crossA.setAutoDraw(True)
                
                # if fix_crossA is active this frame...
                if fix_crossA.status == STARTED:
                    # update params
                    pass
                
                # if fix_crossA is stopping this frame...
                if fix_crossA.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fix_crossA.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        fix_crossA.tStop = t  # not accounting for scr refresh
                        fix_crossA.tStopRefresh = tThisFlipGlobal  # on global time
                        fix_crossA.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fix_crossA.stopped')
                        # update status
                        fix_crossA.status = FINISHED
                        fix_crossA.setAutoDraw(False)
                
                # *lisA_image* updates
                
                # if lisA_image is starting this frame...
                if lisA_image.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    lisA_image.frameNStart = frameN  # exact frame index
                    lisA_image.tStart = t  # local t and not account for scr refresh
                    lisA_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(lisA_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'lisA_image.started')
                    # update status
                    lisA_image.status = STARTED
                    lisA_image.setAutoDraw(True)
                
                # if lisA_image is active this frame...
                if lisA_image.status == STARTED:
                    # update params
                    pass
                
                # if lisA_image is stopping this frame...
                if lisA_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > lisA_image.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        lisA_image.tStop = t  # not accounting for scr refresh
                        lisA_image.tStopRefresh = tThisFlipGlobal  # on global time
                        lisA_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'lisA_image.stopped')
                        # update status
                        lisA_image.status = FINISHED
                        lisA_image.setAutoDraw(False)
                
                # *listA_name* updates
                
                # if listA_name is starting this frame...
                if listA_name.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    listA_name.frameNStart = frameN  # exact frame index
                    listA_name.tStart = t  # local t and not account for scr refresh
                    listA_name.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(listA_name, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'listA_name.started')
                    # update status
                    listA_name.status = STARTED
                    listA_name.setAutoDraw(True)
                
                # if listA_name is active this frame...
                if listA_name.status == STARTED:
                    # update params
                    pass
                
                # if listA_name is stopping this frame...
                if listA_name.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > listA_name.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        listA_name.tStop = t  # not accounting for scr refresh
                        listA_name.tStopRefresh = tThisFlipGlobal  # on global time
                        listA_name.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'listA_name.stopped')
                        # update status
                        listA_name.status = FINISHED
                        listA_name.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=listA,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    listA.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in listA.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "listA" ---
            for thisComponent in listA.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for listA
            listA.tStop = globalClock.getTime(format='float')
            listA.tStopRefresh = tThisFlipGlobal
            thisExp.addData('listA.stopped', listA.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if listA.maxDurationReached:
                routineTimer.addTime(-listA.maxDuration)
            elif listA.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-6.000000)
            # mark thisListA_loop as finished
            if hasattr(thisListA_loop, 'status'):
                thisListA_loop.status = FINISHED
            # if awaiting a pause, pause now
            if listA_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                listA_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'listA_loop'
        listA_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "listB_instrx_1" ---
        # create an object to store info about Routine listB_instrx_1
        listB_instrx_1 = data.Routine(
            name='listB_instrx_1',
            components=[listB_instrx_1_txt, listB_instrx_1_resp, listB_instrx_1_cont],
        )
        listB_instrx_1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for listB_instrx_1_resp
        listB_instrx_1_resp.keys = []
        listB_instrx_1_resp.rt = []
        _listB_instrx_1_resp_allKeys = []
        # store start times for listB_instrx_1
        listB_instrx_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        listB_instrx_1.tStart = globalClock.getTime(format='float')
        listB_instrx_1.status = STARTED
        listB_instrx_1.maxDuration = None
        # keep track of which components have finished
        listB_instrx_1Components = listB_instrx_1.components
        for thisComponent in listB_instrx_1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "listB_instrx_1" ---
        listB_instrx_1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *listB_instrx_1_txt* updates
            
            # if listB_instrx_1_txt is starting this frame...
            if listB_instrx_1_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                listB_instrx_1_txt.frameNStart = frameN  # exact frame index
                listB_instrx_1_txt.tStart = t  # local t and not account for scr refresh
                listB_instrx_1_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listB_instrx_1_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listB_instrx_1_txt.started')
                # update status
                listB_instrx_1_txt.status = STARTED
                listB_instrx_1_txt.setAutoDraw(True)
            
            # if listB_instrx_1_txt is active this frame...
            if listB_instrx_1_txt.status == STARTED:
                # update params
                pass
            
            # *listB_instrx_1_resp* updates
            waitOnFlip = False
            
            # if listB_instrx_1_resp is starting this frame...
            if listB_instrx_1_resp.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                listB_instrx_1_resp.frameNStart = frameN  # exact frame index
                listB_instrx_1_resp.tStart = t  # local t and not account for scr refresh
                listB_instrx_1_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listB_instrx_1_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listB_instrx_1_resp.started')
                # update status
                listB_instrx_1_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(listB_instrx_1_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(listB_instrx_1_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if listB_instrx_1_resp.status == STARTED and not waitOnFlip:
                theseKeys = listB_instrx_1_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _listB_instrx_1_resp_allKeys.extend(theseKeys)
                if len(_listB_instrx_1_resp_allKeys):
                    listB_instrx_1_resp.keys = _listB_instrx_1_resp_allKeys[-1].name  # just the last key pressed
                    listB_instrx_1_resp.rt = _listB_instrx_1_resp_allKeys[-1].rt
                    listB_instrx_1_resp.duration = _listB_instrx_1_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *listB_instrx_1_cont* updates
            
            # if listB_instrx_1_cont is starting this frame...
            if listB_instrx_1_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                listB_instrx_1_cont.frameNStart = frameN  # exact frame index
                listB_instrx_1_cont.tStart = t  # local t and not account for scr refresh
                listB_instrx_1_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listB_instrx_1_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listB_instrx_1_cont.started')
                # update status
                listB_instrx_1_cont.status = STARTED
                listB_instrx_1_cont.setAutoDraw(True)
            
            # if listB_instrx_1_cont is active this frame...
            if listB_instrx_1_cont.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=listB_instrx_1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                listB_instrx_1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in listB_instrx_1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "listB_instrx_1" ---
        for thisComponent in listB_instrx_1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for listB_instrx_1
        listB_instrx_1.tStop = globalClock.getTime(format='float')
        listB_instrx_1.tStopRefresh = tThisFlipGlobal
        # check responses
        if listB_instrx_1_resp.keys in ['', [], None]:  # No response was made
            listB_instrx_1_resp.keys = None
        block_loop.addData('listB_instrx_1_resp.keys',listB_instrx_1_resp.keys)
        if listB_instrx_1_resp.keys != None:  # we had a response
            block_loop.addData('listB_instrx_1_resp.rt', listB_instrx_1_resp.rt)
            block_loop.addData('listB_instrx_1_resp.duration', listB_instrx_1_resp.duration)
        # the Routine "listB_instrx_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "listB_instrx_2" ---
        # create an object to store info about Routine listB_instrx_2
        listB_instrx_2 = data.Routine(
            name='listB_instrx_2',
            components=[listB_instrx_2_txt, listB_instrx_2_resp, listB_instrx_2_cont],
        )
        listB_instrx_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for listB_instrx_2_resp
        listB_instrx_2_resp.keys = []
        listB_instrx_2_resp.rt = []
        _listB_instrx_2_resp_allKeys = []
        # store start times for listB_instrx_2
        listB_instrx_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        listB_instrx_2.tStart = globalClock.getTime(format='float')
        listB_instrx_2.status = STARTED
        listB_instrx_2.maxDuration = None
        # keep track of which components have finished
        listB_instrx_2Components = listB_instrx_2.components
        for thisComponent in listB_instrx_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "listB_instrx_2" ---
        listB_instrx_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *listB_instrx_2_txt* updates
            
            # if listB_instrx_2_txt is starting this frame...
            if listB_instrx_2_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                listB_instrx_2_txt.frameNStart = frameN  # exact frame index
                listB_instrx_2_txt.tStart = t  # local t and not account for scr refresh
                listB_instrx_2_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listB_instrx_2_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listB_instrx_2_txt.started')
                # update status
                listB_instrx_2_txt.status = STARTED
                listB_instrx_2_txt.setAutoDraw(True)
            
            # if listB_instrx_2_txt is active this frame...
            if listB_instrx_2_txt.status == STARTED:
                # update params
                pass
            
            # *listB_instrx_2_resp* updates
            waitOnFlip = False
            
            # if listB_instrx_2_resp is starting this frame...
            if listB_instrx_2_resp.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                listB_instrx_2_resp.frameNStart = frameN  # exact frame index
                listB_instrx_2_resp.tStart = t  # local t and not account for scr refresh
                listB_instrx_2_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listB_instrx_2_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listB_instrx_2_resp.started')
                # update status
                listB_instrx_2_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(listB_instrx_2_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(listB_instrx_2_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if listB_instrx_2_resp.status == STARTED and not waitOnFlip:
                theseKeys = listB_instrx_2_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _listB_instrx_2_resp_allKeys.extend(theseKeys)
                if len(_listB_instrx_2_resp_allKeys):
                    listB_instrx_2_resp.keys = _listB_instrx_2_resp_allKeys[-1].name  # just the last key pressed
                    listB_instrx_2_resp.rt = _listB_instrx_2_resp_allKeys[-1].rt
                    listB_instrx_2_resp.duration = _listB_instrx_2_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *listB_instrx_2_cont* updates
            
            # if listB_instrx_2_cont is starting this frame...
            if listB_instrx_2_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                listB_instrx_2_cont.frameNStart = frameN  # exact frame index
                listB_instrx_2_cont.tStart = t  # local t and not account for scr refresh
                listB_instrx_2_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(listB_instrx_2_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'listB_instrx_2_cont.started')
                # update status
                listB_instrx_2_cont.status = STARTED
                listB_instrx_2_cont.setAutoDraw(True)
            
            # if listB_instrx_2_cont is active this frame...
            if listB_instrx_2_cont.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=listB_instrx_2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                listB_instrx_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in listB_instrx_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "listB_instrx_2" ---
        for thisComponent in listB_instrx_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for listB_instrx_2
        listB_instrx_2.tStop = globalClock.getTime(format='float')
        listB_instrx_2.tStopRefresh = tThisFlipGlobal
        # check responses
        if listB_instrx_2_resp.keys in ['', [], None]:  # No response was made
            listB_instrx_2_resp.keys = None
        block_loop.addData('listB_instrx_2_resp.keys',listB_instrx_2_resp.keys)
        if listB_instrx_2_resp.keys != None:  # we had a response
            block_loop.addData('listB_instrx_2_resp.rt', listB_instrx_2_resp.rt)
            block_loop.addData('listB_instrx_2_resp.duration', listB_instrx_2_resp.duration)
        # the Routine "listB_instrx_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_setup" ---
        # create an object to store info about Routine prac_setup
        prac_setup = data.Routine(
            name='prac_setup',
            components=[prac_text, prac_resp],
        )
        prac_setup.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_text.setText('')
        # create starting attributes for prac_resp
        prac_resp.keys = []
        prac_resp.rt = []
        _prac_resp_allKeys = []
        # Run 'Begin Routine' code from prac_YesNo
        #begin experiment with practice trials instructions, optional after.
        if block_loop.thisN == 0:
            prac_text.setText("To make sure you understand the task, we will show you some practice trials. You must get the practice trials correct to move forward.\n\nPress the 'p' key to continue.")
        else: 
            prac_text.setText("Would you like a reminder of what to do during the task? If so, press the 'p' key. If not, press the 'c' key to continue with the next block.")
            
        # store start times for prac_setup
        prac_setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_setup.tStart = globalClock.getTime(format='float')
        prac_setup.status = STARTED
        prac_setup.maxDuration = None
        # keep track of which components have finished
        prac_setupComponents = prac_setup.components
        for thisComponent in prac_setup.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_setup" ---
        prac_setup.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_text* updates
            
            # if prac_text is starting this frame...
            if prac_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_text.frameNStart = frameN  # exact frame index
                prac_text.tStart = t  # local t and not account for scr refresh
                prac_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_text.started')
                # update status
                prac_text.status = STARTED
                prac_text.setAutoDraw(True)
            
            # if prac_text is active this frame...
            if prac_text.status == STARTED:
                # update params
                pass
            
            # *prac_resp* updates
            waitOnFlip = False
            
            # if prac_resp is starting this frame...
            if prac_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_resp.frameNStart = frameN  # exact frame index
                prac_resp.tStart = t  # local t and not account for scr refresh
                prac_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_resp.started')
                # update status
                prac_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_resp.getKeys(keyList=['p','c'], ignoreKeys=["escape"], waitRelease=False)
                _prac_resp_allKeys.extend(theseKeys)
                if len(_prac_resp_allKeys):
                    prac_resp.keys = _prac_resp_allKeys[-1].name  # just the last key pressed
                    prac_resp.rt = _prac_resp_allKeys[-1].rt
                    prac_resp.duration = _prac_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_setup,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_setup.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_setup.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_setup" ---
        for thisComponent in prac_setup.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_setup
        prac_setup.tStop = globalClock.getTime(format='float')
        prac_setup.tStopRefresh = tThisFlipGlobal
        # check responses
        if prac_resp.keys in ['', [], None]:  # No response was made
            prac_resp.keys = None
        block_loop.addData('prac_resp.keys',prac_resp.keys)
        if prac_resp.keys != None:  # we had a response
            block_loop.addData('prac_resp.rt', prac_resp.rt)
            block_loop.addData('prac_resp.duration', prac_resp.duration)
        # Run 'End Routine' code from prac_YesNo
        #depending on participant response, log answer to continue or skip practice trials
        if 'p' in prac_resp.keys:
            prac_selected = 999
        elif 'c' in prac_resp.keys:
            prac_selected = 0
        # the Routine "prac_setup" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        prac_loop = data.TrialHandler2(
            name='prac_loop',
            nReps=prac_selected, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(prac_loop)  # add the loop to the experiment
        thisPrac_loop = prac_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_loop.rgb)
        if thisPrac_loop != None:
            for paramName in thisPrac_loop:
                globals()[paramName] = thisPrac_loop[paramName]
        
        for thisPrac_loop in prac_loop:
            prac_loop.status = STARTED
            if hasattr(thisPrac_loop, 'status'):
                thisPrac_loop.status = STARTED
            currentLoop = prac_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_loop.rgb)
            if thisPrac_loop != None:
                for paramName in thisPrac_loop:
                    globals()[paramName] = thisPrac_loop[paramName]
            
            # --- Prepare to start Routine "listB_instrx_3" ---
            # create an object to store info about Routine listB_instrx_3
            listB_instrx_3 = data.Routine(
                name='listB_instrx_3',
                components=[listB_instrx_3_txt],
            )
            listB_instrx_3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for listB_instrx_3
            listB_instrx_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            listB_instrx_3.tStart = globalClock.getTime(format='float')
            listB_instrx_3.status = STARTED
            listB_instrx_3.maxDuration = None
            # keep track of which components have finished
            listB_instrx_3Components = listB_instrx_3.components
            for thisComponent in listB_instrx_3.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "listB_instrx_3" ---
            listB_instrx_3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_loop, 'status') and thisPrac_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *listB_instrx_3_txt* updates
                
                # if listB_instrx_3_txt is starting this frame...
                if listB_instrx_3_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    listB_instrx_3_txt.frameNStart = frameN  # exact frame index
                    listB_instrx_3_txt.tStart = t  # local t and not account for scr refresh
                    listB_instrx_3_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(listB_instrx_3_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'listB_instrx_3_txt.started')
                    # update status
                    listB_instrx_3_txt.status = STARTED
                    listB_instrx_3_txt.setAutoDraw(True)
                
                # if listB_instrx_3_txt is active this frame...
                if listB_instrx_3_txt.status == STARTED:
                    # update params
                    pass
                
                # if listB_instrx_3_txt is stopping this frame...
                if listB_instrx_3_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > listB_instrx_3_txt.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        listB_instrx_3_txt.tStop = t  # not accounting for scr refresh
                        listB_instrx_3_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        listB_instrx_3_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'listB_instrx_3_txt.stopped')
                        # update status
                        listB_instrx_3_txt.status = FINISHED
                        listB_instrx_3_txt.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=listB_instrx_3,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    listB_instrx_3.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in listB_instrx_3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "listB_instrx_3" ---
            for thisComponent in listB_instrx_3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for listB_instrx_3
            listB_instrx_3.tStop = globalClock.getTime(format='float')
            listB_instrx_3.tStopRefresh = tThisFlipGlobal
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if listB_instrx_3.maxDurationReached:
                routineTimer.addTime(-listB_instrx_3.maxDuration)
            elif listB_instrx_3.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "pracA" ---
            # create an object to store info about Routine pracA
            pracA = data.Routine(
                name='pracA',
                components=[pracA_image, pracA_name, pracA_isi, pracA_image2, pracA_name2],
            )
            pracA.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for pracA
            pracA.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracA.tStart = globalClock.getTime(format='float')
            pracA.status = STARTED
            thisExp.addData('pracA.started', pracA.tStart)
            pracA.maxDuration = None
            # keep track of which components have finished
            pracAComponents = pracA.components
            for thisComponent in pracA.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracA" ---
            pracA.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 7.0:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_loop, 'status') and thisPrac_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracA_image* updates
                
                # if pracA_image is starting this frame...
                if pracA_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    pracA_image.frameNStart = frameN  # exact frame index
                    pracA_image.tStart = t  # local t and not account for scr refresh
                    pracA_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracA_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracA_image.started')
                    # update status
                    pracA_image.status = STARTED
                    pracA_image.setAutoDraw(True)
                
                # if pracA_image is active this frame...
                if pracA_image.status == STARTED:
                    # update params
                    pass
                
                # if pracA_image is stopping this frame...
                if pracA_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracA_image.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracA_image.tStop = t  # not accounting for scr refresh
                        pracA_image.tStopRefresh = tThisFlipGlobal  # on global time
                        pracA_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracA_image.stopped')
                        # update status
                        pracA_image.status = FINISHED
                        pracA_image.setAutoDraw(False)
                
                # *pracA_name* updates
                
                # if pracA_name is starting this frame...
                if pracA_name.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    pracA_name.frameNStart = frameN  # exact frame index
                    pracA_name.tStart = t  # local t and not account for scr refresh
                    pracA_name.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracA_name, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracA_name.started')
                    # update status
                    pracA_name.status = STARTED
                    pracA_name.setAutoDraw(True)
                
                # if pracA_name is active this frame...
                if pracA_name.status == STARTED:
                    # update params
                    pass
                
                # if pracA_name is stopping this frame...
                if pracA_name.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracA_name.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracA_name.tStop = t  # not accounting for scr refresh
                        pracA_name.tStopRefresh = tThisFlipGlobal  # on global time
                        pracA_name.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracA_name.stopped')
                        # update status
                        pracA_name.status = FINISHED
                        pracA_name.setAutoDraw(False)
                
                # *pracA_isi* updates
                
                # if pracA_isi is starting this frame...
                if pracA_isi.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    pracA_isi.frameNStart = frameN  # exact frame index
                    pracA_isi.tStart = t  # local t and not account for scr refresh
                    pracA_isi.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracA_isi, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracA_isi.started')
                    # update status
                    pracA_isi.status = STARTED
                    pracA_isi.setAutoDraw(True)
                
                # if pracA_isi is active this frame...
                if pracA_isi.status == STARTED:
                    # update params
                    pass
                
                # if pracA_isi is stopping this frame...
                if pracA_isi.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracA_isi.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        pracA_isi.tStop = t  # not accounting for scr refresh
                        pracA_isi.tStopRefresh = tThisFlipGlobal  # on global time
                        pracA_isi.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracA_isi.stopped')
                        # update status
                        pracA_isi.status = FINISHED
                        pracA_isi.setAutoDraw(False)
                
                # *pracA_image2* updates
                
                # if pracA_image2 is starting this frame...
                if pracA_image2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    pracA_image2.frameNStart = frameN  # exact frame index
                    pracA_image2.tStart = t  # local t and not account for scr refresh
                    pracA_image2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracA_image2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracA_image2.started')
                    # update status
                    pracA_image2.status = STARTED
                    pracA_image2.setAutoDraw(True)
                
                # if pracA_image2 is active this frame...
                if pracA_image2.status == STARTED:
                    # update params
                    pass
                
                # if pracA_image2 is stopping this frame...
                if pracA_image2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracA_image2.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracA_image2.tStop = t  # not accounting for scr refresh
                        pracA_image2.tStopRefresh = tThisFlipGlobal  # on global time
                        pracA_image2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracA_image2.stopped')
                        # update status
                        pracA_image2.status = FINISHED
                        pracA_image2.setAutoDraw(False)
                
                # *pracA_name2* updates
                
                # if pracA_name2 is starting this frame...
                if pracA_name2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    pracA_name2.frameNStart = frameN  # exact frame index
                    pracA_name2.tStart = t  # local t and not account for scr refresh
                    pracA_name2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracA_name2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracA_name2.started')
                    # update status
                    pracA_name2.status = STARTED
                    pracA_name2.setAutoDraw(True)
                
                # if pracA_name2 is active this frame...
                if pracA_name2.status == STARTED:
                    # update params
                    pass
                
                # if pracA_name2 is stopping this frame...
                if pracA_name2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracA_name2.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracA_name2.tStop = t  # not accounting for scr refresh
                        pracA_name2.tStopRefresh = tThisFlipGlobal  # on global time
                        pracA_name2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracA_name2.stopped')
                        # update status
                        pracA_name2.status = FINISHED
                        pracA_name2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracA,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracA.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracA.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracA" ---
            for thisComponent in pracA.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracA
            pracA.tStop = globalClock.getTime(format='float')
            pracA.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracA.stopped', pracA.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if pracA.maxDurationReached:
                routineTimer.addTime(-pracA.maxDuration)
            elif pracA.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-7.000000)
            
            # --- Prepare to start Routine "listB_instrx_4" ---
            # create an object to store info about Routine listB_instrx_4
            listB_instrx_4 = data.Routine(
                name='listB_instrx_4',
                components=[listB_instrx_4_txt, listB_instrx_4_resp, listB_instrx_4_cont],
            )
            listB_instrx_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for listB_instrx_4_resp
            listB_instrx_4_resp.keys = []
            listB_instrx_4_resp.rt = []
            _listB_instrx_4_resp_allKeys = []
            # store start times for listB_instrx_4
            listB_instrx_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            listB_instrx_4.tStart = globalClock.getTime(format='float')
            listB_instrx_4.status = STARTED
            thisExp.addData('listB_instrx_4.started', listB_instrx_4.tStart)
            listB_instrx_4.maxDuration = None
            # keep track of which components have finished
            listB_instrx_4Components = listB_instrx_4.components
            for thisComponent in listB_instrx_4.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "listB_instrx_4" ---
            listB_instrx_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_loop, 'status') and thisPrac_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *listB_instrx_4_txt* updates
                
                # if listB_instrx_4_txt is starting this frame...
                if listB_instrx_4_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    listB_instrx_4_txt.frameNStart = frameN  # exact frame index
                    listB_instrx_4_txt.tStart = t  # local t and not account for scr refresh
                    listB_instrx_4_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(listB_instrx_4_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'listB_instrx_4_txt.started')
                    # update status
                    listB_instrx_4_txt.status = STARTED
                    listB_instrx_4_txt.setAutoDraw(True)
                
                # if listB_instrx_4_txt is active this frame...
                if listB_instrx_4_txt.status == STARTED:
                    # update params
                    pass
                
                # *listB_instrx_4_resp* updates
                waitOnFlip = False
                
                # if listB_instrx_4_resp is starting this frame...
                if listB_instrx_4_resp.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    listB_instrx_4_resp.frameNStart = frameN  # exact frame index
                    listB_instrx_4_resp.tStart = t  # local t and not account for scr refresh
                    listB_instrx_4_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(listB_instrx_4_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'listB_instrx_4_resp.started')
                    # update status
                    listB_instrx_4_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(listB_instrx_4_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(listB_instrx_4_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if listB_instrx_4_resp.status == STARTED and not waitOnFlip:
                    theseKeys = listB_instrx_4_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _listB_instrx_4_resp_allKeys.extend(theseKeys)
                    if len(_listB_instrx_4_resp_allKeys):
                        listB_instrx_4_resp.keys = _listB_instrx_4_resp_allKeys[-1].name  # just the last key pressed
                        listB_instrx_4_resp.rt = _listB_instrx_4_resp_allKeys[-1].rt
                        listB_instrx_4_resp.duration = _listB_instrx_4_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *listB_instrx_4_cont* updates
                
                # if listB_instrx_4_cont is starting this frame...
                if listB_instrx_4_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    listB_instrx_4_cont.frameNStart = frameN  # exact frame index
                    listB_instrx_4_cont.tStart = t  # local t and not account for scr refresh
                    listB_instrx_4_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(listB_instrx_4_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'listB_instrx_4_cont.started')
                    # update status
                    listB_instrx_4_cont.status = STARTED
                    listB_instrx_4_cont.setAutoDraw(True)
                
                # if listB_instrx_4_cont is active this frame...
                if listB_instrx_4_cont.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=listB_instrx_4,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    listB_instrx_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in listB_instrx_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "listB_instrx_4" ---
            for thisComponent in listB_instrx_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for listB_instrx_4
            listB_instrx_4.tStop = globalClock.getTime(format='float')
            listB_instrx_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('listB_instrx_4.stopped', listB_instrx_4.tStop)
            # check responses
            if listB_instrx_4_resp.keys in ['', [], None]:  # No response was made
                listB_instrx_4_resp.keys = None
            prac_loop.addData('listB_instrx_4_resp.keys',listB_instrx_4_resp.keys)
            if listB_instrx_4_resp.keys != None:  # we had a response
                prac_loop.addData('listB_instrx_4_resp.rt', listB_instrx_4_resp.rt)
                prac_loop.addData('listB_instrx_4_resp.duration', listB_instrx_4_resp.duration)
            # the Routine "listB_instrx_4" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "pracB1" ---
            # create an object to store info about Routine pracB1
            pracB1 = data.Routine(
                name='pracB1',
                components=[pracB_isi, pracB_text, pracB_image, pracB_name, pracB_verify, pracB_left, pracB_right, LeftB1_resp, RightB1_resp],
            )
            pracB1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # reset LeftB1_resp to account for continued clicks & clear times on/off
            LeftB1_resp.reset()
            # reset RightB1_resp to account for continued clicks & clear times on/off
            RightB1_resp.reset()
            # Run 'Begin Routine' code from pracB_code
            #set blank for participant input
            prac_B1 = None
            # store start times for pracB1
            pracB1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracB1.tStart = globalClock.getTime(format='float')
            pracB1.status = STARTED
            thisExp.addData('pracB1.started', pracB1.tStart)
            pracB1.maxDuration = None
            # keep track of which components have finished
            pracB1Components = pracB1.components
            for thisComponent in pracB1.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracB1" ---
            pracB1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_loop, 'status') and thisPrac_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracB_isi* updates
                
                # if pracB_isi is starting this frame...
                if pracB_isi.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    pracB_isi.frameNStart = frameN  # exact frame index
                    pracB_isi.tStart = t  # local t and not account for scr refresh
                    pracB_isi.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB_isi, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB_isi.started')
                    # update status
                    pracB_isi.status = STARTED
                    pracB_isi.setAutoDraw(True)
                
                # if pracB_isi is active this frame...
                if pracB_isi.status == STARTED:
                    # update params
                    pass
                
                # if pracB_isi is stopping this frame...
                if pracB_isi.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB_isi.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB_isi.tStop = t  # not accounting for scr refresh
                        pracB_isi.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB_isi.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB_isi.stopped')
                        # update status
                        pracB_isi.status = FINISHED
                        pracB_isi.setAutoDraw(False)
                
                # *pracB_text* updates
                
                # if pracB_text is starting this frame...
                if pracB_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    pracB_text.frameNStart = frameN  # exact frame index
                    pracB_text.tStart = t  # local t and not account for scr refresh
                    pracB_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB_text.started')
                    # update status
                    pracB_text.status = STARTED
                    pracB_text.setAutoDraw(True)
                
                # if pracB_text is active this frame...
                if pracB_text.status == STARTED:
                    # update params
                    pass
                
                # if pracB_text is stopping this frame...
                if pracB_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB_text.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB_text.tStop = t  # not accounting for scr refresh
                        pracB_text.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB_text.stopped')
                        # update status
                        pracB_text.status = FINISHED
                        pracB_text.setAutoDraw(False)
                
                # *pracB_image* updates
                
                # if pracB_image is starting this frame...
                if pracB_image.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    pracB_image.frameNStart = frameN  # exact frame index
                    pracB_image.tStart = t  # local t and not account for scr refresh
                    pracB_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB_image.started')
                    # update status
                    pracB_image.status = STARTED
                    pracB_image.setAutoDraw(True)
                
                # if pracB_image is active this frame...
                if pracB_image.status == STARTED:
                    # update params
                    pass
                
                # if pracB_image is stopping this frame...
                if pracB_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB_image.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB_image.tStop = t  # not accounting for scr refresh
                        pracB_image.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB_image.stopped')
                        # update status
                        pracB_image.status = FINISHED
                        pracB_image.setAutoDraw(False)
                
                # *pracB_name* updates
                
                # if pracB_name is starting this frame...
                if pracB_name.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    pracB_name.frameNStart = frameN  # exact frame index
                    pracB_name.tStart = t  # local t and not account for scr refresh
                    pracB_name.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB_name, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB_name.started')
                    # update status
                    pracB_name.status = STARTED
                    pracB_name.setAutoDraw(True)
                
                # if pracB_name is active this frame...
                if pracB_name.status == STARTED:
                    # update params
                    pass
                
                # if pracB_name is stopping this frame...
                if pracB_name.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB_name.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB_name.tStop = t  # not accounting for scr refresh
                        pracB_name.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB_name.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB_name.stopped')
                        # update status
                        pracB_name.status = FINISHED
                        pracB_name.setAutoDraw(False)
                
                # *pracB_verify* updates
                
                # if pracB_verify is starting this frame...
                if pracB_verify.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    pracB_verify.frameNStart = frameN  # exact frame index
                    pracB_verify.tStart = t  # local t and not account for scr refresh
                    pracB_verify.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB_verify, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB_verify.started')
                    # update status
                    pracB_verify.status = STARTED
                    pracB_verify.setAutoDraw(True)
                
                # if pracB_verify is active this frame...
                if pracB_verify.status == STARTED:
                    # update params
                    pass
                
                # *pracB_left* updates
                
                # if pracB_left is starting this frame...
                if pracB_left.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    pracB_left.frameNStart = frameN  # exact frame index
                    pracB_left.tStart = t  # local t and not account for scr refresh
                    pracB_left.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB_left, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB_left.started')
                    # update status
                    pracB_left.status = STARTED
                    pracB_left.setAutoDraw(True)
                
                # if pracB_left is active this frame...
                if pracB_left.status == STARTED:
                    # update params
                    pass
                
                # *pracB_right* updates
                
                # if pracB_right is starting this frame...
                if pracB_right.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    pracB_right.frameNStart = frameN  # exact frame index
                    pracB_right.tStart = t  # local t and not account for scr refresh
                    pracB_right.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB_right, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB_right.started')
                    # update status
                    pracB_right.status = STARTED
                    pracB_right.setAutoDraw(True)
                
                # if pracB_right is active this frame...
                if pracB_right.status == STARTED:
                    # update params
                    pass
                # *LeftB1_resp* updates
                
                # if LeftB1_resp is starting this frame...
                if LeftB1_resp.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    LeftB1_resp.frameNStart = frameN  # exact frame index
                    LeftB1_resp.tStart = t  # local t and not account for scr refresh
                    LeftB1_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(LeftB1_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'LeftB1_resp.started')
                    # update status
                    LeftB1_resp.status = STARTED
                    win.callOnFlip(LeftB1_resp.buttonClock.reset)
                    LeftB1_resp.setAutoDraw(True)
                
                # if LeftB1_resp is active this frame...
                if LeftB1_resp.status == STARTED:
                    # update params
                    pass
                    # check whether LeftB1_resp has been pressed
                    if LeftB1_resp.isClicked:
                        if not LeftB1_resp.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            LeftB1_resp.timesOn.append(LeftB1_resp.buttonClock.getTime())
                            LeftB1_resp.timesOff.append(LeftB1_resp.buttonClock.getTime())
                        elif len(LeftB1_resp.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            LeftB1_resp.timesOff[-1] = LeftB1_resp.buttonClock.getTime()
                        if not LeftB1_resp.wasClicked:
                            # end routine when LeftB1_resp is clicked
                            continueRoutine = False
                        if not LeftB1_resp.wasClicked:
                            # run callback code when LeftB1_resp is clicked
                            pass
                # take note of whether LeftB1_resp was clicked, so that next frame we know if clicks are new
                LeftB1_resp.wasClicked = LeftB1_resp.isClicked and LeftB1_resp.status == STARTED
                # *RightB1_resp* updates
                
                # if RightB1_resp is starting this frame...
                if RightB1_resp.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    RightB1_resp.frameNStart = frameN  # exact frame index
                    RightB1_resp.tStart = t  # local t and not account for scr refresh
                    RightB1_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(RightB1_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RightB1_resp.started')
                    # update status
                    RightB1_resp.status = STARTED
                    win.callOnFlip(RightB1_resp.buttonClock.reset)
                    RightB1_resp.setAutoDraw(True)
                
                # if RightB1_resp is active this frame...
                if RightB1_resp.status == STARTED:
                    # update params
                    pass
                    # check whether RightB1_resp has been pressed
                    if RightB1_resp.isClicked:
                        if not RightB1_resp.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            RightB1_resp.timesOn.append(RightB1_resp.buttonClock.getTime())
                            RightB1_resp.timesOff.append(RightB1_resp.buttonClock.getTime())
                        elif len(RightB1_resp.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            RightB1_resp.timesOff[-1] = RightB1_resp.buttonClock.getTime()
                        if not RightB1_resp.wasClicked:
                            # end routine when RightB1_resp is clicked
                            continueRoutine = False
                        if not RightB1_resp.wasClicked:
                            # run callback code when RightB1_resp is clicked
                            pass
                # take note of whether RightB1_resp was clicked, so that next frame we know if clicks are new
                RightB1_resp.wasClicked = RightB1_resp.isClicked and RightB1_resp.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracB1,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracB1.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracB1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracB1" ---
            for thisComponent in pracB1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracB1
            pracB1.tStop = globalClock.getTime(format='float')
            pracB1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracB1.stopped', pracB1.tStop)
            prac_loop.addData('LeftB1_resp.numClicks', LeftB1_resp.numClicks)
            if LeftB1_resp.numClicks:
               prac_loop.addData('LeftB1_resp.timesOn', LeftB1_resp.timesOn)
               prac_loop.addData('LeftB1_resp.timesOff', LeftB1_resp.timesOff)
            else:
               prac_loop.addData('LeftB1_resp.timesOn', "")
               prac_loop.addData('LeftB1_resp.timesOff', "")
            prac_loop.addData('RightB1_resp.numClicks', RightB1_resp.numClicks)
            if RightB1_resp.numClicks:
               prac_loop.addData('RightB1_resp.timesOn', RightB1_resp.timesOn)
               prac_loop.addData('RightB1_resp.timesOff', RightB1_resp.timesOff)
            else:
               prac_loop.addData('RightB1_resp.timesOn', "")
               prac_loop.addData('RightB1_resp.timesOff', "")
            # Run 'End Routine' code from pracB_code
            #set left key as the correct answer for pracB1 trial
            if LeftB1_resp.isClicked:
                prac_B1 = "correct"
            elif RightB1_resp.isClicked:
                prac_B1 = "incorrect"
            
            #add to all answers if correct
            practice_correct.append(prac_B1 == "correct")
            # the Routine "pracB1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "pracB1_feedback" ---
            # create an object to store info about Routine pracB1_feedback
            pracB1_feedback = data.Routine(
                name='pracB1_feedback',
                components=[B1_feedback, B1_cont, prac_cont],
            )
            pracB1_feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from B1_code
            B1 = ""
            #show the following feedback depending on the participants answer.
            if prac_B1 == "correct":
                B1 = "Correct! The 'back' instruction asks you to think back on the face that was previously paired with, in this case, the name Beyonce."
            elif prac_B1 == "incorrect":
                B1 = "Incorrect! The 'back' instruction asks you to think back on the face that was previously paired with, in this case, Beyonce from the first set. The name 'Beyonce' was paired with a blonde haired woman in the first set."
            B1_feedback.setText(B1)
            # create starting attributes for prac_cont
            prac_cont.keys = []
            prac_cont.rt = []
            _prac_cont_allKeys = []
            # store start times for pracB1_feedback
            pracB1_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracB1_feedback.tStart = globalClock.getTime(format='float')
            pracB1_feedback.status = STARTED
            thisExp.addData('pracB1_feedback.started', pracB1_feedback.tStart)
            pracB1_feedback.maxDuration = None
            # keep track of which components have finished
            pracB1_feedbackComponents = pracB1_feedback.components
            for thisComponent in pracB1_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracB1_feedback" ---
            pracB1_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_loop, 'status') and thisPrac_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *B1_feedback* updates
                
                # if B1_feedback is starting this frame...
                if B1_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    B1_feedback.frameNStart = frameN  # exact frame index
                    B1_feedback.tStart = t  # local t and not account for scr refresh
                    B1_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(B1_feedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'B1_feedback.started')
                    # update status
                    B1_feedback.status = STARTED
                    B1_feedback.setAutoDraw(True)
                
                # if B1_feedback is active this frame...
                if B1_feedback.status == STARTED:
                    # update params
                    pass
                
                # *B1_cont* updates
                
                # if B1_cont is starting this frame...
                if B1_cont.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
                    # keep track of start time/frame for later
                    B1_cont.frameNStart = frameN  # exact frame index
                    B1_cont.tStart = t  # local t and not account for scr refresh
                    B1_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(B1_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'B1_cont.started')
                    # update status
                    B1_cont.status = STARTED
                    B1_cont.setAutoDraw(True)
                
                # if B1_cont is active this frame...
                if B1_cont.status == STARTED:
                    # update params
                    pass
                
                # *prac_cont* updates
                waitOnFlip = False
                
                # if prac_cont is starting this frame...
                if prac_cont.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    prac_cont.frameNStart = frameN  # exact frame index
                    prac_cont.tStart = t  # local t and not account for scr refresh
                    prac_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont.started')
                    # update status
                    prac_cont.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_cont.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_cont.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac_cont.status == STARTED and not waitOnFlip:
                    theseKeys = prac_cont.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_cont_allKeys.extend(theseKeys)
                    if len(_prac_cont_allKeys):
                        prac_cont.keys = _prac_cont_allKeys[-1].name  # just the last key pressed
                        prac_cont.rt = _prac_cont_allKeys[-1].rt
                        prac_cont.duration = _prac_cont_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracB1_feedback,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracB1_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracB1_feedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracB1_feedback" ---
            for thisComponent in pracB1_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracB1_feedback
            pracB1_feedback.tStop = globalClock.getTime(format='float')
            pracB1_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracB1_feedback.stopped', pracB1_feedback.tStop)
            # check responses
            if prac_cont.keys in ['', [], None]:  # No response was made
                prac_cont.keys = None
            prac_loop.addData('prac_cont.keys',prac_cont.keys)
            if prac_cont.keys != None:  # we had a response
                prac_loop.addData('prac_cont.rt', prac_cont.rt)
                prac_loop.addData('prac_cont.duration', prac_cont.duration)
            # the Routine "pracB1_feedback" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "pracB2" ---
            # create an object to store info about Routine pracB2
            pracB2 = data.Routine(
                name='pracB2',
                components=[pracB2_isi, pracB2_text, pracB2_image, pracB2_name, prabB2_verify, pracB2_left, pracB2_right, LeftB2_resp, RightB2_resp],
            )
            pracB2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # reset LeftB2_resp to account for continued clicks & clear times on/off
            LeftB2_resp.reset()
            # reset RightB2_resp to account for continued clicks & clear times on/off
            RightB2_resp.reset()
            # Run 'Begin Routine' code from pracB2_code
            #set blank for participant input
            prac_B2 = None
            # store start times for pracB2
            pracB2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracB2.tStart = globalClock.getTime(format='float')
            pracB2.status = STARTED
            thisExp.addData('pracB2.started', pracB2.tStart)
            pracB2.maxDuration = None
            # keep track of which components have finished
            pracB2Components = pracB2.components
            for thisComponent in pracB2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracB2" ---
            pracB2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_loop, 'status') and thisPrac_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracB2_isi* updates
                
                # if pracB2_isi is starting this frame...
                if pracB2_isi.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    pracB2_isi.frameNStart = frameN  # exact frame index
                    pracB2_isi.tStart = t  # local t and not account for scr refresh
                    pracB2_isi.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB2_isi, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB2_isi.started')
                    # update status
                    pracB2_isi.status = STARTED
                    pracB2_isi.setAutoDraw(True)
                
                # if pracB2_isi is active this frame...
                if pracB2_isi.status == STARTED:
                    # update params
                    pass
                
                # if pracB2_isi is stopping this frame...
                if pracB2_isi.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB2_isi.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB2_isi.tStop = t  # not accounting for scr refresh
                        pracB2_isi.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB2_isi.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB2_isi.stopped')
                        # update status
                        pracB2_isi.status = FINISHED
                        pracB2_isi.setAutoDraw(False)
                
                # *pracB2_text* updates
                
                # if pracB2_text is starting this frame...
                if pracB2_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    pracB2_text.frameNStart = frameN  # exact frame index
                    pracB2_text.tStart = t  # local t and not account for scr refresh
                    pracB2_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB2_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB2_text.started')
                    # update status
                    pracB2_text.status = STARTED
                    pracB2_text.setAutoDraw(True)
                
                # if pracB2_text is active this frame...
                if pracB2_text.status == STARTED:
                    # update params
                    pass
                
                # if pracB2_text is stopping this frame...
                if pracB2_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB2_text.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB2_text.tStop = t  # not accounting for scr refresh
                        pracB2_text.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB2_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB2_text.stopped')
                        # update status
                        pracB2_text.status = FINISHED
                        pracB2_text.setAutoDraw(False)
                
                # *pracB2_image* updates
                
                # if pracB2_image is starting this frame...
                if pracB2_image.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    pracB2_image.frameNStart = frameN  # exact frame index
                    pracB2_image.tStart = t  # local t and not account for scr refresh
                    pracB2_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB2_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB2_image.started')
                    # update status
                    pracB2_image.status = STARTED
                    pracB2_image.setAutoDraw(True)
                
                # if pracB2_image is active this frame...
                if pracB2_image.status == STARTED:
                    # update params
                    pass
                
                # if pracB2_image is stopping this frame...
                if pracB2_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB2_image.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB2_image.tStop = t  # not accounting for scr refresh
                        pracB2_image.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB2_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB2_image.stopped')
                        # update status
                        pracB2_image.status = FINISHED
                        pracB2_image.setAutoDraw(False)
                
                # *pracB2_name* updates
                
                # if pracB2_name is starting this frame...
                if pracB2_name.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    pracB2_name.frameNStart = frameN  # exact frame index
                    pracB2_name.tStart = t  # local t and not account for scr refresh
                    pracB2_name.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB2_name, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB2_name.started')
                    # update status
                    pracB2_name.status = STARTED
                    pracB2_name.setAutoDraw(True)
                
                # if pracB2_name is active this frame...
                if pracB2_name.status == STARTED:
                    # update params
                    pass
                
                # if pracB2_name is stopping this frame...
                if pracB2_name.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracB2_name.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        pracB2_name.tStop = t  # not accounting for scr refresh
                        pracB2_name.tStopRefresh = tThisFlipGlobal  # on global time
                        pracB2_name.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracB2_name.stopped')
                        # update status
                        pracB2_name.status = FINISHED
                        pracB2_name.setAutoDraw(False)
                
                # *prabB2_verify* updates
                
                # if prabB2_verify is starting this frame...
                if prabB2_verify.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    prabB2_verify.frameNStart = frameN  # exact frame index
                    prabB2_verify.tStart = t  # local t and not account for scr refresh
                    prabB2_verify.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prabB2_verify, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prabB2_verify.started')
                    # update status
                    prabB2_verify.status = STARTED
                    prabB2_verify.setAutoDraw(True)
                
                # if prabB2_verify is active this frame...
                if prabB2_verify.status == STARTED:
                    # update params
                    pass
                
                # *pracB2_left* updates
                
                # if pracB2_left is starting this frame...
                if pracB2_left.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    pracB2_left.frameNStart = frameN  # exact frame index
                    pracB2_left.tStart = t  # local t and not account for scr refresh
                    pracB2_left.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB2_left, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB2_left.started')
                    # update status
                    pracB2_left.status = STARTED
                    pracB2_left.setAutoDraw(True)
                
                # if pracB2_left is active this frame...
                if pracB2_left.status == STARTED:
                    # update params
                    pass
                
                # *pracB2_right* updates
                
                # if pracB2_right is starting this frame...
                if pracB2_right.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    pracB2_right.frameNStart = frameN  # exact frame index
                    pracB2_right.tStart = t  # local t and not account for scr refresh
                    pracB2_right.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracB2_right, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracB2_right.started')
                    # update status
                    pracB2_right.status = STARTED
                    pracB2_right.setAutoDraw(True)
                
                # if pracB2_right is active this frame...
                if pracB2_right.status == STARTED:
                    # update params
                    pass
                # *LeftB2_resp* updates
                
                # if LeftB2_resp is starting this frame...
                if LeftB2_resp.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    LeftB2_resp.frameNStart = frameN  # exact frame index
                    LeftB2_resp.tStart = t  # local t and not account for scr refresh
                    LeftB2_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(LeftB2_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'LeftB2_resp.started')
                    # update status
                    LeftB2_resp.status = STARTED
                    win.callOnFlip(LeftB2_resp.buttonClock.reset)
                    LeftB2_resp.setAutoDraw(True)
                
                # if LeftB2_resp is active this frame...
                if LeftB2_resp.status == STARTED:
                    # update params
                    pass
                    # check whether LeftB2_resp has been pressed
                    if LeftB2_resp.isClicked:
                        if not LeftB2_resp.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            LeftB2_resp.timesOn.append(LeftB2_resp.buttonClock.getTime())
                            LeftB2_resp.timesOff.append(LeftB2_resp.buttonClock.getTime())
                        elif len(LeftB2_resp.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            LeftB2_resp.timesOff[-1] = LeftB2_resp.buttonClock.getTime()
                        if not LeftB2_resp.wasClicked:
                            # end routine when LeftB2_resp is clicked
                            continueRoutine = False
                        if not LeftB2_resp.wasClicked:
                            # run callback code when LeftB2_resp is clicked
                            pass
                # take note of whether LeftB2_resp was clicked, so that next frame we know if clicks are new
                LeftB2_resp.wasClicked = LeftB2_resp.isClicked and LeftB2_resp.status == STARTED
                # *RightB2_resp* updates
                
                # if RightB2_resp is starting this frame...
                if RightB2_resp.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                    # keep track of start time/frame for later
                    RightB2_resp.frameNStart = frameN  # exact frame index
                    RightB2_resp.tStart = t  # local t and not account for scr refresh
                    RightB2_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(RightB2_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RightB2_resp.started')
                    # update status
                    RightB2_resp.status = STARTED
                    win.callOnFlip(RightB2_resp.buttonClock.reset)
                    RightB2_resp.setAutoDraw(True)
                
                # if RightB2_resp is active this frame...
                if RightB2_resp.status == STARTED:
                    # update params
                    pass
                    # check whether RightB2_resp has been pressed
                    if RightB2_resp.isClicked:
                        if not RightB2_resp.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            RightB2_resp.timesOn.append(RightB2_resp.buttonClock.getTime())
                            RightB2_resp.timesOff.append(RightB2_resp.buttonClock.getTime())
                        elif len(RightB2_resp.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            RightB2_resp.timesOff[-1] = RightB2_resp.buttonClock.getTime()
                        if not RightB2_resp.wasClicked:
                            # end routine when RightB2_resp is clicked
                            continueRoutine = False
                        if not RightB2_resp.wasClicked:
                            # run callback code when RightB2_resp is clicked
                            pass
                # take note of whether RightB2_resp was clicked, so that next frame we know if clicks are new
                RightB2_resp.wasClicked = RightB2_resp.isClicked and RightB2_resp.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracB2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracB2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracB2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracB2" ---
            for thisComponent in pracB2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracB2
            pracB2.tStop = globalClock.getTime(format='float')
            pracB2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracB2.stopped', pracB2.tStop)
            prac_loop.addData('LeftB2_resp.numClicks', LeftB2_resp.numClicks)
            if LeftB2_resp.numClicks:
               prac_loop.addData('LeftB2_resp.timesOn', LeftB2_resp.timesOn)
               prac_loop.addData('LeftB2_resp.timesOff', LeftB2_resp.timesOff)
            else:
               prac_loop.addData('LeftB2_resp.timesOn', "")
               prac_loop.addData('LeftB2_resp.timesOff', "")
            prac_loop.addData('RightB2_resp.numClicks', RightB2_resp.numClicks)
            if RightB2_resp.numClicks:
               prac_loop.addData('RightB2_resp.timesOn', RightB2_resp.timesOn)
               prac_loop.addData('RightB2_resp.timesOff', RightB2_resp.timesOff)
            else:
               prac_loop.addData('RightB2_resp.timesOn', "")
               prac_loop.addData('RightB2_resp.timesOff', "")
            # Run 'End Routine' code from pracB2_code
            #set right key as the correct answer for pracB2 trial
            if LeftB2_resp.isClicked:
                prac_B2 = "incorrect"
            elif RightB2_resp.isClicked:
                prac_B2 = "correct"
            #add to all answers if correct.
            practice_correct.append(prac_B2 == "correct")
            # the Routine "pracB2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "pracB2_feedback" ---
            # create an object to store info about Routine pracB2_feedback
            pracB2_feedback = data.Routine(
                name='pracB2_feedback',
                components=[B2_feedback, B2_cont, prac_cont2],
            )
            pracB2_feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from B2_code
            B2 = ""
            #show the following feedback depending on the participants answer.
            if prac_B2 == "correct":
                B2= "Correct! The 'next' instruction asks you to think of this name and face pair instead of thinking back to the face of the first set with this name."
            elif prac_B2 == "incorrect":
                B2 = "Incorrect! The 'next' instruction asks you to think of this name and face pair instead of thinking back to the face of the first set that was paired with this name."
            B2_feedback.setText(B2
            )
            # create starting attributes for prac_cont2
            prac_cont2.keys = []
            prac_cont2.rt = []
            _prac_cont2_allKeys = []
            # store start times for pracB2_feedback
            pracB2_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracB2_feedback.tStart = globalClock.getTime(format='float')
            pracB2_feedback.status = STARTED
            thisExp.addData('pracB2_feedback.started', pracB2_feedback.tStart)
            pracB2_feedback.maxDuration = None
            # keep track of which components have finished
            pracB2_feedbackComponents = pracB2_feedback.components
            for thisComponent in pracB2_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracB2_feedback" ---
            pracB2_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_loop, 'status') and thisPrac_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *B2_feedback* updates
                
                # if B2_feedback is starting this frame...
                if B2_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    B2_feedback.frameNStart = frameN  # exact frame index
                    B2_feedback.tStart = t  # local t and not account for scr refresh
                    B2_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(B2_feedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'B2_feedback.started')
                    # update status
                    B2_feedback.status = STARTED
                    B2_feedback.setAutoDraw(True)
                
                # if B2_feedback is active this frame...
                if B2_feedback.status == STARTED:
                    # update params
                    pass
                
                # *B2_cont* updates
                
                # if B2_cont is starting this frame...
                if B2_cont.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
                    # keep track of start time/frame for later
                    B2_cont.frameNStart = frameN  # exact frame index
                    B2_cont.tStart = t  # local t and not account for scr refresh
                    B2_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(B2_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'B2_cont.started')
                    # update status
                    B2_cont.status = STARTED
                    B2_cont.setAutoDraw(True)
                
                # if B2_cont is active this frame...
                if B2_cont.status == STARTED:
                    # update params
                    pass
                
                # *prac_cont2* updates
                waitOnFlip = False
                
                # if prac_cont2 is starting this frame...
                if prac_cont2.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_cont2.frameNStart = frameN  # exact frame index
                    prac_cont2.tStart = t  # local t and not account for scr refresh
                    prac_cont2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_cont2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont2.started')
                    # update status
                    prac_cont2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_cont2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_cont2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac_cont2.status == STARTED and not waitOnFlip:
                    theseKeys = prac_cont2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_cont2_allKeys.extend(theseKeys)
                    if len(_prac_cont2_allKeys):
                        prac_cont2.keys = _prac_cont2_allKeys[-1].name  # just the last key pressed
                        prac_cont2.rt = _prac_cont2_allKeys[-1].rt
                        prac_cont2.duration = _prac_cont2_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracB2_feedback,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracB2_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracB2_feedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracB2_feedback" ---
            for thisComponent in pracB2_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracB2_feedback
            pracB2_feedback.tStop = globalClock.getTime(format='float')
            pracB2_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracB2_feedback.stopped', pracB2_feedback.tStop)
            # Run 'End Routine' code from B2_code
            #to move on, check that there are 4 answers and that they are all correct. 
            if len(practice_correct) == 2 and all(practice_correct):
                prac_loop.finished = True
            else:
                prac_loop.finished = False
            
            #clear for next practice trials
            practice_correct.clear()
            
            #set to null for next use, just in case
            prac_selected = None
            # check responses
            if prac_cont2.keys in ['', [], None]:  # No response was made
                prac_cont2.keys = None
            prac_loop.addData('prac_cont2.keys',prac_cont2.keys)
            if prac_cont2.keys != None:  # we had a response
                prac_loop.addData('prac_cont2.rt', prac_cont2.rt)
                prac_loop.addData('prac_cont2.duration', prac_cont2.duration)
            # the Routine "pracB2_feedback" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisPrac_loop as finished
            if hasattr(thisPrac_loop, 'status'):
                thisPrac_loop.status = FINISHED
            # if awaiting a pause, pause now
            if prac_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                prac_loop.status = STARTED
        # completed prac_selected repeats of 'prac_loop'
        prac_loop.status = FINISHED
        
        
        # --- Prepare to start Routine "begin_listB" ---
        # create an object to store info about Routine begin_listB
        begin_listB = data.Routine(
            name='begin_listB',
            components=[ListB_text],
        )
        begin_listB.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for begin_listB
        begin_listB.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        begin_listB.tStart = globalClock.getTime(format='float')
        begin_listB.status = STARTED
        begin_listB.maxDuration = None
        # keep track of which components have finished
        begin_listBComponents = begin_listB.components
        for thisComponent in begin_listB.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "begin_listB" ---
        begin_listB.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ListB_text* updates
            
            # if ListB_text is starting this frame...
            if ListB_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ListB_text.frameNStart = frameN  # exact frame index
                ListB_text.tStart = t  # local t and not account for scr refresh
                ListB_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ListB_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ListB_text.started')
                # update status
                ListB_text.status = STARTED
                ListB_text.setAutoDraw(True)
            
            # if ListB_text is active this frame...
            if ListB_text.status == STARTED:
                # update params
                pass
            
            # if ListB_text is stopping this frame...
            if ListB_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ListB_text.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    ListB_text.tStop = t  # not accounting for scr refresh
                    ListB_text.tStopRefresh = tThisFlipGlobal  # on global time
                    ListB_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ListB_text.stopped')
                    # update status
                    ListB_text.status = FINISHED
                    ListB_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=begin_listB,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                begin_listB.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in begin_listB.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "begin_listB" ---
        for thisComponent in begin_listB.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for begin_listB
        begin_listB.tStop = globalClock.getTime(format='float')
        begin_listB.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if begin_listB.maxDurationReached:
            routineTimer.addTime(-begin_listB.maxDuration)
        elif begin_listB.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # set up handler to look after randomisation of conditions etc
        listB_loop = data.TrialHandler2(
            name='listB_loop',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(listB_file), 
            seed=None, 
        )
        thisExp.addLoop(listB_loop)  # add the loop to the experiment
        thisListB_loop = listB_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisListB_loop.rgb)
        if thisListB_loop != None:
            for paramName in thisListB_loop:
                globals()[paramName] = thisListB_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisListB_loop in listB_loop:
            listB_loop.status = STARTED
            if hasattr(thisListB_loop, 'status'):
                thisListB_loop.status = STARTED
            currentLoop = listB_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisListB_loop.rgb)
            if thisListB_loop != None:
                for paramName in thisListB_loop:
                    globals()[paramName] = thisListB_loop[paramName]
            
            # --- Prepare to start Routine "listB" ---
            # create an object to store info about Routine listB
            listB = data.Routine(
                name='listB',
                components=[fix_crossB, back_next_text, listB_image, listB_name],
            )
            listB.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from listB_blocks
            #use corresponding block
            currentBlock = block_loop.thisN + 1
            back_next_text.setText(back_next)
            listB_image.setImage(img_path)
            listB_name.setText(name
            )
            # store start times for listB
            listB.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            listB.tStart = globalClock.getTime(format='float')
            listB.status = STARTED
            thisExp.addData('listB.started', listB.tStart)
            listB.maxDuration = None
            # keep track of which components have finished
            listBComponents = listB.components
            for thisComponent in listB.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "listB" ---
            listB.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 8.0:
                # if trial has changed, end Routine now
                if hasattr(thisListB_loop, 'status') and thisListB_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from listB_blocks
                continueRoutine = (block == currentBlock)
                
                # *fix_crossB* updates
                
                # if fix_crossB is starting this frame...
                if fix_crossB.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix_crossB.frameNStart = frameN  # exact frame index
                    fix_crossB.tStart = t  # local t and not account for scr refresh
                    fix_crossB.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix_crossB, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_crossB.started')
                    # update status
                    fix_crossB.status = STARTED
                    fix_crossB.setAutoDraw(True)
                
                # if fix_crossB is active this frame...
                if fix_crossB.status == STARTED:
                    # update params
                    pass
                
                # if fix_crossB is stopping this frame...
                if fix_crossB.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fix_crossB.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        fix_crossB.tStop = t  # not accounting for scr refresh
                        fix_crossB.tStopRefresh = tThisFlipGlobal  # on global time
                        fix_crossB.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fix_crossB.stopped')
                        # update status
                        fix_crossB.status = FINISHED
                        fix_crossB.setAutoDraw(False)
                
                # *back_next_text* updates
                
                # if back_next_text is starting this frame...
                if back_next_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    back_next_text.frameNStart = frameN  # exact frame index
                    back_next_text.tStart = t  # local t and not account for scr refresh
                    back_next_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(back_next_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'back_next_text.started')
                    # update status
                    back_next_text.status = STARTED
                    back_next_text.setAutoDraw(True)
                
                # if back_next_text is active this frame...
                if back_next_text.status == STARTED:
                    # update params
                    pass
                
                # if back_next_text is stopping this frame...
                if back_next_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > back_next_text.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        back_next_text.tStop = t  # not accounting for scr refresh
                        back_next_text.tStopRefresh = tThisFlipGlobal  # on global time
                        back_next_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'back_next_text.stopped')
                        # update status
                        back_next_text.status = FINISHED
                        back_next_text.setAutoDraw(False)
                
                # *listB_image* updates
                
                # if listB_image is starting this frame...
                if listB_image.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    listB_image.frameNStart = frameN  # exact frame index
                    listB_image.tStart = t  # local t and not account for scr refresh
                    listB_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(listB_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'listB_image.started')
                    # update status
                    listB_image.status = STARTED
                    listB_image.setAutoDraw(True)
                
                # if listB_image is active this frame...
                if listB_image.status == STARTED:
                    # update params
                    pass
                
                # if listB_image is stopping this frame...
                if listB_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > listB_image.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        listB_image.tStop = t  # not accounting for scr refresh
                        listB_image.tStopRefresh = tThisFlipGlobal  # on global time
                        listB_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'listB_image.stopped')
                        # update status
                        listB_image.status = FINISHED
                        listB_image.setAutoDraw(False)
                
                # *listB_name* updates
                
                # if listB_name is starting this frame...
                if listB_name.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                    # keep track of start time/frame for later
                    listB_name.frameNStart = frameN  # exact frame index
                    listB_name.tStart = t  # local t and not account for scr refresh
                    listB_name.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(listB_name, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'listB_name.started')
                    # update status
                    listB_name.status = STARTED
                    listB_name.setAutoDraw(True)
                
                # if listB_name is active this frame...
                if listB_name.status == STARTED:
                    # update params
                    pass
                
                # if listB_name is stopping this frame...
                if listB_name.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > listB_name.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        listB_name.tStop = t  # not accounting for scr refresh
                        listB_name.tStopRefresh = tThisFlipGlobal  # on global time
                        listB_name.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'listB_name.stopped')
                        # update status
                        listB_name.status = FINISHED
                        listB_name.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=listB,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    listB.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in listB.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "listB" ---
            for thisComponent in listB.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for listB
            listB.tStop = globalClock.getTime(format='float')
            listB.tStopRefresh = tThisFlipGlobal
            thisExp.addData('listB.stopped', listB.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if listB.maxDurationReached:
                routineTimer.addTime(-listB.maxDuration)
            elif listB.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-8.000000)
            # mark thisListB_loop as finished
            if hasattr(thisListB_loop, 'status'):
                thisListB_loop.status = FINISHED
            # if awaiting a pause, pause now
            if listB_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                listB_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'listB_loop'
        listB_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "rdm_instrx" ---
        # create an object to store info about Routine rdm_instrx
        rdm_instrx = data.Routine(
            name='rdm_instrx',
            components=[rdm_instrx_txt, rdm_instrx_resp, rdm_instrx_cont],
        )
        rdm_instrx.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for rdm_instrx_resp
        rdm_instrx_resp.keys = []
        rdm_instrx_resp.rt = []
        _rdm_instrx_resp_allKeys = []
        # store start times for rdm_instrx
        rdm_instrx.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rdm_instrx.tStart = globalClock.getTime(format='float')
        rdm_instrx.status = STARTED
        rdm_instrx.maxDuration = None
        # keep track of which components have finished
        rdm_instrxComponents = rdm_instrx.components
        for thisComponent in rdm_instrx.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rdm_instrx" ---
        rdm_instrx.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rdm_instrx_txt* updates
            
            # if rdm_instrx_txt is starting this frame...
            if rdm_instrx_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rdm_instrx_txt.frameNStart = frameN  # exact frame index
                rdm_instrx_txt.tStart = t  # local t and not account for scr refresh
                rdm_instrx_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rdm_instrx_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rdm_instrx_txt.started')
                # update status
                rdm_instrx_txt.status = STARTED
                rdm_instrx_txt.setAutoDraw(True)
            
            # if rdm_instrx_txt is active this frame...
            if rdm_instrx_txt.status == STARTED:
                # update params
                pass
            
            # *rdm_instrx_resp* updates
            waitOnFlip = False
            
            # if rdm_instrx_resp is starting this frame...
            if rdm_instrx_resp.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                rdm_instrx_resp.frameNStart = frameN  # exact frame index
                rdm_instrx_resp.tStart = t  # local t and not account for scr refresh
                rdm_instrx_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rdm_instrx_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rdm_instrx_resp.started')
                # update status
                rdm_instrx_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(rdm_instrx_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(rdm_instrx_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if rdm_instrx_resp.status == STARTED and not waitOnFlip:
                theseKeys = rdm_instrx_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _rdm_instrx_resp_allKeys.extend(theseKeys)
                if len(_rdm_instrx_resp_allKeys):
                    rdm_instrx_resp.keys = _rdm_instrx_resp_allKeys[-1].name  # just the last key pressed
                    rdm_instrx_resp.rt = _rdm_instrx_resp_allKeys[-1].rt
                    rdm_instrx_resp.duration = _rdm_instrx_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *rdm_instrx_cont* updates
            
            # if rdm_instrx_cont is starting this frame...
            if rdm_instrx_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                rdm_instrx_cont.frameNStart = frameN  # exact frame index
                rdm_instrx_cont.tStart = t  # local t and not account for scr refresh
                rdm_instrx_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rdm_instrx_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rdm_instrx_cont.started')
                # update status
                rdm_instrx_cont.status = STARTED
                rdm_instrx_cont.setAutoDraw(True)
            
            # if rdm_instrx_cont is active this frame...
            if rdm_instrx_cont.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=rdm_instrx,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rdm_instrx.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rdm_instrx.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rdm_instrx" ---
        for thisComponent in rdm_instrx.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rdm_instrx
        rdm_instrx.tStop = globalClock.getTime(format='float')
        rdm_instrx.tStopRefresh = tThisFlipGlobal
        # check responses
        if rdm_instrx_resp.keys in ['', [], None]:  # No response was made
            rdm_instrx_resp.keys = None
        block_loop.addData('rdm_instrx_resp.keys',rdm_instrx_resp.keys)
        if rdm_instrx_resp.keys != None:  # we had a response
            block_loop.addData('rdm_instrx_resp.rt', rdm_instrx_resp.rt)
            block_loop.addData('rdm_instrx_resp.duration', rdm_instrx_resp.duration)
        # the Routine "rdm_instrx" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "begin_rdm" ---
        # create an object to store info about Routine begin_rdm
        begin_rdm = data.Routine(
            name='begin_rdm',
            components=[begin_rdm_txt],
        )
        begin_rdm.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for begin_rdm
        begin_rdm.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        begin_rdm.tStart = globalClock.getTime(format='float')
        begin_rdm.status = STARTED
        thisExp.addData('begin_rdm.started', begin_rdm.tStart)
        begin_rdm.maxDuration = None
        # keep track of which components have finished
        begin_rdmComponents = begin_rdm.components
        for thisComponent in begin_rdm.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "begin_rdm" ---
        begin_rdm.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *begin_rdm_txt* updates
            
            # if begin_rdm_txt is starting this frame...
            if begin_rdm_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                begin_rdm_txt.frameNStart = frameN  # exact frame index
                begin_rdm_txt.tStart = t  # local t and not account for scr refresh
                begin_rdm_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(begin_rdm_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'begin_rdm_txt.started')
                # update status
                begin_rdm_txt.status = STARTED
                begin_rdm_txt.setAutoDraw(True)
            
            # if begin_rdm_txt is active this frame...
            if begin_rdm_txt.status == STARTED:
                # update params
                pass
            
            # if begin_rdm_txt is stopping this frame...
            if begin_rdm_txt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > begin_rdm_txt.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    begin_rdm_txt.tStop = t  # not accounting for scr refresh
                    begin_rdm_txt.tStopRefresh = tThisFlipGlobal  # on global time
                    begin_rdm_txt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'begin_rdm_txt.stopped')
                    # update status
                    begin_rdm_txt.status = FINISHED
                    begin_rdm_txt.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=begin_rdm,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                begin_rdm.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in begin_rdm.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "begin_rdm" ---
        for thisComponent in begin_rdm.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for begin_rdm
        begin_rdm.tStop = globalClock.getTime(format='float')
        begin_rdm.tStopRefresh = tThisFlipGlobal
        thisExp.addData('begin_rdm.stopped', begin_rdm.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if begin_rdm.maxDurationReached:
            routineTimer.addTime(-begin_rdm.maxDuration)
        elif begin_rdm.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # set up handler to look after randomisation of conditions etc
        rdm_loop = data.TrialHandler2(
            name='rdm_loop',
            nReps=9999.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(rdm_loop)  # add the loop to the experiment
        thisRdm_loop = rdm_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRdm_loop.rgb)
        if thisRdm_loop != None:
            for paramName in thisRdm_loop:
                globals()[paramName] = thisRdm_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisRdm_loop in rdm_loop:
            rdm_loop.status = STARTED
            if hasattr(thisRdm_loop, 'status'):
                thisRdm_loop.status = STARTED
            currentLoop = rdm_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisRdm_loop.rgb)
            if thisRdm_loop != None:
                for paramName in thisRdm_loop:
                    globals()[paramName] = thisRdm_loop[paramName]
            
            # --- Prepare to start Routine "rdm_isi" ---
            # create an object to store info about Routine rdm_isi
            rdm_isi = data.Routine(
                name='rdm_isi',
                components=[isi_rdm],
            )
            rdm_isi.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for rdm_isi
            rdm_isi.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            rdm_isi.tStart = globalClock.getTime(format='float')
            rdm_isi.status = STARTED
            thisExp.addData('rdm_isi.started', rdm_isi.tStart)
            rdm_isi.maxDuration = None
            # keep track of which components have finished
            rdm_isiComponents = rdm_isi.components
            for thisComponent in rdm_isi.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "rdm_isi" ---
            rdm_isi.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisRdm_loop, 'status') and thisRdm_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *isi_rdm* updates
                
                # if isi_rdm is starting this frame...
                if isi_rdm.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    isi_rdm.frameNStart = frameN  # exact frame index
                    isi_rdm.tStart = t  # local t and not account for scr refresh
                    isi_rdm.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(isi_rdm, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'isi_rdm.started')
                    # update status
                    isi_rdm.status = STARTED
                    isi_rdm.setAutoDraw(True)
                
                # if isi_rdm is active this frame...
                if isi_rdm.status == STARTED:
                    # update params
                    pass
                
                # if isi_rdm is stopping this frame...
                if isi_rdm.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > isi_rdm.tStartRefresh + random.uniform(0.5, 1.5)-frameTolerance:
                        # keep track of stop time/frame for later
                        isi_rdm.tStop = t  # not accounting for scr refresh
                        isi_rdm.tStopRefresh = tThisFlipGlobal  # on global time
                        isi_rdm.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'isi_rdm.stopped')
                        # update status
                        isi_rdm.status = FINISHED
                        isi_rdm.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=rdm_isi,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    rdm_isi.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in rdm_isi.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "rdm_isi" ---
            for thisComponent in rdm_isi.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for rdm_isi
            rdm_isi.tStop = globalClock.getTime(format='float')
            rdm_isi.tStopRefresh = tThisFlipGlobal
            thisExp.addData('rdm_isi.stopped', rdm_isi.tStop)
            # the Routine "rdm_isi" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "distractor_dots" ---
            # create an object to store info about Routine distractor_dots
            distractor_dots = data.Routine(
                name='distractor_dots',
                components=[dots, dots_resp],
            )
            distractor_dots.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            dots.setDir(0.0)
            dots.setFieldCoherence(0.2)
            dots.refreshDots()
            # create starting attributes for dots_resp
            dots_resp.keys = []
            dots_resp.rt = []
            _dots_resp_allKeys = []
            # Run 'Begin Routine' code from dots_code
            # Reset the clock on the first trial
            if rdm_loop.thisN == 0:
                rdm_clock.reset()
                rdm_loop.finished = False
            
            # Randomize direction each trial, fixed coherence
            rdm_trial_direction = random.choice([0,180])
            rdm_trial_coherence = 0.20
            
            dots.dir = rdm_trial_direction
            dots.coherence = rdm_trial_coherence
            # store start times for distractor_dots
            distractor_dots.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            distractor_dots.tStart = globalClock.getTime(format='float')
            distractor_dots.status = STARTED
            thisExp.addData('distractor_dots.started', distractor_dots.tStart)
            distractor_dots.maxDuration = None
            # keep track of which components have finished
            distractor_dotsComponents = distractor_dots.components
            for thisComponent in distractor_dots.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "distractor_dots" ---
            distractor_dots.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisRdm_loop, 'status') and thisRdm_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *dots* updates
                
                # if dots is starting this frame...
                if dots.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots.frameNStart = frameN  # exact frame index
                    dots.tStart = t  # local t and not account for scr refresh
                    dots.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dots.started')
                    # update status
                    dots.status = STARTED
                    dots.setAutoDraw(True)
                
                # if dots is active this frame...
                if dots.status == STARTED:
                    # update params
                    pass
                
                # *dots_resp* updates
                waitOnFlip = False
                
                # if dots_resp is starting this frame...
                if dots_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_resp.frameNStart = frameN  # exact frame index
                    dots_resp.tStart = t  # local t and not account for scr refresh
                    dots_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dots_resp.started')
                    # update status
                    dots_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(dots_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(dots_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if dots_resp.status == STARTED and not waitOnFlip:
                    theseKeys = dots_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _dots_resp_allKeys.extend(theseKeys)
                    if len(_dots_resp_allKeys):
                        dots_resp.keys = _dots_resp_allKeys[-1].name  # just the last key pressed
                        dots_resp.rt = _dots_resp_allKeys[-1].rt
                        dots_resp.duration = _dots_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=distractor_dots,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    distractor_dots.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in distractor_dots.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "distractor_dots" ---
            for thisComponent in distractor_dots.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for distractor_dots
            distractor_dots.tStop = globalClock.getTime(format='float')
            distractor_dots.tStopRefresh = tThisFlipGlobal
            thisExp.addData('distractor_dots.stopped', distractor_dots.tStop)
            # check responses
            if dots_resp.keys in ['', [], None]:  # No response was made
                dots_resp.keys = None
            rdm_loop.addData('dots_resp.keys',dots_resp.keys)
            if dots_resp.keys != None:  # we had a response
                rdm_loop.addData('dots_resp.rt', dots_resp.rt)
                rdm_loop.addData('dots_resp.duration', dots_resp.duration)
            # Run 'End Routine' code from dots_code
            correct_key = 'right' if rdm_trial_direction == 0 else 'left'
            if dots_resp.keys == correct_key:
                rdm_trial_correct = 1
            else:
                rdm_trial_correct = 0
                
            thisExp.addData('rdm_trial_correct', rdm_trial_correct)
            thisExp.addData('rdm_trial_direction', rdm_trial_direction)
            # the Routine "distractor_dots" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "distractor_feedback" ---
            # create an object to store info about Routine distractor_feedback
            distractor_feedback = data.Routine(
                name='distractor_feedback',
                components=[dots_feedback_text],
            )
            distractor_feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            dots_feedback_text.setText('')
            # Run 'Begin Routine' code from dots_feedback_code
            if rdm_trial_correct == 1:
                dots_feedback_text.text = "Correct!"
                dots_feedback_text.color = 'green'
            else: 
                dots_feedback_text.text = "Incorrect"
                dots_feedback_text.color = 'red'
            # store start times for distractor_feedback
            distractor_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            distractor_feedback.tStart = globalClock.getTime(format='float')
            distractor_feedback.status = STARTED
            thisExp.addData('distractor_feedback.started', distractor_feedback.tStart)
            distractor_feedback.maxDuration = None
            # keep track of which components have finished
            distractor_feedbackComponents = distractor_feedback.components
            for thisComponent in distractor_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "distractor_feedback" ---
            distractor_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.8:
                # if trial has changed, end Routine now
                if hasattr(thisRdm_loop, 'status') and thisRdm_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *dots_feedback_text* updates
                
                # if dots_feedback_text is starting this frame...
                if dots_feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_feedback_text.frameNStart = frameN  # exact frame index
                    dots_feedback_text.tStart = t  # local t and not account for scr refresh
                    dots_feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_feedback_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dots_feedback_text.started')
                    # update status
                    dots_feedback_text.status = STARTED
                    dots_feedback_text.setAutoDraw(True)
                
                # if dots_feedback_text is active this frame...
                if dots_feedback_text.status == STARTED:
                    # update params
                    pass
                
                # if dots_feedback_text is stopping this frame...
                if dots_feedback_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_feedback_text.tStartRefresh + 0.8-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_feedback_text.tStop = t  # not accounting for scr refresh
                        dots_feedback_text.tStopRefresh = tThisFlipGlobal  # on global time
                        dots_feedback_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'dots_feedback_text.stopped')
                        # update status
                        dots_feedback_text.status = FINISHED
                        dots_feedback_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=distractor_feedback,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    distractor_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in distractor_feedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "distractor_feedback" ---
            for thisComponent in distractor_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for distractor_feedback
            distractor_feedback.tStop = globalClock.getTime(format='float')
            distractor_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('distractor_feedback.stopped', distractor_feedback.tStop)
            # Run 'End Routine' code from dots_feedback_code
            if rdm_clock.getTime() >= rdm_total_duration:
                rdm_loop.finished = True
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if distractor_feedback.maxDurationReached:
                routineTimer.addTime(-distractor_feedback.maxDuration)
            elif distractor_feedback.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.800000)
            # mark thisRdm_loop as finished
            if hasattr(thisRdm_loop, 'status'):
                thisRdm_loop.status = FINISHED
            # if awaiting a pause, pause now
            if rdm_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                rdm_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 9999.0 repeats of 'rdm_loop'
        rdm_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "test_instr_1" ---
        # create an object to store info about Routine test_instr_1
        test_instr_1 = data.Routine(
            name='test_instr_1',
            components=[test_instructions, test_resp, test_continue],
        )
        test_instr_1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for test_resp
        test_resp.keys = []
        test_resp.rt = []
        _test_resp_allKeys = []
        # Run 'Begin Routine' code from test_trials
        #use corresponding test trials for the block shown
        currentBlock = block_loop.thisN + 1
        block_rows = f"{(currentBlock-1)*32}:{currentBlock*32}"
        # store start times for test_instr_1
        test_instr_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test_instr_1.tStart = globalClock.getTime(format='float')
        test_instr_1.status = STARTED
        test_instr_1.maxDuration = None
        # keep track of which components have finished
        test_instr_1Components = test_instr_1.components
        for thisComponent in test_instr_1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_instr_1" ---
        test_instr_1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *test_instructions* updates
            
            # if test_instructions is starting this frame...
            if test_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_instructions.frameNStart = frameN  # exact frame index
                test_instructions.tStart = t  # local t and not account for scr refresh
                test_instructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_instructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_instructions.started')
                # update status
                test_instructions.status = STARTED
                test_instructions.setAutoDraw(True)
            
            # if test_instructions is active this frame...
            if test_instructions.status == STARTED:
                # update params
                pass
            
            # *test_resp* updates
            waitOnFlip = False
            
            # if test_resp is starting this frame...
            if test_resp.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_resp.frameNStart = frameN  # exact frame index
                test_resp.tStart = t  # local t and not account for scr refresh
                test_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_resp.started')
                # update status
                test_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(test_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(test_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if test_resp.status == STARTED and not waitOnFlip:
                theseKeys = test_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _test_resp_allKeys.extend(theseKeys)
                if len(_test_resp_allKeys):
                    test_resp.keys = _test_resp_allKeys[-1].name  # just the last key pressed
                    test_resp.rt = _test_resp_allKeys[-1].rt
                    test_resp.duration = _test_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *test_continue* updates
            
            # if test_continue is starting this frame...
            if test_continue.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_continue.frameNStart = frameN  # exact frame index
                test_continue.tStart = t  # local t and not account for scr refresh
                test_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_continue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_continue.started')
                # update status
                test_continue.status = STARTED
                test_continue.setAutoDraw(True)
            
            # if test_continue is active this frame...
            if test_continue.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test_instr_1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                test_instr_1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_instr_1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_instr_1" ---
        for thisComponent in test_instr_1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test_instr_1
        test_instr_1.tStop = globalClock.getTime(format='float')
        test_instr_1.tStopRefresh = tThisFlipGlobal
        # check responses
        if test_resp.keys in ['', [], None]:  # No response was made
            test_resp.keys = None
        block_loop.addData('test_resp.keys',test_resp.keys)
        if test_resp.keys != None:  # we had a response
            block_loop.addData('test_resp.rt', test_resp.rt)
            block_loop.addData('test_resp.duration', test_resp.duration)
        # the Routine "test_instr_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "test_instr_2" ---
        # create an object to store info about Routine test_instr_2
        test_instr_2 = data.Routine(
            name='test_instr_2',
            components=[test_instructions_2, test_resp_2, test_continue_2],
        )
        test_instr_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        test_instructions_2.setText(on_instruction_text)
        # create starting attributes for test_resp_2
        test_resp_2.keys = []
        test_resp_2.rt = []
        _test_resp_2_allKeys = []
        # Run 'Begin Routine' code from test_trials_2
        #use corresponding test trials for the block shown
        currentBlock = block_loop.thisN + 1
        block_rows = f"{(currentBlock-1)*32}:{currentBlock*32}"
        # store start times for test_instr_2
        test_instr_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test_instr_2.tStart = globalClock.getTime(format='float')
        test_instr_2.status = STARTED
        test_instr_2.maxDuration = None
        # keep track of which components have finished
        test_instr_2Components = test_instr_2.components
        for thisComponent in test_instr_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_instr_2" ---
        test_instr_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *test_instructions_2* updates
            
            # if test_instructions_2 is starting this frame...
            if test_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_instructions_2.frameNStart = frameN  # exact frame index
                test_instructions_2.tStart = t  # local t and not account for scr refresh
                test_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_instructions_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_instructions_2.started')
                # update status
                test_instructions_2.status = STARTED
                test_instructions_2.setAutoDraw(True)
            
            # if test_instructions_2 is active this frame...
            if test_instructions_2.status == STARTED:
                # update params
                pass
            
            # *test_resp_2* updates
            waitOnFlip = False
            
            # if test_resp_2 is starting this frame...
            if test_resp_2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_resp_2.frameNStart = frameN  # exact frame index
                test_resp_2.tStart = t  # local t and not account for scr refresh
                test_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_resp_2.started')
                # update status
                test_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(test_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(test_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if test_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = test_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _test_resp_2_allKeys.extend(theseKeys)
                if len(_test_resp_2_allKeys):
                    test_resp_2.keys = _test_resp_2_allKeys[-1].name  # just the last key pressed
                    test_resp_2.rt = _test_resp_2_allKeys[-1].rt
                    test_resp_2.duration = _test_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *test_continue_2* updates
            
            # if test_continue_2 is starting this frame...
            if test_continue_2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_continue_2.frameNStart = frameN  # exact frame index
                test_continue_2.tStart = t  # local t and not account for scr refresh
                test_continue_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_continue_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_continue_2.started')
                # update status
                test_continue_2.status = STARTED
                test_continue_2.setAutoDraw(True)
            
            # if test_continue_2 is active this frame...
            if test_continue_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test_instr_2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                test_instr_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_instr_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_instr_2" ---
        for thisComponent in test_instr_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test_instr_2
        test_instr_2.tStop = globalClock.getTime(format='float')
        test_instr_2.tStopRefresh = tThisFlipGlobal
        # check responses
        if test_resp_2.keys in ['', [], None]:  # No response was made
            test_resp_2.keys = None
        block_loop.addData('test_resp_2.keys',test_resp_2.keys)
        if test_resp_2.keys != None:  # we had a response
            block_loop.addData('test_resp_2.rt', test_resp_2.rt)
            block_loop.addData('test_resp_2.duration', test_resp_2.duration)
        # the Routine "test_instr_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "test_instr_3" ---
        # create an object to store info about Routine test_instr_3
        test_instr_3 = data.Routine(
            name='test_instr_3',
            components=[test_instructions_3, test_resp_3, test_continue_3],
        )
        test_instr_3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for test_resp_3
        test_resp_3.keys = []
        test_resp_3.rt = []
        _test_resp_3_allKeys = []
        # store start times for test_instr_3
        test_instr_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test_instr_3.tStart = globalClock.getTime(format='float')
        test_instr_3.status = STARTED
        test_instr_3.maxDuration = None
        # keep track of which components have finished
        test_instr_3Components = test_instr_3.components
        for thisComponent in test_instr_3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_instr_3" ---
        test_instr_3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *test_instructions_3* updates
            
            # if test_instructions_3 is starting this frame...
            if test_instructions_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_instructions_3.frameNStart = frameN  # exact frame index
                test_instructions_3.tStart = t  # local t and not account for scr refresh
                test_instructions_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_instructions_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_instructions_3.started')
                # update status
                test_instructions_3.status = STARTED
                test_instructions_3.setAutoDraw(True)
            
            # if test_instructions_3 is active this frame...
            if test_instructions_3.status == STARTED:
                # update params
                pass
            
            # *test_resp_3* updates
            waitOnFlip = False
            
            # if test_resp_3 is starting this frame...
            if test_resp_3.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_resp_3.frameNStart = frameN  # exact frame index
                test_resp_3.tStart = t  # local t and not account for scr refresh
                test_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_resp_3.started')
                # update status
                test_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(test_resp_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(test_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if test_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = test_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _test_resp_3_allKeys.extend(theseKeys)
                if len(_test_resp_3_allKeys):
                    test_resp_3.keys = _test_resp_3_allKeys[-1].name  # just the last key pressed
                    test_resp_3.rt = _test_resp_3_allKeys[-1].rt
                    test_resp_3.duration = _test_resp_3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *test_continue_3* updates
            
            # if test_continue_3 is starting this frame...
            if test_continue_3.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_continue_3.frameNStart = frameN  # exact frame index
                test_continue_3.tStart = t  # local t and not account for scr refresh
                test_continue_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_continue_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_continue_3.started')
                # update status
                test_continue_3.status = STARTED
                test_continue_3.setAutoDraw(True)
            
            # if test_continue_3 is active this frame...
            if test_continue_3.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test_instr_3,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                test_instr_3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_instr_3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_instr_3" ---
        for thisComponent in test_instr_3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test_instr_3
        test_instr_3.tStop = globalClock.getTime(format='float')
        test_instr_3.tStopRefresh = tThisFlipGlobal
        # check responses
        if test_resp_3.keys in ['', [], None]:  # No response was made
            test_resp_3.keys = None
        block_loop.addData('test_resp_3.keys',test_resp_3.keys)
        if test_resp_3.keys != None:  # we had a response
            block_loop.addData('test_resp_3.rt', test_resp_3.rt)
            block_loop.addData('test_resp_3.duration', test_resp_3.duration)
        # the Routine "test_instr_3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_test_setup" ---
        # create an object to store info about Routine prac_test_setup
        prac_test_setup = data.Routine(
            name='prac_test_setup',
            components=[prac_test_txt, prac_test_resp],
        )
        prac_test_setup.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_test_txt.setText('')
        # create starting attributes for prac_test_resp
        prac_test_resp.keys = []
        prac_test_resp.rt = []
        _prac_test_resp_allKeys = []
        # Run 'Begin Routine' code from prac_test_YesNo
        #begin experiment with practice trials instructions, optional after.
        if block_loop.thisN == 0:
            prac_test_txt.setText("To make sure you understand the task, we will show you some practice trials. You must get the practice trials correct to move forward.\n\nPress the 'p' key to continue.")
        else: 
            prac_test_txt.setText("Would you like a reminder of what to do during the task? If so, press the 'p' key. If not, press the 'c' key to continue with the next block.")
            
        # store start times for prac_test_setup
        prac_test_setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_test_setup.tStart = globalClock.getTime(format='float')
        prac_test_setup.status = STARTED
        prac_test_setup.maxDuration = None
        # keep track of which components have finished
        prac_test_setupComponents = prac_test_setup.components
        for thisComponent in prac_test_setup.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_test_setup" ---
        prac_test_setup.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_test_txt* updates
            
            # if prac_test_txt is starting this frame...
            if prac_test_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_test_txt.frameNStart = frameN  # exact frame index
                prac_test_txt.tStart = t  # local t and not account for scr refresh
                prac_test_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_test_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_test_txt.started')
                # update status
                prac_test_txt.status = STARTED
                prac_test_txt.setAutoDraw(True)
            
            # if prac_test_txt is active this frame...
            if prac_test_txt.status == STARTED:
                # update params
                pass
            
            # *prac_test_resp* updates
            waitOnFlip = False
            
            # if prac_test_resp is starting this frame...
            if prac_test_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_test_resp.frameNStart = frameN  # exact frame index
                prac_test_resp.tStart = t  # local t and not account for scr refresh
                prac_test_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_test_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_test_resp.started')
                # update status
                prac_test_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_test_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_test_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_test_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_test_resp.getKeys(keyList=['p','c'], ignoreKeys=["escape"], waitRelease=False)
                _prac_test_resp_allKeys.extend(theseKeys)
                if len(_prac_test_resp_allKeys):
                    prac_test_resp.keys = _prac_test_resp_allKeys[-1].name  # just the last key pressed
                    prac_test_resp.rt = _prac_test_resp_allKeys[-1].rt
                    prac_test_resp.duration = _prac_test_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_test_setup,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_test_setup.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_test_setup.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_test_setup" ---
        for thisComponent in prac_test_setup.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_test_setup
        prac_test_setup.tStop = globalClock.getTime(format='float')
        prac_test_setup.tStopRefresh = tThisFlipGlobal
        # check responses
        if prac_test_resp.keys in ['', [], None]:  # No response was made
            prac_test_resp.keys = None
        block_loop.addData('prac_test_resp.keys',prac_test_resp.keys)
        if prac_test_resp.keys != None:  # we had a response
            block_loop.addData('prac_test_resp.rt', prac_test_resp.rt)
            block_loop.addData('prac_test_resp.duration', prac_test_resp.duration)
        # Run 'End Routine' code from prac_test_YesNo
        #depending on participant response, log answer to continue or skip practice trials
        if 'p' in prac_test_resp.keys:
            prac_test_selected = 999
        elif 'c' in prac_test_resp.keys:
            prac_test_selected = 0
        # the Routine "prac_test_setup" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        prac_test_loop = data.TrialHandler2(
            name='prac_test_loop',
            nReps=prac_test_selected, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(prac_test_loop)  # add the loop to the experiment
        thisPrac_test_loop = prac_test_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_test_loop.rgb)
        if thisPrac_test_loop != None:
            for paramName in thisPrac_test_loop:
                globals()[paramName] = thisPrac_test_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPrac_test_loop in prac_test_loop:
            prac_test_loop.status = STARTED
            if hasattr(thisPrac_test_loop, 'status'):
                thisPrac_test_loop.status = STARTED
            currentLoop = prac_test_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_test_loop.rgb)
            if thisPrac_test_loop != None:
                for paramName in thisPrac_test_loop:
                    globals()[paramName] = thisPrac_test_loop[paramName]
            
            # --- Prepare to start Routine "prac_test1" ---
            # create an object to store info about Routine prac_test1
            prac_test1 = data.Routine(
                name='prac_test1',
                components=[prac1_isi, prac_image, prac_OldNew, prac1_test_resp, prac1_left_resp, prac1_right_resp],
            )
            prac_test1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from prac_TestCode
            #set black for participant input
            prac_TestResp = None
            prac_image.setImage('practice_trials/c859eac8ff5175847ec6aca0a830261a.jpg')
            # create starting attributes for prac1_test_resp
            prac1_test_resp.keys = []
            prac1_test_resp.rt = []
            _prac1_test_resp_allKeys = []
            prac1_left_resp.setText(left_label_text)
            prac1_right_resp.setText(right_label_text)
            # store start times for prac_test1
            prac_test1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_test1.tStart = globalClock.getTime(format='float')
            prac_test1.status = STARTED
            thisExp.addData('prac_test1.started', prac_test1.tStart)
            prac_test1.maxDuration = None
            # keep track of which components have finished
            prac_test1Components = prac_test1.components
            for thisComponent in prac_test1.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_test1" ---
            prac_test1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac1_isi* updates
                
                # if prac1_isi is starting this frame...
                if prac1_isi.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac1_isi.frameNStart = frameN  # exact frame index
                    prac1_isi.tStart = t  # local t and not account for scr refresh
                    prac1_isi.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac1_isi, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac1_isi.started')
                    # update status
                    prac1_isi.status = STARTED
                    prac1_isi.setAutoDraw(True)
                
                # if prac1_isi is active this frame...
                if prac1_isi.status == STARTED:
                    # update params
                    pass
                
                # if prac1_isi is stopping this frame...
                if prac1_isi.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac1_isi.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        prac1_isi.tStop = t  # not accounting for scr refresh
                        prac1_isi.tStopRefresh = tThisFlipGlobal  # on global time
                        prac1_isi.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac1_isi.stopped')
                        # update status
                        prac1_isi.status = FINISHED
                        prac1_isi.setAutoDraw(False)
                
                # *prac_image* updates
                
                # if prac_image is starting this frame...
                if prac_image.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac_image.frameNStart = frameN  # exact frame index
                    prac_image.tStart = t  # local t and not account for scr refresh
                    prac_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_image.started')
                    # update status
                    prac_image.status = STARTED
                    prac_image.setAutoDraw(True)
                
                # if prac_image is active this frame...
                if prac_image.status == STARTED:
                    # update params
                    pass
                
                # *prac_OldNew* updates
                
                # if prac_OldNew is starting this frame...
                if prac_OldNew.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac_OldNew.frameNStart = frameN  # exact frame index
                    prac_OldNew.tStart = t  # local t and not account for scr refresh
                    prac_OldNew.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_OldNew, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_OldNew.started')
                    # update status
                    prac_OldNew.status = STARTED
                    prac_OldNew.setAutoDraw(True)
                
                # if prac_OldNew is active this frame...
                if prac_OldNew.status == STARTED:
                    # update params
                    pass
                
                # *prac1_test_resp* updates
                waitOnFlip = False
                
                # if prac1_test_resp is starting this frame...
                if prac1_test_resp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac1_test_resp.frameNStart = frameN  # exact frame index
                    prac1_test_resp.tStart = t  # local t and not account for scr refresh
                    prac1_test_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac1_test_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac1_test_resp.started')
                    # update status
                    prac1_test_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac1_test_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac1_test_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac1_test_resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac1_test_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _prac1_test_resp_allKeys.extend(theseKeys)
                    if len(_prac1_test_resp_allKeys):
                        prac1_test_resp.keys = _prac1_test_resp_allKeys[-1].name  # just the last key pressed
                        prac1_test_resp.rt = _prac1_test_resp_allKeys[-1].rt
                        prac1_test_resp.duration = _prac1_test_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *prac1_left_resp* updates
                
                # if prac1_left_resp is starting this frame...
                if prac1_left_resp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac1_left_resp.frameNStart = frameN  # exact frame index
                    prac1_left_resp.tStart = t  # local t and not account for scr refresh
                    prac1_left_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac1_left_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac1_left_resp.started')
                    # update status
                    prac1_left_resp.status = STARTED
                    prac1_left_resp.setAutoDraw(True)
                
                # if prac1_left_resp is active this frame...
                if prac1_left_resp.status == STARTED:
                    # update params
                    pass
                
                # *prac1_right_resp* updates
                
                # if prac1_right_resp is starting this frame...
                if prac1_right_resp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac1_right_resp.frameNStart = frameN  # exact frame index
                    prac1_right_resp.tStart = t  # local t and not account for scr refresh
                    prac1_right_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac1_right_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac1_right_resp.started')
                    # update status
                    prac1_right_resp.status = STARTED
                    prac1_right_resp.setAutoDraw(True)
                
                # if prac1_right_resp is active this frame...
                if prac1_right_resp.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac_test1,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_test1.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_test1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_test1" ---
            for thisComponent in prac_test1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_test1
            prac_test1.tStop = globalClock.getTime(format='float')
            prac_test1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_test1.stopped', prac_test1.tStop)
            # Run 'End Routine' code from prac_TestCode
            #Log response
            
            if prac1_test_resp.keys == yes_key:
                prac_TestResp = 'old'
            elif prac1_test_resp.keys == no_key:
                prac_TestResp = 'new'
                prac_typed_name = ""
            thisExp.addData('prac_TestResp', prac_TestResp)
                    
            
            #add to all answers if correct
            practice_test_correct.append(prac_TestResp == "new")
            # check responses
            if prac1_test_resp.keys in ['', [], None]:  # No response was made
                prac1_test_resp.keys = None
            prac_test_loop.addData('prac1_test_resp.keys',prac1_test_resp.keys)
            if prac1_test_resp.keys != None:  # we had a response
                prac_test_loop.addData('prac1_test_resp.rt', prac1_test_resp.rt)
                prac_test_loop.addData('prac1_test_resp.duration', prac1_test_resp.duration)
            # the Routine "prac_test1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac1_confidence" ---
            # create an object to store info about Routine prac1_confidence
            prac1_confidence = data.Routine(
                name='prac1_confidence',
                components=[prac_image2, prac_question, prac_conf1, prac_conf1_cont],
            )
            prac1_confidence.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_conf1.reset()
            # reset prac_conf1_cont to account for continued clicks & clear times on/off
            prac_conf1_cont.reset()
            # store start times for prac1_confidence
            prac1_confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac1_confidence.tStart = globalClock.getTime(format='float')
            prac1_confidence.status = STARTED
            thisExp.addData('prac1_confidence.started', prac1_confidence.tStart)
            prac1_confidence.maxDuration = None
            # keep track of which components have finished
            prac1_confidenceComponents = prac1_confidence.components
            for thisComponent in prac1_confidence.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac1_confidence" ---
            prac1_confidence.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_image2* updates
                
                # if prac_image2 is starting this frame...
                if prac_image2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_image2.frameNStart = frameN  # exact frame index
                    prac_image2.tStart = t  # local t and not account for scr refresh
                    prac_image2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_image2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_image2.started')
                    # update status
                    prac_image2.status = STARTED
                    prac_image2.setAutoDraw(True)
                
                # if prac_image2 is active this frame...
                if prac_image2.status == STARTED:
                    # update params
                    pass
                
                # *prac_question* updates
                
                # if prac_question is starting this frame...
                if prac_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_question.frameNStart = frameN  # exact frame index
                    prac_question.tStart = t  # local t and not account for scr refresh
                    prac_question.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_question, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_question.started')
                    # update status
                    prac_question.status = STARTED
                    prac_question.setAutoDraw(True)
                
                # if prac_question is active this frame...
                if prac_question.status == STARTED:
                    # update params
                    pass
                
                # *prac_conf1* updates
                
                # if prac_conf1 is starting this frame...
                if prac_conf1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_conf1.frameNStart = frameN  # exact frame index
                    prac_conf1.tStart = t  # local t and not account for scr refresh
                    prac_conf1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_conf1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_conf1.started')
                    # update status
                    prac_conf1.status = STARTED
                    prac_conf1.setAutoDraw(True)
                
                # if prac_conf1 is active this frame...
                if prac_conf1.status == STARTED:
                    # update params
                    pass
                # *prac_conf1_cont* updates
                
                # if prac_conf1_cont is starting this frame...
                if prac_conf1_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_conf1_cont.frameNStart = frameN  # exact frame index
                    prac_conf1_cont.tStart = t  # local t and not account for scr refresh
                    prac_conf1_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_conf1_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_conf1_cont.started')
                    # update status
                    prac_conf1_cont.status = STARTED
                    win.callOnFlip(prac_conf1_cont.buttonClock.reset)
                    prac_conf1_cont.setAutoDraw(True)
                
                # if prac_conf1_cont is active this frame...
                if prac_conf1_cont.status == STARTED:
                    # update params
                    pass
                    # check whether prac_conf1_cont has been pressed
                    if prac_conf1_cont.isClicked:
                        if not prac_conf1_cont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac_conf1_cont.timesOn.append(prac_conf1_cont.buttonClock.getTime())
                            prac_conf1_cont.timesOff.append(prac_conf1_cont.buttonClock.getTime())
                        elif len(prac_conf1_cont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac_conf1_cont.timesOff[-1] = prac_conf1_cont.buttonClock.getTime()
                        if not prac_conf1_cont.wasClicked:
                            # run callback code when prac_conf1_cont is clicked
                            pass
                # take note of whether prac_conf1_cont was clicked, so that next frame we know if clicks are new
                prac_conf1_cont.wasClicked = prac_conf1_cont.isClicked and prac_conf1_cont.status == STARTED
                # Run 'Each Frame' code from prac_conf1_code
                #Cont. only allowed after answering the slider and pressing continue. 
                if prac_conf1_cont.isClicked and prac_conf1.getRating() is not None:
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac1_confidence,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac1_confidence.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac1_confidence.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac1_confidence" ---
            for thisComponent in prac1_confidence.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac1_confidence
            prac1_confidence.tStop = globalClock.getTime(format='float')
            prac1_confidence.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac1_confidence.stopped', prac1_confidence.tStop)
            prac_test_loop.addData('prac_conf1.response', prac_conf1.getRating())
            prac_test_loop.addData('prac_conf1.rt', prac_conf1.getRT())
            prac_test_loop.addData('prac_conf1_cont.numClicks', prac_conf1_cont.numClicks)
            if prac_conf1_cont.numClicks:
               prac_test_loop.addData('prac_conf1_cont.timesOn', prac_conf1_cont.timesOn)
               prac_test_loop.addData('prac_conf1_cont.timesOff', prac_conf1_cont.timesOff)
            else:
               prac_test_loop.addData('prac_conf1_cont.timesOn', "")
               prac_test_loop.addData('prac_conf1_cont.timesOff', "")
            # the Routine "prac1_confidence" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac1_textbox" ---
            # create an object to store info about Routine prac1_textbox
            prac1_textbox = data.Routine(
                name='prac1_textbox',
                components=[prac_image3, prac_OldTextbox, prac_TextboxCont, prac_idk_button],
            )
            prac1_textbox.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_OldTextbox.reset()
            prac_OldTextbox.setText('')
            prac_OldTextbox.setPlaceholder('Name?')
            # reset prac_TextboxCont to account for continued clicks & clear times on/off
            prac_TextboxCont.reset()
            # reset prac_idk_button to account for continued clicks & clear times on/off
            prac_idk_button.reset()
            # Run 'Begin Routine' code from prac_TextboxCode
            dont_remember_clicked = False
            # store start times for prac1_textbox
            prac1_textbox.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac1_textbox.tStart = globalClock.getTime(format='float')
            prac1_textbox.status = STARTED
            thisExp.addData('prac1_textbox.started', prac1_textbox.tStart)
            prac1_textbox.maxDuration = None
            # skip Routine prac1_textbox if its 'Skip if' condition is True
            prac1_textbox.skipped = continueRoutine and not (prac_TestResp != "old")
            continueRoutine = prac1_textbox.skipped
            # keep track of which components have finished
            prac1_textboxComponents = prac1_textbox.components
            for thisComponent in prac1_textbox.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac1_textbox" ---
            prac1_textbox.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_image3* updates
                
                # if prac_image3 is starting this frame...
                if prac_image3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_image3.frameNStart = frameN  # exact frame index
                    prac_image3.tStart = t  # local t and not account for scr refresh
                    prac_image3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_image3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_image3.started')
                    # update status
                    prac_image3.status = STARTED
                    prac_image3.setAutoDraw(True)
                
                # if prac_image3 is active this frame...
                if prac_image3.status == STARTED:
                    # update params
                    pass
                
                # *prac_OldTextbox* updates
                
                # if prac_OldTextbox is starting this frame...
                if prac_OldTextbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_OldTextbox.frameNStart = frameN  # exact frame index
                    prac_OldTextbox.tStart = t  # local t and not account for scr refresh
                    prac_OldTextbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_OldTextbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_OldTextbox.started')
                    # update status
                    prac_OldTextbox.status = STARTED
                    prac_OldTextbox.setAutoDraw(True)
                
                # if prac_OldTextbox is active this frame...
                if prac_OldTextbox.status == STARTED:
                    # update params
                    pass
                # *prac_TextboxCont* updates
                
                # if prac_TextboxCont is starting this frame...
                if prac_TextboxCont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_TextboxCont.frameNStart = frameN  # exact frame index
                    prac_TextboxCont.tStart = t  # local t and not account for scr refresh
                    prac_TextboxCont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_TextboxCont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_TextboxCont.started')
                    # update status
                    prac_TextboxCont.status = STARTED
                    win.callOnFlip(prac_TextboxCont.buttonClock.reset)
                    prac_TextboxCont.setAutoDraw(True)
                
                # if prac_TextboxCont is active this frame...
                if prac_TextboxCont.status == STARTED:
                    # update params
                    pass
                    # check whether prac_TextboxCont has been pressed
                    if prac_TextboxCont.isClicked:
                        if not prac_TextboxCont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac_TextboxCont.timesOn.append(prac_TextboxCont.buttonClock.getTime())
                            prac_TextboxCont.timesOff.append(prac_TextboxCont.buttonClock.getTime())
                        elif len(prac_TextboxCont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac_TextboxCont.timesOff[-1] = prac_TextboxCont.buttonClock.getTime()
                        if not prac_TextboxCont.wasClicked:
                            # run callback code when prac_TextboxCont is clicked
                            pass
                # take note of whether prac_TextboxCont was clicked, so that next frame we know if clicks are new
                prac_TextboxCont.wasClicked = prac_TextboxCont.isClicked and prac_TextboxCont.status == STARTED
                # *prac_idk_button* updates
                
                # if prac_idk_button is starting this frame...
                if prac_idk_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_idk_button.frameNStart = frameN  # exact frame index
                    prac_idk_button.tStart = t  # local t and not account for scr refresh
                    prac_idk_button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_idk_button, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_idk_button.started')
                    # update status
                    prac_idk_button.status = STARTED
                    win.callOnFlip(prac_idk_button.buttonClock.reset)
                    prac_idk_button.setAutoDraw(True)
                
                # if prac_idk_button is active this frame...
                if prac_idk_button.status == STARTED:
                    # update params
                    pass
                    # check whether prac_idk_button has been pressed
                    if prac_idk_button.isClicked:
                        if not prac_idk_button.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac_idk_button.timesOn.append(prac_idk_button.buttonClock.getTime())
                            prac_idk_button.timesOff.append(prac_idk_button.buttonClock.getTime())
                        elif len(prac_idk_button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac_idk_button.timesOff[-1] = prac_idk_button.buttonClock.getTime()
                        if not prac_idk_button.wasClicked:
                            # run callback code when prac_idk_button is clicked
                            pass
                # take note of whether prac_idk_button was clicked, so that next frame we know if clicks are new
                prac_idk_button.wasClicked = prac_idk_button.isClicked and prac_idk_button.status == STARTED
                # Run 'Each Frame' code from prac_TextboxCode
                #Cont. only allowed after answering textbox and pressing continue. 
                if prac_idk_button.numClicks > 0 and not dont_remember_clicked:
                    prac_OldTextbox.setText('idk')
                    dont_remember_clicked = True
                    
                prac_typed_name = prac_OldTextbox.text
                
                if prac_TextboxCont.wasClicked and prac_OldTextbox.text != "":
                    continueRoutine = False
                    
                
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac1_textbox,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac1_textbox.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac1_textbox.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac1_textbox" ---
            for thisComponent in prac1_textbox.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac1_textbox
            prac1_textbox.tStop = globalClock.getTime(format='float')
            prac1_textbox.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac1_textbox.stopped', prac1_textbox.tStop)
            prac_test_loop.addData('prac_OldTextbox.text',prac_OldTextbox.text)
            prac_test_loop.addData('prac_TextboxCont.numClicks', prac_TextboxCont.numClicks)
            if prac_TextboxCont.numClicks:
               prac_test_loop.addData('prac_TextboxCont.timesOn', prac_TextboxCont.timesOn)
               prac_test_loop.addData('prac_TextboxCont.timesOff', prac_TextboxCont.timesOff)
            else:
               prac_test_loop.addData('prac_TextboxCont.timesOn', "")
               prac_test_loop.addData('prac_TextboxCont.timesOff', "")
            prac_test_loop.addData('prac_idk_button.numClicks', prac_idk_button.numClicks)
            if prac_idk_button.numClicks:
               prac_test_loop.addData('prac_idk_button.timesOn', prac_idk_button.timesOn)
               prac_test_loop.addData('prac_idk_button.timesOff', prac_idk_button.timesOff)
            else:
               prac_test_loop.addData('prac_idk_button.timesOn', "")
               prac_test_loop.addData('prac_idk_button.timesOff', "")
            # the Routine "prac1_textbox" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac1_confidence2" ---
            # create an object to store info about Routine prac1_confidence2
            prac1_confidence2 = data.Routine(
                name='prac1_confidence2',
                components=[prac_image4, prac_question2, prac_conf2, prac_conf2_cont],
            )
            prac1_confidence2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from prac_conf2_code
            if prac_typed_name == "idk":
                continueRoutine = False
            prac_conf2.reset()
            # reset prac_conf2_cont to account for continued clicks & clear times on/off
            prac_conf2_cont.reset()
            # store start times for prac1_confidence2
            prac1_confidence2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac1_confidence2.tStart = globalClock.getTime(format='float')
            prac1_confidence2.status = STARTED
            thisExp.addData('prac1_confidence2.started', prac1_confidence2.tStart)
            prac1_confidence2.maxDuration = None
            # skip Routine prac1_confidence2 if its 'Skip if' condition is True
            prac1_confidence2.skipped = continueRoutine and not (prac_TestResp != "old")
            continueRoutine = prac1_confidence2.skipped
            # keep track of which components have finished
            prac1_confidence2Components = prac1_confidence2.components
            for thisComponent in prac1_confidence2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac1_confidence2" ---
            prac1_confidence2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from prac_conf2_code
                #Cont. only allowed after answering slider and pressing continue. 
                if prac_conf2_cont.isClicked and prac_conf2.getRating() is not None:
                    continueRoutine = False
                
                # *prac_image4* updates
                
                # if prac_image4 is starting this frame...
                if prac_image4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_image4.frameNStart = frameN  # exact frame index
                    prac_image4.tStart = t  # local t and not account for scr refresh
                    prac_image4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_image4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_image4.started')
                    # update status
                    prac_image4.status = STARTED
                    prac_image4.setAutoDraw(True)
                
                # if prac_image4 is active this frame...
                if prac_image4.status == STARTED:
                    # update params
                    pass
                
                # *prac_question2* updates
                
                # if prac_question2 is starting this frame...
                if prac_question2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_question2.frameNStart = frameN  # exact frame index
                    prac_question2.tStart = t  # local t and not account for scr refresh
                    prac_question2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_question2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_question2.started')
                    # update status
                    prac_question2.status = STARTED
                    prac_question2.setAutoDraw(True)
                
                # if prac_question2 is active this frame...
                if prac_question2.status == STARTED:
                    # update params
                    pass
                
                # *prac_conf2* updates
                
                # if prac_conf2 is starting this frame...
                if prac_conf2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_conf2.frameNStart = frameN  # exact frame index
                    prac_conf2.tStart = t  # local t and not account for scr refresh
                    prac_conf2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_conf2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_conf2.started')
                    # update status
                    prac_conf2.status = STARTED
                    prac_conf2.setAutoDraw(True)
                
                # if prac_conf2 is active this frame...
                if prac_conf2.status == STARTED:
                    # update params
                    pass
                # *prac_conf2_cont* updates
                
                # if prac_conf2_cont is starting this frame...
                if prac_conf2_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_conf2_cont.frameNStart = frameN  # exact frame index
                    prac_conf2_cont.tStart = t  # local t and not account for scr refresh
                    prac_conf2_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_conf2_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_conf2_cont.started')
                    # update status
                    prac_conf2_cont.status = STARTED
                    win.callOnFlip(prac_conf2_cont.buttonClock.reset)
                    prac_conf2_cont.setAutoDraw(True)
                
                # if prac_conf2_cont is active this frame...
                if prac_conf2_cont.status == STARTED:
                    # update params
                    pass
                    # check whether prac_conf2_cont has been pressed
                    if prac_conf2_cont.isClicked:
                        if not prac_conf2_cont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac_conf2_cont.timesOn.append(prac_conf2_cont.buttonClock.getTime())
                            prac_conf2_cont.timesOff.append(prac_conf2_cont.buttonClock.getTime())
                        elif len(prac_conf2_cont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac_conf2_cont.timesOff[-1] = prac_conf2_cont.buttonClock.getTime()
                        if not prac_conf2_cont.wasClicked:
                            # run callback code when prac_conf2_cont is clicked
                            pass
                # take note of whether prac_conf2_cont was clicked, so that next frame we know if clicks are new
                prac_conf2_cont.wasClicked = prac_conf2_cont.isClicked and prac_conf2_cont.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac1_confidence2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac1_confidence2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac1_confidence2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac1_confidence2" ---
            for thisComponent in prac1_confidence2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac1_confidence2
            prac1_confidence2.tStop = globalClock.getTime(format='float')
            prac1_confidence2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac1_confidence2.stopped', prac1_confidence2.tStop)
            # Run 'End Routine' code from prac_conf2_code
            prac_feedback = ""
            
            #show the following feedback depending on the participants answer.
            if prac_TestResp == "old":
                prac_feedback = "I am sorry, that answer is incorrect. This face was never shown"
            elif prac_TestResp == "new":
                prac_feedback = "Correct! You should not remember this face because it was not shown in the previous trials"
            prac_test_loop.addData('prac_conf2.response', prac_conf2.getRating())
            prac_test_loop.addData('prac_conf2.rt', prac_conf2.getRT())
            prac_test_loop.addData('prac_conf2_cont.numClicks', prac_conf2_cont.numClicks)
            if prac_conf2_cont.numClicks:
               prac_test_loop.addData('prac_conf2_cont.timesOn', prac_conf2_cont.timesOn)
               prac_test_loop.addData('prac_conf2_cont.timesOff', prac_conf2_cont.timesOff)
            else:
               prac_test_loop.addData('prac_conf2_cont.timesOn', "")
               prac_test_loop.addData('prac_conf2_cont.timesOff', "")
            # the Routine "prac1_confidence2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac1_respFeedback" ---
            # create an object to store info about Routine prac1_respFeedback
            prac1_respFeedback = data.Routine(
                name='prac1_respFeedback',
                components=[feedback_text, prac1_cont, prac_cont3],
            )
            prac1_respFeedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            feedback_text.setText(prac_feedback)
            # create starting attributes for prac_cont3
            prac_cont3.keys = []
            prac_cont3.rt = []
            _prac_cont3_allKeys = []
            # store start times for prac1_respFeedback
            prac1_respFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac1_respFeedback.tStart = globalClock.getTime(format='float')
            prac1_respFeedback.status = STARTED
            thisExp.addData('prac1_respFeedback.started', prac1_respFeedback.tStart)
            prac1_respFeedback.maxDuration = None
            # keep track of which components have finished
            prac1_respFeedbackComponents = prac1_respFeedback.components
            for thisComponent in prac1_respFeedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac1_respFeedback" ---
            prac1_respFeedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *feedback_text* updates
                
                # if feedback_text is starting this frame...
                if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_text.frameNStart = frameN  # exact frame index
                    feedback_text.tStart = t  # local t and not account for scr refresh
                    feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_text.started')
                    # update status
                    feedback_text.status = STARTED
                    feedback_text.setAutoDraw(True)
                
                # if feedback_text is active this frame...
                if feedback_text.status == STARTED:
                    # update params
                    pass
                
                # *prac1_cont* updates
                
                # if prac1_cont is starting this frame...
                if prac1_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    prac1_cont.frameNStart = frameN  # exact frame index
                    prac1_cont.tStart = t  # local t and not account for scr refresh
                    prac1_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac1_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac1_cont.started')
                    # update status
                    prac1_cont.status = STARTED
                    prac1_cont.setAutoDraw(True)
                
                # if prac1_cont is active this frame...
                if prac1_cont.status == STARTED:
                    # update params
                    pass
                
                # *prac_cont3* updates
                waitOnFlip = False
                
                # if prac_cont3 is starting this frame...
                if prac_cont3.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    prac_cont3.frameNStart = frameN  # exact frame index
                    prac_cont3.tStart = t  # local t and not account for scr refresh
                    prac_cont3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_cont3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont3.started')
                    # update status
                    prac_cont3.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_cont3.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_cont3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac_cont3.status == STARTED and not waitOnFlip:
                    theseKeys = prac_cont3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_cont3_allKeys.extend(theseKeys)
                    if len(_prac_cont3_allKeys):
                        prac_cont3.keys = _prac_cont3_allKeys[-1].name  # just the last key pressed
                        prac_cont3.rt = _prac_cont3_allKeys[-1].rt
                        prac_cont3.duration = _prac_cont3_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac1_respFeedback,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac1_respFeedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac1_respFeedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac1_respFeedback" ---
            for thisComponent in prac1_respFeedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac1_respFeedback
            prac1_respFeedback.tStop = globalClock.getTime(format='float')
            prac1_respFeedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac1_respFeedback.stopped', prac1_respFeedback.tStop)
            # check responses
            if prac_cont3.keys in ['', [], None]:  # No response was made
                prac_cont3.keys = None
            prac_test_loop.addData('prac_cont3.keys',prac_cont3.keys)
            if prac_cont3.keys != None:  # we had a response
                prac_test_loop.addData('prac_cont3.rt', prac_cont3.rt)
                prac_test_loop.addData('prac_cont3.duration', prac_cont3.duration)
            # the Routine "prac1_respFeedback" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac_test2" ---
            # create an object to store info about Routine prac_test2
            prac_test2 = data.Routine(
                name='prac_test2',
                components=[prac2_isi, prac2_image, prac2_OldNew, prac2_test_resp, prac2_left_resp, prac2_right_resp],
            )
            prac_test2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from prac2_TestCode
            #set black for participant input
            prac_TestResp = None
            prac2_image.setImage('practice_trials/Black_Hair_Derpina.jpg')
            # create starting attributes for prac2_test_resp
            prac2_test_resp.keys = []
            prac2_test_resp.rt = []
            _prac2_test_resp_allKeys = []
            prac2_left_resp.setText(left_label_text)
            prac2_right_resp.setText(right_label_text)
            # store start times for prac_test2
            prac_test2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_test2.tStart = globalClock.getTime(format='float')
            prac_test2.status = STARTED
            thisExp.addData('prac_test2.started', prac_test2.tStart)
            prac_test2.maxDuration = None
            # keep track of which components have finished
            prac_test2Components = prac_test2.components
            for thisComponent in prac_test2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_test2" ---
            prac_test2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac2_isi* updates
                
                # if prac2_isi is starting this frame...
                if prac2_isi.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_isi.frameNStart = frameN  # exact frame index
                    prac2_isi.tStart = t  # local t and not account for scr refresh
                    prac2_isi.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_isi, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_isi.started')
                    # update status
                    prac2_isi.status = STARTED
                    prac2_isi.setAutoDraw(True)
                
                # if prac2_isi is active this frame...
                if prac2_isi.status == STARTED:
                    # update params
                    pass
                
                # if prac2_isi is stopping this frame...
                if prac2_isi.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac2_isi.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        prac2_isi.tStop = t  # not accounting for scr refresh
                        prac2_isi.tStopRefresh = tThisFlipGlobal  # on global time
                        prac2_isi.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac2_isi.stopped')
                        # update status
                        prac2_isi.status = FINISHED
                        prac2_isi.setAutoDraw(False)
                
                # *prac2_image* updates
                
                # if prac2_image is starting this frame...
                if prac2_image.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_image.frameNStart = frameN  # exact frame index
                    prac2_image.tStart = t  # local t and not account for scr refresh
                    prac2_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_image.started')
                    # update status
                    prac2_image.status = STARTED
                    prac2_image.setAutoDraw(True)
                
                # if prac2_image is active this frame...
                if prac2_image.status == STARTED:
                    # update params
                    pass
                
                # *prac2_OldNew* updates
                
                # if prac2_OldNew is starting this frame...
                if prac2_OldNew.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_OldNew.frameNStart = frameN  # exact frame index
                    prac2_OldNew.tStart = t  # local t and not account for scr refresh
                    prac2_OldNew.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_OldNew, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_OldNew.started')
                    # update status
                    prac2_OldNew.status = STARTED
                    prac2_OldNew.setAutoDraw(True)
                
                # if prac2_OldNew is active this frame...
                if prac2_OldNew.status == STARTED:
                    # update params
                    pass
                
                # *prac2_test_resp* updates
                waitOnFlip = False
                
                # if prac2_test_resp is starting this frame...
                if prac2_test_resp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_test_resp.frameNStart = frameN  # exact frame index
                    prac2_test_resp.tStart = t  # local t and not account for scr refresh
                    prac2_test_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_test_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_test_resp.started')
                    # update status
                    prac2_test_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac2_test_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac2_test_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac2_test_resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac2_test_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _prac2_test_resp_allKeys.extend(theseKeys)
                    if len(_prac2_test_resp_allKeys):
                        prac2_test_resp.keys = _prac2_test_resp_allKeys[-1].name  # just the last key pressed
                        prac2_test_resp.rt = _prac2_test_resp_allKeys[-1].rt
                        prac2_test_resp.duration = _prac2_test_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *prac2_left_resp* updates
                
                # if prac2_left_resp is starting this frame...
                if prac2_left_resp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_left_resp.frameNStart = frameN  # exact frame index
                    prac2_left_resp.tStart = t  # local t and not account for scr refresh
                    prac2_left_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_left_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_left_resp.started')
                    # update status
                    prac2_left_resp.status = STARTED
                    prac2_left_resp.setAutoDraw(True)
                
                # if prac2_left_resp is active this frame...
                if prac2_left_resp.status == STARTED:
                    # update params
                    pass
                
                # *prac2_right_resp* updates
                
                # if prac2_right_resp is starting this frame...
                if prac2_right_resp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_right_resp.frameNStart = frameN  # exact frame index
                    prac2_right_resp.tStart = t  # local t and not account for scr refresh
                    prac2_right_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_right_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_right_resp.started')
                    # update status
                    prac2_right_resp.status = STARTED
                    prac2_right_resp.setAutoDraw(True)
                
                # if prac2_right_resp is active this frame...
                if prac2_right_resp.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac_test2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_test2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_test2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_test2" ---
            for thisComponent in prac_test2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_test2
            prac_test2.tStop = globalClock.getTime(format='float')
            prac_test2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_test2.stopped', prac_test2.tStop)
            # Run 'End Routine' code from prac2_TestCode
            #Log response
            
            if prac2_test_resp.keys == yes_key:
                prac2_TestResp = 'old'
            elif prac2_test_resp.keys == no_key:
                prac2_TestResp = 'new'
                prac2_typed_name = ""
            thisExp.addData('prac2_TestResp', prac2_TestResp)
                    
            
            #add to all answers if correct
            practice_test_correct.append(prac2_TestResp == "old")
            # check responses
            if prac2_test_resp.keys in ['', [], None]:  # No response was made
                prac2_test_resp.keys = None
            prac_test_loop.addData('prac2_test_resp.keys',prac2_test_resp.keys)
            if prac2_test_resp.keys != None:  # we had a response
                prac_test_loop.addData('prac2_test_resp.rt', prac2_test_resp.rt)
                prac_test_loop.addData('prac2_test_resp.duration', prac2_test_resp.duration)
            # the Routine "prac_test2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac2_confidence" ---
            # create an object to store info about Routine prac2_confidence
            prac2_confidence = data.Routine(
                name='prac2_confidence',
                components=[prac2_image2, prac2_question, prac2_conf1, prac2_conf1_cont],
            )
            prac2_confidence.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac2_conf1.reset()
            # reset prac2_conf1_cont to account for continued clicks & clear times on/off
            prac2_conf1_cont.reset()
            # store start times for prac2_confidence
            prac2_confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac2_confidence.tStart = globalClock.getTime(format='float')
            prac2_confidence.status = STARTED
            thisExp.addData('prac2_confidence.started', prac2_confidence.tStart)
            prac2_confidence.maxDuration = None
            # keep track of which components have finished
            prac2_confidenceComponents = prac2_confidence.components
            for thisComponent in prac2_confidence.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac2_confidence" ---
            prac2_confidence.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac2_image2* updates
                
                # if prac2_image2 is starting this frame...
                if prac2_image2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_image2.frameNStart = frameN  # exact frame index
                    prac2_image2.tStart = t  # local t and not account for scr refresh
                    prac2_image2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_image2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_image2.started')
                    # update status
                    prac2_image2.status = STARTED
                    prac2_image2.setAutoDraw(True)
                
                # if prac2_image2 is active this frame...
                if prac2_image2.status == STARTED:
                    # update params
                    pass
                
                # *prac2_question* updates
                
                # if prac2_question is starting this frame...
                if prac2_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_question.frameNStart = frameN  # exact frame index
                    prac2_question.tStart = t  # local t and not account for scr refresh
                    prac2_question.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_question, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_question.started')
                    # update status
                    prac2_question.status = STARTED
                    prac2_question.setAutoDraw(True)
                
                # if prac2_question is active this frame...
                if prac2_question.status == STARTED:
                    # update params
                    pass
                
                # *prac2_conf1* updates
                
                # if prac2_conf1 is starting this frame...
                if prac2_conf1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_conf1.frameNStart = frameN  # exact frame index
                    prac2_conf1.tStart = t  # local t and not account for scr refresh
                    prac2_conf1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_conf1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_conf1.started')
                    # update status
                    prac2_conf1.status = STARTED
                    prac2_conf1.setAutoDraw(True)
                
                # if prac2_conf1 is active this frame...
                if prac2_conf1.status == STARTED:
                    # update params
                    pass
                # *prac2_conf1_cont* updates
                
                # if prac2_conf1_cont is starting this frame...
                if prac2_conf1_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_conf1_cont.frameNStart = frameN  # exact frame index
                    prac2_conf1_cont.tStart = t  # local t and not account for scr refresh
                    prac2_conf1_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_conf1_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_conf1_cont.started')
                    # update status
                    prac2_conf1_cont.status = STARTED
                    win.callOnFlip(prac2_conf1_cont.buttonClock.reset)
                    prac2_conf1_cont.setAutoDraw(True)
                
                # if prac2_conf1_cont is active this frame...
                if prac2_conf1_cont.status == STARTED:
                    # update params
                    pass
                    # check whether prac2_conf1_cont has been pressed
                    if prac2_conf1_cont.isClicked:
                        if not prac2_conf1_cont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac2_conf1_cont.timesOn.append(prac2_conf1_cont.buttonClock.getTime())
                            prac2_conf1_cont.timesOff.append(prac2_conf1_cont.buttonClock.getTime())
                        elif len(prac2_conf1_cont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac2_conf1_cont.timesOff[-1] = prac2_conf1_cont.buttonClock.getTime()
                        if not prac2_conf1_cont.wasClicked:
                            # run callback code when prac2_conf1_cont is clicked
                            pass
                # take note of whether prac2_conf1_cont was clicked, so that next frame we know if clicks are new
                prac2_conf1_cont.wasClicked = prac2_conf1_cont.isClicked and prac2_conf1_cont.status == STARTED
                # Run 'Each Frame' code from prac2_conf1_code
                #Cont. only allowed after answering slider and pressing continue. 
                if prac2_conf1_cont.isClicked and prac2_conf1.getRating() is not None:
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac2_confidence,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac2_confidence.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac2_confidence.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac2_confidence" ---
            for thisComponent in prac2_confidence.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac2_confidence
            prac2_confidence.tStop = globalClock.getTime(format='float')
            prac2_confidence.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac2_confidence.stopped', prac2_confidence.tStop)
            prac_test_loop.addData('prac2_conf1.response', prac2_conf1.getRating())
            prac_test_loop.addData('prac2_conf1.rt', prac2_conf1.getRT())
            prac_test_loop.addData('prac2_conf1_cont.numClicks', prac2_conf1_cont.numClicks)
            if prac2_conf1_cont.numClicks:
               prac_test_loop.addData('prac2_conf1_cont.timesOn', prac2_conf1_cont.timesOn)
               prac_test_loop.addData('prac2_conf1_cont.timesOff', prac2_conf1_cont.timesOff)
            else:
               prac_test_loop.addData('prac2_conf1_cont.timesOn', "")
               prac_test_loop.addData('prac2_conf1_cont.timesOff', "")
            # the Routine "prac2_confidence" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac2_textbox" ---
            # create an object to store info about Routine prac2_textbox
            prac2_textbox = data.Routine(
                name='prac2_textbox',
                components=[prac2_image3, prac2_OldTextbox, prac2_TextboxCont, prac2_idk_button],
            )
            prac2_textbox.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac2_OldTextbox.reset()
            prac2_OldTextbox.setText('')
            prac2_OldTextbox.setPlaceholder('Name?')
            # reset prac2_TextboxCont to account for continued clicks & clear times on/off
            prac2_TextboxCont.reset()
            # reset prac2_idk_button to account for continued clicks & clear times on/off
            prac2_idk_button.reset()
            # Run 'Begin Routine' code from prac2_TextboxCode
            dont_remember_clicked = False
            # store start times for prac2_textbox
            prac2_textbox.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac2_textbox.tStart = globalClock.getTime(format='float')
            prac2_textbox.status = STARTED
            thisExp.addData('prac2_textbox.started', prac2_textbox.tStart)
            prac2_textbox.maxDuration = None
            # skip Routine prac2_textbox if its 'Skip if' condition is True
            prac2_textbox.skipped = continueRoutine and not (prac2_TestResp != "old")
            continueRoutine = prac2_textbox.skipped
            # keep track of which components have finished
            prac2_textboxComponents = prac2_textbox.components
            for thisComponent in prac2_textbox.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac2_textbox" ---
            prac2_textbox.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac2_image3* updates
                
                # if prac2_image3 is starting this frame...
                if prac2_image3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_image3.frameNStart = frameN  # exact frame index
                    prac2_image3.tStart = t  # local t and not account for scr refresh
                    prac2_image3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_image3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_image3.started')
                    # update status
                    prac2_image3.status = STARTED
                    prac2_image3.setAutoDraw(True)
                
                # if prac2_image3 is active this frame...
                if prac2_image3.status == STARTED:
                    # update params
                    pass
                
                # *prac2_OldTextbox* updates
                
                # if prac2_OldTextbox is starting this frame...
                if prac2_OldTextbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_OldTextbox.frameNStart = frameN  # exact frame index
                    prac2_OldTextbox.tStart = t  # local t and not account for scr refresh
                    prac2_OldTextbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_OldTextbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_OldTextbox.started')
                    # update status
                    prac2_OldTextbox.status = STARTED
                    prac2_OldTextbox.setAutoDraw(True)
                
                # if prac2_OldTextbox is active this frame...
                if prac2_OldTextbox.status == STARTED:
                    # update params
                    pass
                # *prac2_TextboxCont* updates
                
                # if prac2_TextboxCont is starting this frame...
                if prac2_TextboxCont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_TextboxCont.frameNStart = frameN  # exact frame index
                    prac2_TextboxCont.tStart = t  # local t and not account for scr refresh
                    prac2_TextboxCont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_TextboxCont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_TextboxCont.started')
                    # update status
                    prac2_TextboxCont.status = STARTED
                    win.callOnFlip(prac2_TextboxCont.buttonClock.reset)
                    prac2_TextboxCont.setAutoDraw(True)
                
                # if prac2_TextboxCont is active this frame...
                if prac2_TextboxCont.status == STARTED:
                    # update params
                    pass
                    # check whether prac2_TextboxCont has been pressed
                    if prac2_TextboxCont.isClicked:
                        if not prac2_TextboxCont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac2_TextboxCont.timesOn.append(prac2_TextboxCont.buttonClock.getTime())
                            prac2_TextboxCont.timesOff.append(prac2_TextboxCont.buttonClock.getTime())
                        elif len(prac2_TextboxCont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac2_TextboxCont.timesOff[-1] = prac2_TextboxCont.buttonClock.getTime()
                        if not prac2_TextboxCont.wasClicked:
                            # run callback code when prac2_TextboxCont is clicked
                            pass
                # take note of whether prac2_TextboxCont was clicked, so that next frame we know if clicks are new
                prac2_TextboxCont.wasClicked = prac2_TextboxCont.isClicked and prac2_TextboxCont.status == STARTED
                # *prac2_idk_button* updates
                
                # if prac2_idk_button is starting this frame...
                if prac2_idk_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_idk_button.frameNStart = frameN  # exact frame index
                    prac2_idk_button.tStart = t  # local t and not account for scr refresh
                    prac2_idk_button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_idk_button, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_idk_button.started')
                    # update status
                    prac2_idk_button.status = STARTED
                    win.callOnFlip(prac2_idk_button.buttonClock.reset)
                    prac2_idk_button.setAutoDraw(True)
                
                # if prac2_idk_button is active this frame...
                if prac2_idk_button.status == STARTED:
                    # update params
                    pass
                    # check whether prac2_idk_button has been pressed
                    if prac2_idk_button.isClicked:
                        if not prac2_idk_button.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac2_idk_button.timesOn.append(prac2_idk_button.buttonClock.getTime())
                            prac2_idk_button.timesOff.append(prac2_idk_button.buttonClock.getTime())
                        elif len(prac2_idk_button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac2_idk_button.timesOff[-1] = prac2_idk_button.buttonClock.getTime()
                        if not prac2_idk_button.wasClicked:
                            # run callback code when prac2_idk_button is clicked
                            pass
                # take note of whether prac2_idk_button was clicked, so that next frame we know if clicks are new
                prac2_idk_button.wasClicked = prac2_idk_button.isClicked and prac2_idk_button.status == STARTED
                # Run 'Each Frame' code from prac2_TextboxCode
                #Cont. only allowed after answering textbox and pressing continue. 
                if prac2_idk_button.numClicks > 0 and not dont_remember_clicked:
                    prac2_OldTextbox.setText('idk')
                    dont_remember_clicked = True
                    
                prac2_typed_name = prac2_OldTextbox.text
                
                if prac2_TextboxCont.wasClicked and prac2_OldTextbox.text != "":
                    continueRoutine = False
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac2_textbox,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac2_textbox.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac2_textbox.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac2_textbox" ---
            for thisComponent in prac2_textbox.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac2_textbox
            prac2_textbox.tStop = globalClock.getTime(format='float')
            prac2_textbox.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac2_textbox.stopped', prac2_textbox.tStop)
            prac_test_loop.addData('prac2_OldTextbox.text',prac2_OldTextbox.text)
            prac_test_loop.addData('prac2_TextboxCont.numClicks', prac2_TextboxCont.numClicks)
            if prac2_TextboxCont.numClicks:
               prac_test_loop.addData('prac2_TextboxCont.timesOn', prac2_TextboxCont.timesOn)
               prac_test_loop.addData('prac2_TextboxCont.timesOff', prac2_TextboxCont.timesOff)
            else:
               prac_test_loop.addData('prac2_TextboxCont.timesOn', "")
               prac_test_loop.addData('prac2_TextboxCont.timesOff', "")
            prac_test_loop.addData('prac2_idk_button.numClicks', prac2_idk_button.numClicks)
            if prac2_idk_button.numClicks:
               prac_test_loop.addData('prac2_idk_button.timesOn', prac2_idk_button.timesOn)
               prac_test_loop.addData('prac2_idk_button.timesOff', prac2_idk_button.timesOff)
            else:
               prac_test_loop.addData('prac2_idk_button.timesOn', "")
               prac_test_loop.addData('prac2_idk_button.timesOff', "")
            # the Routine "prac2_textbox" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac2_confidence2" ---
            # create an object to store info about Routine prac2_confidence2
            prac2_confidence2 = data.Routine(
                name='prac2_confidence2',
                components=[prac2_image4, prac2_question2, prac2_conf2, prac2_conf2_cont],
            )
            prac2_confidence2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from prac2_conf2_code
            if prac2_typed_name == "idk":
                continueRoutine = False
            prac2_conf2.reset()
            # reset prac2_conf2_cont to account for continued clicks & clear times on/off
            prac2_conf2_cont.reset()
            # store start times for prac2_confidence2
            prac2_confidence2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac2_confidence2.tStart = globalClock.getTime(format='float')
            prac2_confidence2.status = STARTED
            thisExp.addData('prac2_confidence2.started', prac2_confidence2.tStart)
            prac2_confidence2.maxDuration = None
            # skip Routine prac2_confidence2 if its 'Skip if' condition is True
            prac2_confidence2.skipped = continueRoutine and not (prac2_TestResp == "new")
            continueRoutine = prac2_confidence2.skipped
            # keep track of which components have finished
            prac2_confidence2Components = prac2_confidence2.components
            for thisComponent in prac2_confidence2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac2_confidence2" ---
            prac2_confidence2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from prac2_conf2_code
                #Cont. only allowed after answering slider and pressing continue. 
                if prac2_conf2_cont.isClicked and prac2_conf2.getRating() is not None:
                    continueRoutine = False
                
                # *prac2_image4* updates
                
                # if prac2_image4 is starting this frame...
                if prac2_image4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_image4.frameNStart = frameN  # exact frame index
                    prac2_image4.tStart = t  # local t and not account for scr refresh
                    prac2_image4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_image4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_image4.started')
                    # update status
                    prac2_image4.status = STARTED
                    prac2_image4.setAutoDraw(True)
                
                # if prac2_image4 is active this frame...
                if prac2_image4.status == STARTED:
                    # update params
                    pass
                
                # *prac2_question2* updates
                
                # if prac2_question2 is starting this frame...
                if prac2_question2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_question2.frameNStart = frameN  # exact frame index
                    prac2_question2.tStart = t  # local t and not account for scr refresh
                    prac2_question2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_question2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_question2.started')
                    # update status
                    prac2_question2.status = STARTED
                    prac2_question2.setAutoDraw(True)
                
                # if prac2_question2 is active this frame...
                if prac2_question2.status == STARTED:
                    # update params
                    pass
                
                # *prac2_conf2* updates
                
                # if prac2_conf2 is starting this frame...
                if prac2_conf2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_conf2.frameNStart = frameN  # exact frame index
                    prac2_conf2.tStart = t  # local t and not account for scr refresh
                    prac2_conf2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_conf2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_conf2.started')
                    # update status
                    prac2_conf2.status = STARTED
                    prac2_conf2.setAutoDraw(True)
                
                # if prac2_conf2 is active this frame...
                if prac2_conf2.status == STARTED:
                    # update params
                    pass
                # *prac2_conf2_cont* updates
                
                # if prac2_conf2_cont is starting this frame...
                if prac2_conf2_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_conf2_cont.frameNStart = frameN  # exact frame index
                    prac2_conf2_cont.tStart = t  # local t and not account for scr refresh
                    prac2_conf2_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_conf2_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_conf2_cont.started')
                    # update status
                    prac2_conf2_cont.status = STARTED
                    win.callOnFlip(prac2_conf2_cont.buttonClock.reset)
                    prac2_conf2_cont.setAutoDraw(True)
                
                # if prac2_conf2_cont is active this frame...
                if prac2_conf2_cont.status == STARTED:
                    # update params
                    pass
                    # check whether prac2_conf2_cont has been pressed
                    if prac2_conf2_cont.isClicked:
                        if not prac2_conf2_cont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            prac2_conf2_cont.timesOn.append(prac2_conf2_cont.buttonClock.getTime())
                            prac2_conf2_cont.timesOff.append(prac2_conf2_cont.buttonClock.getTime())
                        elif len(prac2_conf2_cont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            prac2_conf2_cont.timesOff[-1] = prac2_conf2_cont.buttonClock.getTime()
                        if not prac2_conf2_cont.wasClicked:
                            # run callback code when prac2_conf2_cont is clicked
                            pass
                # take note of whether prac2_conf2_cont was clicked, so that next frame we know if clicks are new
                prac2_conf2_cont.wasClicked = prac2_conf2_cont.isClicked and prac2_conf2_cont.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac2_confidence2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac2_confidence2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac2_confidence2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac2_confidence2" ---
            for thisComponent in prac2_confidence2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac2_confidence2
            prac2_confidence2.tStop = globalClock.getTime(format='float')
            prac2_confidence2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac2_confidence2.stopped', prac2_confidence2.tStop)
            # Run 'End Routine' code from prac2_conf2_code
            prac2_feedback = ""
            
            #show the following feedback depending on the participants answer.
            if prac2_TestResp == "new":
                prac2_feedback = "I am sorry, that answer is incorrect. Recall this face being shown as paired with beyonce in the second set."
            elif prac2_TestResp == "old":
                prac2_feedback = "Good Job! That is correct. This face was paired with the name 'beyonce' in the second set"
            prac_test_loop.addData('prac2_conf2.response', prac2_conf2.getRating())
            prac_test_loop.addData('prac2_conf2.rt', prac2_conf2.getRT())
            prac_test_loop.addData('prac2_conf2_cont.numClicks', prac2_conf2_cont.numClicks)
            if prac2_conf2_cont.numClicks:
               prac_test_loop.addData('prac2_conf2_cont.timesOn', prac2_conf2_cont.timesOn)
               prac_test_loop.addData('prac2_conf2_cont.timesOff', prac2_conf2_cont.timesOff)
            else:
               prac_test_loop.addData('prac2_conf2_cont.timesOn', "")
               prac_test_loop.addData('prac2_conf2_cont.timesOff', "")
            # the Routine "prac2_confidence2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac2_respFeedback" ---
            # create an object to store info about Routine prac2_respFeedback
            prac2_respFeedback = data.Routine(
                name='prac2_respFeedback',
                components=[feedback_text2, prac2_cont, prac_cont4],
            )
            prac2_respFeedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            feedback_text2.setText(prac2_feedback)
            # create starting attributes for prac_cont4
            prac_cont4.keys = []
            prac_cont4.rt = []
            _prac_cont4_allKeys = []
            # store start times for prac2_respFeedback
            prac2_respFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac2_respFeedback.tStart = globalClock.getTime(format='float')
            prac2_respFeedback.status = STARTED
            thisExp.addData('prac2_respFeedback.started', prac2_respFeedback.tStart)
            prac2_respFeedback.maxDuration = None
            # keep track of which components have finished
            prac2_respFeedbackComponents = prac2_respFeedback.components
            for thisComponent in prac2_respFeedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac2_respFeedback" ---
            prac2_respFeedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_test_loop, 'status') and thisPrac_test_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *feedback_text2* updates
                
                # if feedback_text2 is starting this frame...
                if feedback_text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_text2.frameNStart = frameN  # exact frame index
                    feedback_text2.tStart = t  # local t and not account for scr refresh
                    feedback_text2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_text2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_text2.started')
                    # update status
                    feedback_text2.status = STARTED
                    feedback_text2.setAutoDraw(True)
                
                # if feedback_text2 is active this frame...
                if feedback_text2.status == STARTED:
                    # update params
                    pass
                
                # *prac2_cont* updates
                
                # if prac2_cont is starting this frame...
                if prac2_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    prac2_cont.frameNStart = frameN  # exact frame index
                    prac2_cont.tStart = t  # local t and not account for scr refresh
                    prac2_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2_cont.started')
                    # update status
                    prac2_cont.status = STARTED
                    prac2_cont.setAutoDraw(True)
                
                # if prac2_cont is active this frame...
                if prac2_cont.status == STARTED:
                    # update params
                    pass
                
                # *prac_cont4* updates
                waitOnFlip = False
                
                # if prac_cont4 is starting this frame...
                if prac_cont4.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                    # keep track of start time/frame for later
                    prac_cont4.frameNStart = frameN  # exact frame index
                    prac_cont4.tStart = t  # local t and not account for scr refresh
                    prac_cont4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_cont4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont4.started')
                    # update status
                    prac_cont4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_cont4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_cont4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac_cont4.status == STARTED and not waitOnFlip:
                    theseKeys = prac_cont4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_cont4_allKeys.extend(theseKeys)
                    if len(_prac_cont4_allKeys):
                        prac_cont4.keys = _prac_cont4_allKeys[-1].name  # just the last key pressed
                        prac_cont4.rt = _prac_cont4_allKeys[-1].rt
                        prac_cont4.duration = _prac_cont4_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac2_respFeedback,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac2_respFeedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac2_respFeedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac2_respFeedback" ---
            for thisComponent in prac2_respFeedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac2_respFeedback
            prac2_respFeedback.tStop = globalClock.getTime(format='float')
            prac2_respFeedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac2_respFeedback.stopped', prac2_respFeedback.tStop)
            # check responses
            if prac_cont4.keys in ['', [], None]:  # No response was made
                prac_cont4.keys = None
            prac_test_loop.addData('prac_cont4.keys',prac_cont4.keys)
            if prac_cont4.keys != None:  # we had a response
                prac_test_loop.addData('prac_cont4.rt', prac_cont4.rt)
                prac_test_loop.addData('prac_cont4.duration', prac_cont4.duration)
            # Run 'End Routine' code from PracEnd_code
            #to move on, check that there are 4 answers and that they are all correct. 
            if len(practice_test_correct) == 2 and all(practice_test_correct):
                prac_test_loop.finished = True
            else:
                prac_test_loop.finished = False
            
            #clear for next practice trials
            practice_test_correct.clear()
            
            #set to null for next use, just in case
            prac_selected = None
            # the Routine "prac2_respFeedback" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisPrac_test_loop as finished
            if hasattr(thisPrac_test_loop, 'status'):
                thisPrac_test_loop.status = FINISHED
            # if awaiting a pause, pause now
            if prac_test_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                prac_test_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed prac_test_selected repeats of 'prac_test_loop'
        prac_test_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "test_instr_4" ---
        # create an object to store info about Routine test_instr_4
        test_instr_4 = data.Routine(
            name='test_instr_4',
            components=[test_instructions_4, test_resp_4, test_continue_4],
        )
        test_instr_4.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for test_resp_4
        test_resp_4.keys = []
        test_resp_4.rt = []
        _test_resp_4_allKeys = []
        # Run 'Begin Routine' code from test_trials_4
        #use corresponding test trials for the block shown
        currentBlock = block_loop.thisN + 1
        block_rows = f"{(currentBlock-1)*32}:{currentBlock*32}"
        # store start times for test_instr_4
        test_instr_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test_instr_4.tStart = globalClock.getTime(format='float')
        test_instr_4.status = STARTED
        test_instr_4.maxDuration = None
        # keep track of which components have finished
        test_instr_4Components = test_instr_4.components
        for thisComponent in test_instr_4.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_instr_4" ---
        test_instr_4.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *test_instructions_4* updates
            
            # if test_instructions_4 is starting this frame...
            if test_instructions_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_instructions_4.frameNStart = frameN  # exact frame index
                test_instructions_4.tStart = t  # local t and not account for scr refresh
                test_instructions_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_instructions_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_instructions_4.started')
                # update status
                test_instructions_4.status = STARTED
                test_instructions_4.setAutoDraw(True)
            
            # if test_instructions_4 is active this frame...
            if test_instructions_4.status == STARTED:
                # update params
                pass
            
            # *test_resp_4* updates
            waitOnFlip = False
            
            # if test_resp_4 is starting this frame...
            if test_resp_4.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_resp_4.frameNStart = frameN  # exact frame index
                test_resp_4.tStart = t  # local t and not account for scr refresh
                test_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_resp_4.started')
                # update status
                test_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(test_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(test_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if test_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = test_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _test_resp_4_allKeys.extend(theseKeys)
                if len(_test_resp_4_allKeys):
                    test_resp_4.keys = _test_resp_4_allKeys[-1].name  # just the last key pressed
                    test_resp_4.rt = _test_resp_4_allKeys[-1].rt
                    test_resp_4.duration = _test_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *test_continue_4* updates
            
            # if test_continue_4 is starting this frame...
            if test_continue_4.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                test_continue_4.frameNStart = frameN  # exact frame index
                test_continue_4.tStart = t  # local t and not account for scr refresh
                test_continue_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_continue_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_continue_4.started')
                # update status
                test_continue_4.status = STARTED
                test_continue_4.setAutoDraw(True)
            
            # if test_continue_4 is active this frame...
            if test_continue_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test_instr_4,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                test_instr_4.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_instr_4.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_instr_4" ---
        for thisComponent in test_instr_4.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test_instr_4
        test_instr_4.tStop = globalClock.getTime(format='float')
        test_instr_4.tStopRefresh = tThisFlipGlobal
        # check responses
        if test_resp_4.keys in ['', [], None]:  # No response was made
            test_resp_4.keys = None
        block_loop.addData('test_resp_4.keys',test_resp_4.keys)
        if test_resp_4.keys != None:  # we had a response
            block_loop.addData('test_resp_4.rt', test_resp_4.rt)
            block_loop.addData('test_resp_4.duration', test_resp_4.duration)
        # the Routine "test_instr_4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        test_loop = data.TrialHandler2(
            name='test_loop',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            test_file, 
            selection=block_rows
        )
        , 
            seed=None, 
        )
        thisExp.addLoop(test_loop)  # add the loop to the experiment
        thisTest_loop = test_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTest_loop.rgb)
        if thisTest_loop != None:
            for paramName in thisTest_loop:
                globals()[paramName] = thisTest_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTest_loop in test_loop:
            test_loop.status = STARTED
            if hasattr(thisTest_loop, 'status'):
                thisTest_loop.status = STARTED
            currentLoop = test_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTest_loop.rgb)
            if thisTest_loop != None:
                for paramName in thisTest_loop:
                    globals()[paramName] = thisTest_loop[paramName]
            
            # --- Prepare to start Routine "test_prep" ---
            # create an object to store info about Routine test_prep
            test_prep = data.Routine(
                name='test_prep',
                components=[test_prep_txt],
            )
            test_prep.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for test_prep
            test_prep.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            test_prep.tStart = globalClock.getTime(format='float')
            test_prep.status = STARTED
            thisExp.addData('test_prep.started', test_prep.tStart)
            test_prep.maxDuration = None
            # keep track of which components have finished
            test_prepComponents = test_prep.components
            for thisComponent in test_prep.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "test_prep" ---
            test_prep.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # if trial has changed, end Routine now
                if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *test_prep_txt* updates
                
                # if test_prep_txt is starting this frame...
                if test_prep_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_prep_txt.frameNStart = frameN  # exact frame index
                    test_prep_txt.tStart = t  # local t and not account for scr refresh
                    test_prep_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_prep_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_prep_txt.started')
                    # update status
                    test_prep_txt.status = STARTED
                    test_prep_txt.setAutoDraw(True)
                
                # if test_prep_txt is active this frame...
                if test_prep_txt.status == STARTED:
                    # update params
                    pass
                
                # if test_prep_txt is stopping this frame...
                if test_prep_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > test_prep_txt.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        test_prep_txt.tStop = t  # not accounting for scr refresh
                        test_prep_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        test_prep_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'test_prep_txt.stopped')
                        # update status
                        test_prep_txt.status = FINISHED
                        test_prep_txt.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=test_prep,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    test_prep.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in test_prep.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "test_prep" ---
            for thisComponent in test_prep.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for test_prep
            test_prep.tStop = globalClock.getTime(format='float')
            test_prep.tStopRefresh = tThisFlipGlobal
            thisExp.addData('test_prep.stopped', test_prep.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if test_prep.maxDurationReached:
                routineTimer.addTime(-test_prep.maxDuration)
            elif test_prep.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "isi" ---
            # create an object to store info about Routine isi
            isi = data.Routine(
                name='isi',
                components=[isi_test],
            )
            isi.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from test_code
            #set black for participant input
            on_response = None
            # store start times for isi
            isi.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            isi.tStart = globalClock.getTime(format='float')
            isi.status = STARTED
            thisExp.addData('isi.started', isi.tStart)
            isi.maxDuration = None
            # keep track of which components have finished
            isiComponents = isi.components
            for thisComponent in isi.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "isi" ---
            isi.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *isi_test* updates
                
                # if isi_test is starting this frame...
                if isi_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    isi_test.frameNStart = frameN  # exact frame index
                    isi_test.tStart = t  # local t and not account for scr refresh
                    isi_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(isi_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'isi_test.started')
                    # update status
                    isi_test.status = STARTED
                    isi_test.setAutoDraw(True)
                
                # if isi_test is active this frame...
                if isi_test.status == STARTED:
                    # update params
                    pass
                
                # if isi_test is stopping this frame...
                if isi_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > isi_test.tStartRefresh + random.uniform(0.5, 1.5)-frameTolerance:
                        # keep track of stop time/frame for later
                        isi_test.tStop = t  # not accounting for scr refresh
                        isi_test.tStopRefresh = tThisFlipGlobal  # on global time
                        isi_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'isi_test.stopped')
                        # update status
                        isi_test.status = FINISHED
                        isi_test.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=isi,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    isi.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in isi.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "isi" ---
            for thisComponent in isi.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for isi
            isi.tStop = globalClock.getTime(format='float')
            isi.tStopRefresh = tThisFlipGlobal
            thisExp.addData('isi.stopped', isi.tStop)
            # the Routine "isi" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "test" ---
            # create an object to store info about Routine test
            test = data.Routine(
                name='test',
                components=[test_image, test_oldnew_txt, test_on_resp, test_left_resp, test_right_resp],
            )
            test.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from TestCode
            #set black for participant input
            prac_TestResp = None
            test_image.setImage(img_path)
            # create starting attributes for test_on_resp
            test_on_resp.keys = []
            test_on_resp.rt = []
            _test_on_resp_allKeys = []
            test_left_resp.setText(left_label_text)
            test_right_resp.setText(right_label_text)
            # store start times for test
            test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            test.tStart = globalClock.getTime(format='float')
            test.status = STARTED
            thisExp.addData('test.started', test.tStart)
            test.maxDuration = None
            # keep track of which components have finished
            testComponents = test.components
            for thisComponent in test.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "test" ---
            test.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *test_image* updates
                
                # if test_image is starting this frame...
                if test_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image.frameNStart = frameN  # exact frame index
                    test_image.tStart = t  # local t and not account for scr refresh
                    test_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_image.started')
                    # update status
                    test_image.status = STARTED
                    test_image.setAutoDraw(True)
                
                # if test_image is active this frame...
                if test_image.status == STARTED:
                    # update params
                    pass
                
                # *test_oldnew_txt* updates
                
                # if test_oldnew_txt is starting this frame...
                if test_oldnew_txt.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_oldnew_txt.frameNStart = frameN  # exact frame index
                    test_oldnew_txt.tStart = t  # local t and not account for scr refresh
                    test_oldnew_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_oldnew_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_oldnew_txt.started')
                    # update status
                    test_oldnew_txt.status = STARTED
                    test_oldnew_txt.setAutoDraw(True)
                
                # if test_oldnew_txt is active this frame...
                if test_oldnew_txt.status == STARTED:
                    # update params
                    pass
                
                # *test_on_resp* updates
                waitOnFlip = False
                
                # if test_on_resp is starting this frame...
                if test_on_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_on_resp.frameNStart = frameN  # exact frame index
                    test_on_resp.tStart = t  # local t and not account for scr refresh
                    test_on_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_on_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_on_resp.started')
                    # update status
                    test_on_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(test_on_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(test_on_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if test_on_resp.status == STARTED and not waitOnFlip:
                    theseKeys = test_on_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _test_on_resp_allKeys.extend(theseKeys)
                    if len(_test_on_resp_allKeys):
                        test_on_resp.keys = _test_on_resp_allKeys[-1].name  # just the last key pressed
                        test_on_resp.rt = _test_on_resp_allKeys[-1].rt
                        test_on_resp.duration = _test_on_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *test_left_resp* updates
                
                # if test_left_resp is starting this frame...
                if test_left_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_left_resp.frameNStart = frameN  # exact frame index
                    test_left_resp.tStart = t  # local t and not account for scr refresh
                    test_left_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_left_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_left_resp.started')
                    # update status
                    test_left_resp.status = STARTED
                    test_left_resp.setAutoDraw(True)
                
                # if test_left_resp is active this frame...
                if test_left_resp.status == STARTED:
                    # update params
                    pass
                
                # *test_right_resp* updates
                
                # if test_right_resp is starting this frame...
                if test_right_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_right_resp.frameNStart = frameN  # exact frame index
                    test_right_resp.tStart = t  # local t and not account for scr refresh
                    test_right_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_right_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_right_resp.started')
                    # update status
                    test_right_resp.status = STARTED
                    test_right_resp.setAutoDraw(True)
                
                # if test_right_resp is active this frame...
                if test_right_resp.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=test,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    test.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in test.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "test" ---
            for thisComponent in test.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for test
            test.tStop = globalClock.getTime(format='float')
            test.tStopRefresh = tThisFlipGlobal
            thisExp.addData('test.stopped', test.tStop)
            # Run 'End Routine' code from TestCode
            #Log response
            if test_on_resp.keys == yes_key:
                on_response = 'old'
            elif test_on_resp.keys == no_key:
                on_response = 'new'
                typed_name = ""
            thisExp.addData('on_response', on_response)
            
            # check responses
            if test_on_resp.keys in ['', [], None]:  # No response was made
                test_on_resp.keys = None
            test_loop.addData('test_on_resp.keys',test_on_resp.keys)
            if test_on_resp.keys != None:  # we had a response
                test_loop.addData('test_on_resp.rt', test_on_resp.rt)
                test_loop.addData('test_on_resp.duration', test_on_resp.duration)
            # the Routine "test" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "resp_confidence" ---
            # create an object to store info about Routine resp_confidence
            resp_confidence = data.Routine(
                name='resp_confidence',
                components=[test_image2, conf_question1, confidence1, conf1_cont],
            )
            resp_confidence.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            test_image2.setImage(img_path)
            confidence1.reset()
            # reset conf1_cont to account for continued clicks & clear times on/off
            conf1_cont.reset()
            # store start times for resp_confidence
            resp_confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            resp_confidence.tStart = globalClock.getTime(format='float')
            resp_confidence.status = STARTED
            thisExp.addData('resp_confidence.started', resp_confidence.tStart)
            resp_confidence.maxDuration = None
            # keep track of which components have finished
            resp_confidenceComponents = resp_confidence.components
            for thisComponent in resp_confidence.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "resp_confidence" ---
            resp_confidence.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *test_image2* updates
                
                # if test_image2 is starting this frame...
                if test_image2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image2.frameNStart = frameN  # exact frame index
                    test_image2.tStart = t  # local t and not account for scr refresh
                    test_image2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_image2.started')
                    # update status
                    test_image2.status = STARTED
                    test_image2.setAutoDraw(True)
                
                # if test_image2 is active this frame...
                if test_image2.status == STARTED:
                    # update params
                    pass
                
                # *conf_question1* updates
                
                # if conf_question1 is starting this frame...
                if conf_question1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    conf_question1.frameNStart = frameN  # exact frame index
                    conf_question1.tStart = t  # local t and not account for scr refresh
                    conf_question1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(conf_question1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'conf_question1.started')
                    # update status
                    conf_question1.status = STARTED
                    conf_question1.setAutoDraw(True)
                
                # if conf_question1 is active this frame...
                if conf_question1.status == STARTED:
                    # update params
                    pass
                
                # *confidence1* updates
                
                # if confidence1 is starting this frame...
                if confidence1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confidence1.frameNStart = frameN  # exact frame index
                    confidence1.tStart = t  # local t and not account for scr refresh
                    confidence1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confidence1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'confidence1.started')
                    # update status
                    confidence1.status = STARTED
                    confidence1.setAutoDraw(True)
                
                # if confidence1 is active this frame...
                if confidence1.status == STARTED:
                    # update params
                    pass
                # *conf1_cont* updates
                
                # if conf1_cont is starting this frame...
                if conf1_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    conf1_cont.frameNStart = frameN  # exact frame index
                    conf1_cont.tStart = t  # local t and not account for scr refresh
                    conf1_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(conf1_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'conf1_cont.started')
                    # update status
                    conf1_cont.status = STARTED
                    win.callOnFlip(conf1_cont.buttonClock.reset)
                    conf1_cont.setAutoDraw(True)
                
                # if conf1_cont is active this frame...
                if conf1_cont.status == STARTED:
                    # update params
                    pass
                    # check whether conf1_cont has been pressed
                    if conf1_cont.isClicked:
                        if not conf1_cont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            conf1_cont.timesOn.append(conf1_cont.buttonClock.getTime())
                            conf1_cont.timesOff.append(conf1_cont.buttonClock.getTime())
                        elif len(conf1_cont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            conf1_cont.timesOff[-1] = conf1_cont.buttonClock.getTime()
                        if not conf1_cont.wasClicked:
                            # run callback code when conf1_cont is clicked
                            pass
                # take note of whether conf1_cont was clicked, so that next frame we know if clicks are new
                conf1_cont.wasClicked = conf1_cont.isClicked and conf1_cont.status == STARTED
                # Run 'Each Frame' code from conf1_code
                #Cont. only allowed after answering slider and pressing continue. 
                if conf1_cont.wasClicked and confidence1.getRating() is not None:
                    continueRoutine = False
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=resp_confidence,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    resp_confidence.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in resp_confidence.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "resp_confidence" ---
            for thisComponent in resp_confidence.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for resp_confidence
            resp_confidence.tStop = globalClock.getTime(format='float')
            resp_confidence.tStopRefresh = tThisFlipGlobal
            thisExp.addData('resp_confidence.stopped', resp_confidence.tStop)
            test_loop.addData('confidence1.response', confidence1.getRating())
            test_loop.addData('confidence1.rt', confidence1.getRT())
            test_loop.addData('conf1_cont.numClicks', conf1_cont.numClicks)
            if conf1_cont.numClicks:
               test_loop.addData('conf1_cont.timesOn', conf1_cont.timesOn)
               test_loop.addData('conf1_cont.timesOff', conf1_cont.timesOff)
            else:
               test_loop.addData('conf1_cont.timesOn', "")
               test_loop.addData('conf1_cont.timesOff', "")
            # the Routine "resp_confidence" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "textbox" ---
            # create an object to store info about Routine textbox
            textbox = data.Routine(
                name='textbox',
                components=[test_image3, old_textbox, textbox_cont, textbox_idk],
            )
            textbox.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            test_image3.setImage(img_path)
            old_textbox.reset()
            old_textbox.setText('')
            old_textbox.setPlaceholder('Name?')
            # reset textbox_cont to account for continued clicks & clear times on/off
            textbox_cont.reset()
            # reset textbox_idk to account for continued clicks & clear times on/off
            textbox_idk.reset()
            # Run 'Begin Routine' code from textbox_code
            dont_remember_clicked = False
            # store start times for textbox
            textbox.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            textbox.tStart = globalClock.getTime(format='float')
            textbox.status = STARTED
            thisExp.addData('textbox.started', textbox.tStart)
            textbox.maxDuration = None
            # skip Routine textbox if its 'Skip if' condition is True
            textbox.skipped = continueRoutine and not (on_response != "old")
            continueRoutine = textbox.skipped
            # keep track of which components have finished
            textboxComponents = textbox.components
            for thisComponent in textbox.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "textbox" ---
            textbox.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *test_image3* updates
                
                # if test_image3 is starting this frame...
                if test_image3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image3.frameNStart = frameN  # exact frame index
                    test_image3.tStart = t  # local t and not account for scr refresh
                    test_image3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_image3.started')
                    # update status
                    test_image3.status = STARTED
                    test_image3.setAutoDraw(True)
                
                # if test_image3 is active this frame...
                if test_image3.status == STARTED:
                    # update params
                    pass
                
                # *old_textbox* updates
                
                # if old_textbox is starting this frame...
                if old_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    old_textbox.frameNStart = frameN  # exact frame index
                    old_textbox.tStart = t  # local t and not account for scr refresh
                    old_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(old_textbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'old_textbox.started')
                    # update status
                    old_textbox.status = STARTED
                    old_textbox.setAutoDraw(True)
                
                # if old_textbox is active this frame...
                if old_textbox.status == STARTED:
                    # update params
                    pass
                # *textbox_cont* updates
                
                # if textbox_cont is starting this frame...
                if textbox_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_cont.frameNStart = frameN  # exact frame index
                    textbox_cont.tStart = t  # local t and not account for scr refresh
                    textbox_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_cont.started')
                    # update status
                    textbox_cont.status = STARTED
                    win.callOnFlip(textbox_cont.buttonClock.reset)
                    textbox_cont.setAutoDraw(True)
                
                # if textbox_cont is active this frame...
                if textbox_cont.status == STARTED:
                    # update params
                    pass
                    # check whether textbox_cont has been pressed
                    if textbox_cont.isClicked:
                        if not textbox_cont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            textbox_cont.timesOn.append(textbox_cont.buttonClock.getTime())
                            textbox_cont.timesOff.append(textbox_cont.buttonClock.getTime())
                        elif len(textbox_cont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            textbox_cont.timesOff[-1] = textbox_cont.buttonClock.getTime()
                        if not textbox_cont.wasClicked:
                            # run callback code when textbox_cont is clicked
                            pass
                # take note of whether textbox_cont was clicked, so that next frame we know if clicks are new
                textbox_cont.wasClicked = textbox_cont.isClicked and textbox_cont.status == STARTED
                # *textbox_idk* updates
                
                # if textbox_idk is starting this frame...
                if textbox_idk.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_idk.frameNStart = frameN  # exact frame index
                    textbox_idk.tStart = t  # local t and not account for scr refresh
                    textbox_idk.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_idk, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_idk.started')
                    # update status
                    textbox_idk.status = STARTED
                    win.callOnFlip(textbox_idk.buttonClock.reset)
                    textbox_idk.setAutoDraw(True)
                
                # if textbox_idk is active this frame...
                if textbox_idk.status == STARTED:
                    # update params
                    pass
                    # check whether textbox_idk has been pressed
                    if textbox_idk.isClicked:
                        if not textbox_idk.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            textbox_idk.timesOn.append(textbox_idk.buttonClock.getTime())
                            textbox_idk.timesOff.append(textbox_idk.buttonClock.getTime())
                        elif len(textbox_idk.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            textbox_idk.timesOff[-1] = textbox_idk.buttonClock.getTime()
                        if not textbox_idk.wasClicked:
                            # run callback code when textbox_idk is clicked
                            pass
                # take note of whether textbox_idk was clicked, so that next frame we know if clicks are new
                textbox_idk.wasClicked = textbox_idk.isClicked and textbox_idk.status == STARTED
                # Run 'Each Frame' code from textbox_code
                ##Cont. only allowed after answering textbox and pressing continue. 
                if textbox_idk.numClicks > 0 and not dont_remember_clicked:
                    old_textbox.setText('idk')
                    dont_remember_clicked = True
                    
                typed_name = old_textbox.text
                
                if textbox_cont.wasClicked and old_textbox.text != "":
                    continueRoutine = False
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=textbox,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    textbox.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in textbox.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "textbox" ---
            for thisComponent in textbox.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for textbox
            textbox.tStop = globalClock.getTime(format='float')
            textbox.tStopRefresh = tThisFlipGlobal
            thisExp.addData('textbox.stopped', textbox.tStop)
            test_loop.addData('old_textbox.text',old_textbox.text)
            test_loop.addData('textbox_cont.numClicks', textbox_cont.numClicks)
            if textbox_cont.numClicks:
               test_loop.addData('textbox_cont.timesOn', textbox_cont.timesOn)
               test_loop.addData('textbox_cont.timesOff', textbox_cont.timesOff)
            else:
               test_loop.addData('textbox_cont.timesOn', "")
               test_loop.addData('textbox_cont.timesOff', "")
            test_loop.addData('textbox_idk.numClicks', textbox_idk.numClicks)
            if textbox_idk.numClicks:
               test_loop.addData('textbox_idk.timesOn', textbox_idk.timesOn)
               test_loop.addData('textbox_idk.timesOff', textbox_idk.timesOff)
            else:
               test_loop.addData('textbox_idk.timesOn', "")
               test_loop.addData('textbox_idk.timesOff', "")
            # the Routine "textbox" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "resp_confidence2" ---
            # create an object to store info about Routine resp_confidence2
            resp_confidence2 = data.Routine(
                name='resp_confidence2',
                components=[test_image4, conf_question2, confidence2, conf2_cont],
            )
            resp_confidence2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from conf2_code
            if typed_name == "idk" or typed_name == "":
                continueRoutine = False
            test_image4.setImage(img_path)
            confidence2.reset()
            # reset conf2_cont to account for continued clicks & clear times on/off
            conf2_cont.reset()
            # store start times for resp_confidence2
            resp_confidence2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            resp_confidence2.tStart = globalClock.getTime(format='float')
            resp_confidence2.status = STARTED
            thisExp.addData('resp_confidence2.started', resp_confidence2.tStart)
            resp_confidence2.maxDuration = None
            # skip Routine resp_confidence2 if its 'Skip if' condition is True
            resp_confidence2.skipped = continueRoutine and not (on_response != "old")
            continueRoutine = resp_confidence2.skipped
            # keep track of which components have finished
            resp_confidence2Components = resp_confidence2.components
            for thisComponent in resp_confidence2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "resp_confidence2" ---
            resp_confidence2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from conf2_code
                #Cont. only allowed after answering slider and pressing continue. 
                if conf2_cont.wasClicked and confidence2.getRating() is not None:
                    continueRoutine = False
                
                
                # *test_image4* updates
                
                # if test_image4 is starting this frame...
                if test_image4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image4.frameNStart = frameN  # exact frame index
                    test_image4.tStart = t  # local t and not account for scr refresh
                    test_image4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'test_image4.started')
                    # update status
                    test_image4.status = STARTED
                    test_image4.setAutoDraw(True)
                
                # if test_image4 is active this frame...
                if test_image4.status == STARTED:
                    # update params
                    pass
                
                # *conf_question2* updates
                
                # if conf_question2 is starting this frame...
                if conf_question2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    conf_question2.frameNStart = frameN  # exact frame index
                    conf_question2.tStart = t  # local t and not account for scr refresh
                    conf_question2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(conf_question2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'conf_question2.started')
                    # update status
                    conf_question2.status = STARTED
                    conf_question2.setAutoDraw(True)
                
                # if conf_question2 is active this frame...
                if conf_question2.status == STARTED:
                    # update params
                    pass
                
                # *confidence2* updates
                
                # if confidence2 is starting this frame...
                if confidence2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confidence2.frameNStart = frameN  # exact frame index
                    confidence2.tStart = t  # local t and not account for scr refresh
                    confidence2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confidence2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'confidence2.started')
                    # update status
                    confidence2.status = STARTED
                    confidence2.setAutoDraw(True)
                
                # if confidence2 is active this frame...
                if confidence2.status == STARTED:
                    # update params
                    pass
                # *conf2_cont* updates
                
                # if conf2_cont is starting this frame...
                if conf2_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    conf2_cont.frameNStart = frameN  # exact frame index
                    conf2_cont.tStart = t  # local t and not account for scr refresh
                    conf2_cont.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(conf2_cont, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'conf2_cont.started')
                    # update status
                    conf2_cont.status = STARTED
                    win.callOnFlip(conf2_cont.buttonClock.reset)
                    conf2_cont.setAutoDraw(True)
                
                # if conf2_cont is active this frame...
                if conf2_cont.status == STARTED:
                    # update params
                    pass
                    # check whether conf2_cont has been pressed
                    if conf2_cont.isClicked:
                        if not conf2_cont.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            conf2_cont.timesOn.append(conf2_cont.buttonClock.getTime())
                            conf2_cont.timesOff.append(conf2_cont.buttonClock.getTime())
                        elif len(conf2_cont.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            conf2_cont.timesOff[-1] = conf2_cont.buttonClock.getTime()
                        if not conf2_cont.wasClicked:
                            # run callback code when conf2_cont is clicked
                            pass
                # take note of whether conf2_cont was clicked, so that next frame we know if clicks are new
                conf2_cont.wasClicked = conf2_cont.isClicked and conf2_cont.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=resp_confidence2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    resp_confidence2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in resp_confidence2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "resp_confidence2" ---
            for thisComponent in resp_confidence2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for resp_confidence2
            resp_confidence2.tStop = globalClock.getTime(format='float')
            resp_confidence2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('resp_confidence2.stopped', resp_confidence2.tStop)
            test_loop.addData('confidence2.response', confidence2.getRating())
            test_loop.addData('confidence2.rt', confidence2.getRT())
            test_loop.addData('conf2_cont.numClicks', conf2_cont.numClicks)
            if conf2_cont.numClicks:
               test_loop.addData('conf2_cont.timesOn', conf2_cont.timesOn)
               test_loop.addData('conf2_cont.timesOff', conf2_cont.timesOff)
            else:
               test_loop.addData('conf2_cont.timesOn', "")
               test_loop.addData('conf2_cont.timesOff', "")
            # the Routine "resp_confidence2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTest_loop as finished
            if hasattr(thisTest_loop, 'status'):
                thisTest_loop.status = FINISHED
            # if awaiting a pause, pause now
            if test_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                test_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'test_loop'
        test_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisBlock_loop as finished
        if hasattr(thisBlock_loop, 'status'):
            thisBlock_loop.status = FINISHED
        # if awaiting a pause, pause now
        if block_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            block_loop.status = STARTED
    # completed 4.0 repeats of 'block_loop'
    block_loop.status = FINISHED
    
    
    # --- Prepare to start Routine "survey_instrx_1" ---
    # create an object to store info about Routine survey_instrx_1
    survey_instrx_1 = data.Routine(
        name='survey_instrx_1',
        components=[survey_instrx_1_txt, survey_instrx_1_resp, survey_instrx_1_cont],
    )
    survey_instrx_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for survey_instrx_1_resp
    survey_instrx_1_resp.keys = []
    survey_instrx_1_resp.rt = []
    _survey_instrx_1_resp_allKeys = []
    # store start times for survey_instrx_1
    survey_instrx_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    survey_instrx_1.tStart = globalClock.getTime(format='float')
    survey_instrx_1.status = STARTED
    survey_instrx_1.maxDuration = None
    # keep track of which components have finished
    survey_instrx_1Components = survey_instrx_1.components
    for thisComponent in survey_instrx_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "survey_instrx_1" ---
    survey_instrx_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *survey_instrx_1_txt* updates
        
        # if survey_instrx_1_txt is starting this frame...
        if survey_instrx_1_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            survey_instrx_1_txt.frameNStart = frameN  # exact frame index
            survey_instrx_1_txt.tStart = t  # local t and not account for scr refresh
            survey_instrx_1_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_instrx_1_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_instrx_1_txt.started')
            # update status
            survey_instrx_1_txt.status = STARTED
            survey_instrx_1_txt.setAutoDraw(True)
        
        # if survey_instrx_1_txt is active this frame...
        if survey_instrx_1_txt.status == STARTED:
            # update params
            pass
        
        # *survey_instrx_1_resp* updates
        waitOnFlip = False
        
        # if survey_instrx_1_resp is starting this frame...
        if survey_instrx_1_resp.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
            # keep track of start time/frame for later
            survey_instrx_1_resp.frameNStart = frameN  # exact frame index
            survey_instrx_1_resp.tStart = t  # local t and not account for scr refresh
            survey_instrx_1_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_instrx_1_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_instrx_1_resp.started')
            # update status
            survey_instrx_1_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(survey_instrx_1_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(survey_instrx_1_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if survey_instrx_1_resp.status == STARTED and not waitOnFlip:
            theseKeys = survey_instrx_1_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _survey_instrx_1_resp_allKeys.extend(theseKeys)
            if len(_survey_instrx_1_resp_allKeys):
                survey_instrx_1_resp.keys = _survey_instrx_1_resp_allKeys[-1].name  # just the last key pressed
                survey_instrx_1_resp.rt = _survey_instrx_1_resp_allKeys[-1].rt
                survey_instrx_1_resp.duration = _survey_instrx_1_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *survey_instrx_1_cont* updates
        
        # if survey_instrx_1_cont is starting this frame...
        if survey_instrx_1_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
            # keep track of start time/frame for later
            survey_instrx_1_cont.frameNStart = frameN  # exact frame index
            survey_instrx_1_cont.tStart = t  # local t and not account for scr refresh
            survey_instrx_1_cont.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_instrx_1_cont, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_instrx_1_cont.started')
            # update status
            survey_instrx_1_cont.status = STARTED
            survey_instrx_1_cont.setAutoDraw(True)
        
        # if survey_instrx_1_cont is active this frame...
        if survey_instrx_1_cont.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=survey_instrx_1,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            survey_instrx_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in survey_instrx_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "survey_instrx_1" ---
    for thisComponent in survey_instrx_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for survey_instrx_1
    survey_instrx_1.tStop = globalClock.getTime(format='float')
    survey_instrx_1.tStopRefresh = tThisFlipGlobal
    # check responses
    if survey_instrx_1_resp.keys in ['', [], None]:  # No response was made
        survey_instrx_1_resp.keys = None
    thisExp.addData('survey_instrx_1_resp.keys',survey_instrx_1_resp.keys)
    if survey_instrx_1_resp.keys != None:  # we had a response
        thisExp.addData('survey_instrx_1_resp.rt', survey_instrx_1_resp.rt)
        thisExp.addData('survey_instrx_1_resp.duration', survey_instrx_1_resp.duration)
    thisExp.nextEntry()
    # the Routine "survey_instrx_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "survey_instrx_2" ---
    # create an object to store info about Routine survey_instrx_2
    survey_instrx_2 = data.Routine(
        name='survey_instrx_2',
        components=[survey_instrx_2_txt, survey_instrx_2_resp, survey_instrx_2_cont],
    )
    survey_instrx_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for survey_instrx_2_resp
    survey_instrx_2_resp.keys = []
    survey_instrx_2_resp.rt = []
    _survey_instrx_2_resp_allKeys = []
    # store start times for survey_instrx_2
    survey_instrx_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    survey_instrx_2.tStart = globalClock.getTime(format='float')
    survey_instrx_2.status = STARTED
    survey_instrx_2.maxDuration = None
    # keep track of which components have finished
    survey_instrx_2Components = survey_instrx_2.components
    for thisComponent in survey_instrx_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "survey_instrx_2" ---
    survey_instrx_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *survey_instrx_2_txt* updates
        
        # if survey_instrx_2_txt is starting this frame...
        if survey_instrx_2_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            survey_instrx_2_txt.frameNStart = frameN  # exact frame index
            survey_instrx_2_txt.tStart = t  # local t and not account for scr refresh
            survey_instrx_2_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_instrx_2_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_instrx_2_txt.started')
            # update status
            survey_instrx_2_txt.status = STARTED
            survey_instrx_2_txt.setAutoDraw(True)
        
        # if survey_instrx_2_txt is active this frame...
        if survey_instrx_2_txt.status == STARTED:
            # update params
            pass
        
        # *survey_instrx_2_resp* updates
        waitOnFlip = False
        
        # if survey_instrx_2_resp is starting this frame...
        if survey_instrx_2_resp.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
            # keep track of start time/frame for later
            survey_instrx_2_resp.frameNStart = frameN  # exact frame index
            survey_instrx_2_resp.tStart = t  # local t and not account for scr refresh
            survey_instrx_2_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_instrx_2_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_instrx_2_resp.started')
            # update status
            survey_instrx_2_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(survey_instrx_2_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(survey_instrx_2_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if survey_instrx_2_resp.status == STARTED and not waitOnFlip:
            theseKeys = survey_instrx_2_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _survey_instrx_2_resp_allKeys.extend(theseKeys)
            if len(_survey_instrx_2_resp_allKeys):
                survey_instrx_2_resp.keys = _survey_instrx_2_resp_allKeys[-1].name  # just the last key pressed
                survey_instrx_2_resp.rt = _survey_instrx_2_resp_allKeys[-1].rt
                survey_instrx_2_resp.duration = _survey_instrx_2_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *survey_instrx_2_cont* updates
        
        # if survey_instrx_2_cont is starting this frame...
        if survey_instrx_2_cont.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
            # keep track of start time/frame for later
            survey_instrx_2_cont.frameNStart = frameN  # exact frame index
            survey_instrx_2_cont.tStart = t  # local t and not account for scr refresh
            survey_instrx_2_cont.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_instrx_2_cont, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_instrx_2_cont.started')
            # update status
            survey_instrx_2_cont.status = STARTED
            survey_instrx_2_cont.setAutoDraw(True)
        
        # if survey_instrx_2_cont is active this frame...
        if survey_instrx_2_cont.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=survey_instrx_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            survey_instrx_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in survey_instrx_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "survey_instrx_2" ---
    for thisComponent in survey_instrx_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for survey_instrx_2
    survey_instrx_2.tStop = globalClock.getTime(format='float')
    survey_instrx_2.tStopRefresh = tThisFlipGlobal
    # check responses
    if survey_instrx_2_resp.keys in ['', [], None]:  # No response was made
        survey_instrx_2_resp.keys = None
    thisExp.addData('survey_instrx_2_resp.keys',survey_instrx_2_resp.keys)
    if survey_instrx_2_resp.keys != None:  # we had a response
        thisExp.addData('survey_instrx_2_resp.rt', survey_instrx_2_resp.rt)
        thisExp.addData('survey_instrx_2_resp.duration', survey_instrx_2_resp.duration)
    thisExp.nextEntry()
    # the Routine "survey_instrx_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "name_prac" ---
    # create an object to store info about Routine name_prac
    name_prac = data.Routine(
        name='name_prac',
        components=[survey_prac_isi, name_prac_text, survey_prac_RT, survey_prac_moveon],
    )
    name_prac.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from survey_prac_code
    #set random time for isi
    random_isi = random.uniform(0.5, 1.5)
    
    #set blank for participant input
    survey_known = None
    # create starting attributes for survey_prac_RT
    survey_prac_RT.keys = []
    survey_prac_RT.rt = []
    _survey_prac_RT_allKeys = []
    # store start times for name_prac
    name_prac.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    name_prac.tStart = globalClock.getTime(format='float')
    name_prac.status = STARTED
    name_prac.maxDuration = None
    # keep track of which components have finished
    name_pracComponents = name_prac.components
    for thisComponent in name_prac.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "name_prac" ---
    name_prac.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *survey_prac_isi* updates
        
        # if survey_prac_isi is starting this frame...
        if survey_prac_isi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            survey_prac_isi.frameNStart = frameN  # exact frame index
            survey_prac_isi.tStart = t  # local t and not account for scr refresh
            survey_prac_isi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_prac_isi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_prac_isi.started')
            # update status
            survey_prac_isi.status = STARTED
            survey_prac_isi.setAutoDraw(True)
        
        # if survey_prac_isi is active this frame...
        if survey_prac_isi.status == STARTED:
            # update params
            pass
        
        # if survey_prac_isi is stopping this frame...
        if survey_prac_isi.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > survey_prac_isi.tStartRefresh + random_isi-frameTolerance:
                # keep track of stop time/frame for later
                survey_prac_isi.tStop = t  # not accounting for scr refresh
                survey_prac_isi.tStopRefresh = tThisFlipGlobal  # on global time
                survey_prac_isi.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'survey_prac_isi.stopped')
                # update status
                survey_prac_isi.status = FINISHED
                survey_prac_isi.setAutoDraw(False)
        
        # *name_prac_text* updates
        
        # if name_prac_text is starting this frame...
        if name_prac_text.status == NOT_STARTED and survey_prac_isi.status == FINISHED:
            # keep track of start time/frame for later
            name_prac_text.frameNStart = frameN  # exact frame index
            name_prac_text.tStart = t  # local t and not account for scr refresh
            name_prac_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(name_prac_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'name_prac_text.started')
            # update status
            name_prac_text.status = STARTED
            name_prac_text.setAutoDraw(True)
        
        # if name_prac_text is active this frame...
        if name_prac_text.status == STARTED:
            # update params
            pass
        
        # *survey_prac_RT* updates
        waitOnFlip = False
        
        # if survey_prac_RT is starting this frame...
        if survey_prac_RT.status == NOT_STARTED and survey_prac_isi.status == FINISHED:
            # keep track of start time/frame for later
            survey_prac_RT.frameNStart = frameN  # exact frame index
            survey_prac_RT.tStart = t  # local t and not account for scr refresh
            survey_prac_RT.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_prac_RT, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_prac_RT.started')
            # update status
            survey_prac_RT.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(survey_prac_RT.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(survey_prac_RT.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if survey_prac_RT.status == STARTED and not waitOnFlip:
            theseKeys = survey_prac_RT.getKeys(keyList=['space','lctrl','rctrl'], ignoreKeys=["escape"], waitRelease=False)
            _survey_prac_RT_allKeys.extend(theseKeys)
            if len(_survey_prac_RT_allKeys):
                survey_prac_RT.keys = _survey_prac_RT_allKeys[-1].name  # just the last key pressed
                survey_prac_RT.rt = _survey_prac_RT_allKeys[-1].rt
                survey_prac_RT.duration = _survey_prac_RT_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *survey_prac_moveon* updates
        
        # if survey_prac_moveon is starting this frame...
        if survey_prac_moveon.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            survey_prac_moveon.frameNStart = frameN  # exact frame index
            survey_prac_moveon.tStart = t  # local t and not account for scr refresh
            survey_prac_moveon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_prac_moveon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_prac_moveon.started')
            # update status
            survey_prac_moveon.status = STARTED
            survey_prac_moveon.setAutoDraw(True)
        
        # if survey_prac_moveon is active this frame...
        if survey_prac_moveon.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=name_prac,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            name_prac.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in name_prac.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "name_prac" ---
    for thisComponent in name_prac.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for name_prac
    name_prac.tStop = globalClock.getTime(format='float')
    name_prac.tStopRefresh = tThisFlipGlobal
    # Run 'End Routine' code from survey_prac_code
    #depending on participant answer, log resp for the name shown 
    if survey_prac_RT.keys == 'space':
        survey_known = True
    elif survey_prac_RT.keys == ['lctrl','rctrl']:
        survey_known = False
    else:
        survey_known = False
    # check responses
    if survey_prac_RT.keys in ['', [], None]:  # No response was made
        survey_prac_RT.keys = None
    thisExp.addData('survey_prac_RT.keys',survey_prac_RT.keys)
    if survey_prac_RT.keys != None:  # we had a response
        thisExp.addData('survey_prac_RT.rt', survey_prac_RT.rt)
        thisExp.addData('survey_prac_RT.duration', survey_prac_RT.duration)
    thisExp.nextEntry()
    # the Routine "name_prac" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "survey_prac" ---
    # create an object to store info about Routine survey_prac
    survey_prac = data.Routine(
        name='survey_prac',
        components=[name_survey_prac, HowMany_prac_slider, HowEasy_prac_slider, HowEasy_prac_text, HowMany_prac_text, survey_prac_cont],
    )
    survey_prac.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from survey_prac_code_2
    #depending on response, skip if survey_know = False, continue if survey_known is True
    continueRoutine = survey_known
    win.mouseVisible = True
    name_survey_prac.setText('Barack\n')
    HowMany_prac_slider.reset()
    HowEasy_prac_slider.reset()
    # reset survey_prac_cont to account for continued clicks & clear times on/off
    survey_prac_cont.reset()
    # store start times for survey_prac
    survey_prac.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    survey_prac.tStart = globalClock.getTime(format='float')
    survey_prac.status = STARTED
    thisExp.addData('survey_prac.started', survey_prac.tStart)
    survey_prac.maxDuration = None
    # keep track of which components have finished
    survey_pracComponents = survey_prac.components
    for thisComponent in survey_prac.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "survey_prac" ---
    survey_prac.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from survey_prac_code_2
        #continue button once both sliders are answered
        if HowMany_prac_slider.getRating() is not None and HowEasy_prac_slider.getRating() is not None:
            survey_prac_cont.opacity = 1
            allowClick = True
        else:
            survey_prac_cont.opacity = .3
            allowClick = False
        
        #Cont. if participant clicks the cont. button.
        if mouse.isPressedIn(survey_prac_cont) and allowClick:
            continueRoutine = False
        
        
        
        # *name_survey_prac* updates
        
        # if name_survey_prac is starting this frame...
        if name_survey_prac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            name_survey_prac.frameNStart = frameN  # exact frame index
            name_survey_prac.tStart = t  # local t and not account for scr refresh
            name_survey_prac.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(name_survey_prac, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'name_survey_prac.started')
            # update status
            name_survey_prac.status = STARTED
            name_survey_prac.setAutoDraw(True)
        
        # if name_survey_prac is active this frame...
        if name_survey_prac.status == STARTED:
            # update params
            pass
        
        # *HowMany_prac_slider* updates
        
        # if HowMany_prac_slider is starting this frame...
        if HowMany_prac_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            HowMany_prac_slider.frameNStart = frameN  # exact frame index
            HowMany_prac_slider.tStart = t  # local t and not account for scr refresh
            HowMany_prac_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(HowMany_prac_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'HowMany_prac_slider.started')
            # update status
            HowMany_prac_slider.status = STARTED
            HowMany_prac_slider.setAutoDraw(True)
        
        # if HowMany_prac_slider is active this frame...
        if HowMany_prac_slider.status == STARTED:
            # update params
            pass
        
        # *HowEasy_prac_slider* updates
        
        # if HowEasy_prac_slider is starting this frame...
        if HowEasy_prac_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            HowEasy_prac_slider.frameNStart = frameN  # exact frame index
            HowEasy_prac_slider.tStart = t  # local t and not account for scr refresh
            HowEasy_prac_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(HowEasy_prac_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'HowEasy_prac_slider.started')
            # update status
            HowEasy_prac_slider.status = STARTED
            HowEasy_prac_slider.setAutoDraw(True)
        
        # if HowEasy_prac_slider is active this frame...
        if HowEasy_prac_slider.status == STARTED:
            # update params
            pass
        
        # *HowEasy_prac_text* updates
        
        # if HowEasy_prac_text is starting this frame...
        if HowEasy_prac_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            HowEasy_prac_text.frameNStart = frameN  # exact frame index
            HowEasy_prac_text.tStart = t  # local t and not account for scr refresh
            HowEasy_prac_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(HowEasy_prac_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'HowEasy_prac_text.started')
            # update status
            HowEasy_prac_text.status = STARTED
            HowEasy_prac_text.setAutoDraw(True)
        
        # if HowEasy_prac_text is active this frame...
        if HowEasy_prac_text.status == STARTED:
            # update params
            pass
        
        # *HowMany_prac_text* updates
        
        # if HowMany_prac_text is starting this frame...
        if HowMany_prac_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            HowMany_prac_text.frameNStart = frameN  # exact frame index
            HowMany_prac_text.tStart = t  # local t and not account for scr refresh
            HowMany_prac_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(HowMany_prac_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'HowMany_prac_text.started')
            # update status
            HowMany_prac_text.status = STARTED
            HowMany_prac_text.setAutoDraw(True)
        
        # if HowMany_prac_text is active this frame...
        if HowMany_prac_text.status == STARTED:
            # update params
            pass
        # *survey_prac_cont* updates
        
        # if survey_prac_cont is starting this frame...
        if survey_prac_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            survey_prac_cont.frameNStart = frameN  # exact frame index
            survey_prac_cont.tStart = t  # local t and not account for scr refresh
            survey_prac_cont.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_prac_cont, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_prac_cont.started')
            # update status
            survey_prac_cont.status = STARTED
            win.callOnFlip(survey_prac_cont.buttonClock.reset)
            survey_prac_cont.setAutoDraw(True)
        
        # if survey_prac_cont is active this frame...
        if survey_prac_cont.status == STARTED:
            # update params
            pass
            # check whether survey_prac_cont has been pressed
            if survey_prac_cont.isClicked:
                if not survey_prac_cont.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    survey_prac_cont.timesOn.append(survey_prac_cont.buttonClock.getTime())
                    survey_prac_cont.timesOff.append(survey_prac_cont.buttonClock.getTime())
                elif len(survey_prac_cont.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    survey_prac_cont.timesOff[-1] = survey_prac_cont.buttonClock.getTime()
                if not survey_prac_cont.wasClicked:
                    # run callback code when survey_prac_cont is clicked
                    pass
        # take note of whether survey_prac_cont was clicked, so that next frame we know if clicks are new
        survey_prac_cont.wasClicked = survey_prac_cont.isClicked and survey_prac_cont.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=survey_prac,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            survey_prac.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in survey_prac.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "survey_prac" ---
    for thisComponent in survey_prac.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for survey_prac
    survey_prac.tStop = globalClock.getTime(format='float')
    survey_prac.tStopRefresh = tThisFlipGlobal
    thisExp.addData('survey_prac.stopped', survey_prac.tStop)
    thisExp.addData('HowMany_prac_slider.response', HowMany_prac_slider.getRating())
    thisExp.addData('HowMany_prac_slider.rt', HowMany_prac_slider.getRT())
    thisExp.addData('HowEasy_prac_slider.response', HowEasy_prac_slider.getRating())
    thisExp.addData('HowEasy_prac_slider.rt', HowEasy_prac_slider.getRT())
    thisExp.addData('survey_prac_cont.numClicks', survey_prac_cont.numClicks)
    if survey_prac_cont.numClicks:
       thisExp.addData('survey_prac_cont.timesOn', survey_prac_cont.timesOn)
       thisExp.addData('survey_prac_cont.timesOff', survey_prac_cont.timesOff)
    else:
       thisExp.addData('survey_prac_cont.timesOn', "")
       thisExp.addData('survey_prac_cont.timesOff', "")
    thisExp.nextEntry()
    # the Routine "survey_prac" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    survey_loop = data.TrialHandler2(
        name='survey_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(names_file), 
        seed=None, 
    )
    thisExp.addLoop(survey_loop)  # add the loop to the experiment
    thisSurvey_loop = survey_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSurvey_loop.rgb)
    if thisSurvey_loop != None:
        for paramName in thisSurvey_loop:
            globals()[paramName] = thisSurvey_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSurvey_loop in survey_loop:
        survey_loop.status = STARTED
        if hasattr(thisSurvey_loop, 'status'):
            thisSurvey_loop.status = STARTED
        currentLoop = survey_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSurvey_loop.rgb)
        if thisSurvey_loop != None:
            for paramName in thisSurvey_loop:
                globals()[paramName] = thisSurvey_loop[paramName]
        
        # --- Prepare to start Routine "name_display" ---
        # create an object to store info about Routine name_display
        name_display = data.Routine(
            name='name_display',
            components=[survey_isi, name_count, name_text, survey_RT, survey_move_on],
        )
        name_display.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from survey_code
        #set random time for isi
        random_isi = random.uniform(0.5, 1.5)
        
        #show the count
        current_count = f"{survey_loop.thisN + 1}/64"
        
        #set blank for participant input
        survey_known = None
        name_count.setText(current_count)
        name_text.setText(name
        )
        # create starting attributes for survey_RT
        survey_RT.keys = []
        survey_RT.rt = []
        _survey_RT_allKeys = []
        # store start times for name_display
        name_display.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        name_display.tStart = globalClock.getTime(format='float')
        name_display.status = STARTED
        thisExp.addData('name_display.started', name_display.tStart)
        name_display.maxDuration = None
        # keep track of which components have finished
        name_displayComponents = name_display.components
        for thisComponent in name_display.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "name_display" ---
        name_display.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisSurvey_loop, 'status') and thisSurvey_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *survey_isi* updates
            
            # if survey_isi is starting this frame...
            if survey_isi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                survey_isi.frameNStart = frameN  # exact frame index
                survey_isi.tStart = t  # local t and not account for scr refresh
                survey_isi.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(survey_isi, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'survey_isi.started')
                # update status
                survey_isi.status = STARTED
                survey_isi.setAutoDraw(True)
            
            # if survey_isi is active this frame...
            if survey_isi.status == STARTED:
                # update params
                pass
            
            # if survey_isi is stopping this frame...
            if survey_isi.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > survey_isi.tStartRefresh + random_isi-frameTolerance:
                    # keep track of stop time/frame for later
                    survey_isi.tStop = t  # not accounting for scr refresh
                    survey_isi.tStopRefresh = tThisFlipGlobal  # on global time
                    survey_isi.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'survey_isi.stopped')
                    # update status
                    survey_isi.status = FINISHED
                    survey_isi.setAutoDraw(False)
            
            # *name_count* updates
            
            # if name_count is starting this frame...
            if name_count.status == NOT_STARTED and survey_isi.status == FINISHED:
                # keep track of start time/frame for later
                name_count.frameNStart = frameN  # exact frame index
                name_count.tStart = t  # local t and not account for scr refresh
                name_count.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(name_count, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'name_count.started')
                # update status
                name_count.status = STARTED
                name_count.setAutoDraw(True)
            
            # if name_count is active this frame...
            if name_count.status == STARTED:
                # update params
                pass
            
            # *name_text* updates
            
            # if name_text is starting this frame...
            if name_text.status == NOT_STARTED and survey_isi.status == FINISHED:
                # keep track of start time/frame for later
                name_text.frameNStart = frameN  # exact frame index
                name_text.tStart = t  # local t and not account for scr refresh
                name_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(name_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'name_text.started')
                # update status
                name_text.status = STARTED
                name_text.setAutoDraw(True)
            
            # if name_text is active this frame...
            if name_text.status == STARTED:
                # update params
                pass
            
            # *survey_RT* updates
            waitOnFlip = False
            
            # if survey_RT is starting this frame...
            if survey_RT.status == NOT_STARTED and survey_isi.status == FINISHED:
                # keep track of start time/frame for later
                survey_RT.frameNStart = frameN  # exact frame index
                survey_RT.tStart = t  # local t and not account for scr refresh
                survey_RT.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(survey_RT, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'survey_RT.started')
                # update status
                survey_RT.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(survey_RT.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(survey_RT.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if survey_RT.status == STARTED and not waitOnFlip:
                theseKeys = survey_RT.getKeys(keyList=['space','lctrl','rctrl'], ignoreKeys=["escape"], waitRelease=False)
                _survey_RT_allKeys.extend(theseKeys)
                if len(_survey_RT_allKeys):
                    survey_RT.keys = _survey_RT_allKeys[-1].name  # just the last key pressed
                    survey_RT.rt = _survey_RT_allKeys[-1].rt
                    survey_RT.duration = _survey_RT_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *survey_move_on* updates
            
            # if survey_move_on is starting this frame...
            if survey_move_on.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                # keep track of start time/frame for later
                survey_move_on.frameNStart = frameN  # exact frame index
                survey_move_on.tStart = t  # local t and not account for scr refresh
                survey_move_on.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(survey_move_on, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'survey_move_on.started')
                # update status
                survey_move_on.status = STARTED
                survey_move_on.setAutoDraw(True)
            
            # if survey_move_on is active this frame...
            if survey_move_on.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=name_display,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                name_display.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in name_display.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "name_display" ---
        for thisComponent in name_display.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for name_display
        name_display.tStop = globalClock.getTime(format='float')
        name_display.tStopRefresh = tThisFlipGlobal
        thisExp.addData('name_display.stopped', name_display.tStop)
        # Run 'End Routine' code from survey_code
        #depending on participant answer, log resp for the name shown 
        if survey_RT.keys == 'space':
            survey_known = True
        elif survey_RT.keys == ['lctrl','rctrl']:
            survey_known = False
        else:
            survey_known = False
        # check responses
        if survey_RT.keys in ['', [], None]:  # No response was made
            survey_RT.keys = None
        survey_loop.addData('survey_RT.keys',survey_RT.keys)
        if survey_RT.keys != None:  # we had a response
            survey_loop.addData('survey_RT.rt', survey_RT.rt)
            survey_loop.addData('survey_RT.duration', survey_RT.duration)
        # the Routine "name_display" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "survey" ---
        # create an object to store info about Routine survey
        survey = data.Routine(
            name='survey',
            components=[name_text2, name_count2, HowMany_slider, HowEasy_slider, HowEasy_test, HowMany_text, survey_cont],
        )
        survey.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from survey_code2
        #depending on response, skip if survey_know = False, continue if survey_known is True
        continueRoutine = survey_known
        win.mouseVisible = True
        name_text2.setText(name
        )
        name_count2.setText(current_count)
        HowMany_slider.reset()
        HowEasy_slider.reset()
        # reset survey_cont to account for continued clicks & clear times on/off
        survey_cont.reset()
        # store start times for survey
        survey.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        survey.tStart = globalClock.getTime(format='float')
        survey.status = STARTED
        thisExp.addData('survey.started', survey.tStart)
        survey.maxDuration = None
        # keep track of which components have finished
        surveyComponents = survey.components
        for thisComponent in survey.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "survey" ---
        survey.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisSurvey_loop, 'status') and thisSurvey_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from survey_code2
            #continue button once both sliders are answered
            if HowMany_slider.getRating() is not None and HowEasy_slider.getRating() is not None:
                survey_cont.opacity = 1
                allowClick = True
            else:
                survey_cont.opacity = .3
                allowClick = False
            
            #Cont. if participant clicks the cont. button.
            if mouse.isPressedIn(survey_cont) and allowClick:
                continueRoutine = False
            
            
            
            # *name_text2* updates
            
            # if name_text2 is starting this frame...
            if name_text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                name_text2.frameNStart = frameN  # exact frame index
                name_text2.tStart = t  # local t and not account for scr refresh
                name_text2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(name_text2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'name_text2.started')
                # update status
                name_text2.status = STARTED
                name_text2.setAutoDraw(True)
            
            # if name_text2 is active this frame...
            if name_text2.status == STARTED:
                # update params
                pass
            
            # *name_count2* updates
            
            # if name_count2 is starting this frame...
            if name_count2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                name_count2.frameNStart = frameN  # exact frame index
                name_count2.tStart = t  # local t and not account for scr refresh
                name_count2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(name_count2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'name_count2.started')
                # update status
                name_count2.status = STARTED
                name_count2.setAutoDraw(True)
            
            # if name_count2 is active this frame...
            if name_count2.status == STARTED:
                # update params
                pass
            
            # *HowMany_slider* updates
            
            # if HowMany_slider is starting this frame...
            if HowMany_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                HowMany_slider.frameNStart = frameN  # exact frame index
                HowMany_slider.tStart = t  # local t and not account for scr refresh
                HowMany_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(HowMany_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'HowMany_slider.started')
                # update status
                HowMany_slider.status = STARTED
                HowMany_slider.setAutoDraw(True)
            
            # if HowMany_slider is active this frame...
            if HowMany_slider.status == STARTED:
                # update params
                pass
            
            # *HowEasy_slider* updates
            
            # if HowEasy_slider is starting this frame...
            if HowEasy_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                HowEasy_slider.frameNStart = frameN  # exact frame index
                HowEasy_slider.tStart = t  # local t and not account for scr refresh
                HowEasy_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(HowEasy_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'HowEasy_slider.started')
                # update status
                HowEasy_slider.status = STARTED
                HowEasy_slider.setAutoDraw(True)
            
            # if HowEasy_slider is active this frame...
            if HowEasy_slider.status == STARTED:
                # update params
                pass
            
            # *HowEasy_test* updates
            
            # if HowEasy_test is starting this frame...
            if HowEasy_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                HowEasy_test.frameNStart = frameN  # exact frame index
                HowEasy_test.tStart = t  # local t and not account for scr refresh
                HowEasy_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(HowEasy_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'HowEasy_test.started')
                # update status
                HowEasy_test.status = STARTED
                HowEasy_test.setAutoDraw(True)
            
            # if HowEasy_test is active this frame...
            if HowEasy_test.status == STARTED:
                # update params
                pass
            
            # *HowMany_text* updates
            
            # if HowMany_text is starting this frame...
            if HowMany_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                HowMany_text.frameNStart = frameN  # exact frame index
                HowMany_text.tStart = t  # local t and not account for scr refresh
                HowMany_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(HowMany_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'HowMany_text.started')
                # update status
                HowMany_text.status = STARTED
                HowMany_text.setAutoDraw(True)
            
            # if HowMany_text is active this frame...
            if HowMany_text.status == STARTED:
                # update params
                pass
            # *survey_cont* updates
            
            # if survey_cont is starting this frame...
            if survey_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                survey_cont.frameNStart = frameN  # exact frame index
                survey_cont.tStart = t  # local t and not account for scr refresh
                survey_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(survey_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'survey_cont.started')
                # update status
                survey_cont.status = STARTED
                win.callOnFlip(survey_cont.buttonClock.reset)
                survey_cont.setAutoDraw(True)
            
            # if survey_cont is active this frame...
            if survey_cont.status == STARTED:
                # update params
                pass
                # check whether survey_cont has been pressed
                if survey_cont.isClicked:
                    if not survey_cont.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        survey_cont.timesOn.append(survey_cont.buttonClock.getTime())
                        survey_cont.timesOff.append(survey_cont.buttonClock.getTime())
                    elif len(survey_cont.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        survey_cont.timesOff[-1] = survey_cont.buttonClock.getTime()
                    if not survey_cont.wasClicked:
                        # run callback code when survey_cont is clicked
                        pass
            # take note of whether survey_cont was clicked, so that next frame we know if clicks are new
            survey_cont.wasClicked = survey_cont.isClicked and survey_cont.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=survey,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                survey.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in survey.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "survey" ---
        for thisComponent in survey.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for survey
        survey.tStop = globalClock.getTime(format='float')
        survey.tStopRefresh = tThisFlipGlobal
        thisExp.addData('survey.stopped', survey.tStop)
        survey_loop.addData('HowMany_slider.response', HowMany_slider.getRating())
        survey_loop.addData('HowMany_slider.rt', HowMany_slider.getRT())
        survey_loop.addData('HowEasy_slider.response', HowEasy_slider.getRating())
        survey_loop.addData('HowEasy_slider.rt', HowEasy_slider.getRT())
        survey_loop.addData('survey_cont.numClicks', survey_cont.numClicks)
        if survey_cont.numClicks:
           survey_loop.addData('survey_cont.timesOn', survey_cont.timesOn)
           survey_loop.addData('survey_cont.timesOff', survey_cont.timesOff)
        else:
           survey_loop.addData('survey_cont.timesOn', "")
           survey_loop.addData('survey_cont.timesOff', "")
        # the Routine "survey" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisSurvey_loop as finished
        if hasattr(thisSurvey_loop, 'status'):
            thisSurvey_loop.status = FINISHED
        # if awaiting a pause, pause now
        if survey_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            survey_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'survey_loop'
    survey_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[end_text, end_resp],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_resp
    end_resp.keys = []
    end_resp.rt = []
    _end_resp_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # *end_resp* updates
        waitOnFlip = False
        
        # if end_resp is starting this frame...
        if end_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_resp.frameNStart = frameN  # exact frame index
            end_resp.tStart = t  # local t and not account for scr refresh
            end_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_resp.started')
            # update status
            end_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_resp.status == STARTED and not waitOnFlip:
            theseKeys = end_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_resp_allKeys.extend(theseKeys)
            if len(_end_resp_allKeys):
                end_resp.keys = _end_resp_allKeys[-1].name  # just the last key pressed
                end_resp.rt = _end_resp_allKeys[-1].rt
                end_resp.duration = _end_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    # check responses
    if end_resp.keys in ['', [], None]:  # No response was made
        end_resp.keys = None
    thisExp.addData('end_resp.keys',end_resp.keys)
    if end_resp.keys != None:  # we had a response
        thisExp.addData('end_resp.rt', end_resp.rt)
        thisExp.addData('end_resp.duration', end_resp.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
