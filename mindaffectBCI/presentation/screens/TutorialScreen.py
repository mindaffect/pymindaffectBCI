from mindaffectBCI.presentation.screens.SubscreenMenuScreen import SubscreenMenuScreen

subscreens = {
    'conn':('ConnectingScreen',{}),
    'qual':('ElectrodeQualityScreen',{}),
    'stim':('SelectionGridScreen',
                {'symbols':[['+']],
                 'noisetag_mode':'prediction',
                 'noisetag_args':{
                    'stimSeq':'level2_gold_soa5.txt',
                    'framesperbit':6,
                    'nTrials':40,
                    'duration':15,
                    'waitduration':4
                    }
                }
            )
}
# start with connecting screen & transition to menu when connection is done
start_screen = 'conn'
subscreen_transitions = { 'conn':'menu' }

# create the noisetag object and load a stimulus sequence to play
from mindaffectBCI.noisetag import Noisetag
# the one we just made
nt = Noisetag(stimSeq='level2_gold_soa5.txt')


from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen, run
# make the window and run it
window = initPyglet(width=640, height=480)

# connect the screen to the window and noise-tag
screen = SubscreenMenuScreen(window=window, noisetag=nt, 
                             subscreens=subscreens,
                             subscreen_transitions=subscreen_transitions, start_screen=start_screen)

# run the screen
run_screen(window, screen)


# Alternative way of doing the same with the configuration given in a .json compatiable python dict()
from mindaffectBCI.presentation.ScreenRunner import run
config={
    # set to run from the ScreenRunner
    "presentation": "mindaffectBCI.presentation.ScreenRunner",
    # setup the screen to run..
    "presentation_args": {
        "width": 1024,
        "height": 768,
        # set the noisetag object to use, and the stimsequence to play
        "noisetag": {
            "stimSeq": "level2_gold_soa5.txt"
        },
        # set main screen to show.  Here we use the sub-screen menu screen to build a menu of sub-screens to show
        "screen": [
            "mindaffectBCI.presentation.screens.SubscreenMenuScreen.SubscreenMenuScreen",
            {
                "title": "Minimal Application Example",
                # here is the list of sub-screens to select from in the main menu
                # the format is : "name":[ScreenClassName, screen_args]
                # where screenclass name is either a fullyqualified class name, or a class in 'mindaffectBCI.presentation.screens' with the same filename and class name
                # and screen_args is a dictionary of arguments to pass to the screen constructor
                "subscreens": {
                    "conn": ["ConnectingScreen", {} ],
                    "elect": [ "ElectrodeQualityScreen", { "label": "Electrode Quality Screen" } ],
                    "stim": ["SelectionGridScreen", {
                            # single block in the middle of the screen with a + as fixation
                            "symbols": [ [ "+" ] ],
                            # use the correct color mapping
                            "state2color": { "0":[5,5,5], "1":[255,255,255] },
                            # setup to run thie stimulus sequence in prediction mode, i.e. no 'target' or cueing
                            "noisetag_mode": "prediction",
                            # setup the playback arguments -- slowdown, num-trials, trial-duration etc.
                            "noisetag_args": {
                                "stimSeq": "level2_gold_soa5.txt", # 5x slowdown = 2 events / 10 frames
                                "framesperbit": 6, # 6x slowdown = 100ms stimulus rate = 10 frames / sec
                                "nTrials": 40,
                                "duration": 15, # 15s trial duration
                                "waitduration": 4
                            }
                        }
                    ]
                },
                # start with connecting screen & transition to menu when connection is done
                "start_screen": "conn",
                "subscreen_transitions": { "conn": "menu" }
            }
        ]
    }
}
run(config_file=config)