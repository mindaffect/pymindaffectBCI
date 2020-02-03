# mindaffectBCI
This repository contains the python SDK code for the Brain Computer Interface (BCI) developed by the company [Mindaffect](https://mindaffect.nl).

## File Structure
This repository is organized roughly as follows:
  * `mindaffectBCI` - contains the python package containing the mindaffectBCI SDK.  Important modules within this package are:
     * noisetag.py - This module contains the main API for developing User Interfaces with BCI control
     * utopiaController.py - This module contains the application level APIs for interacting with the MindAffect Decoder.
     * utopiaclient.py - This module contains the low-level networking functions for communicating with the MindAffect Decoder - which is normally a separate computer running the eeg analysis software.
     * stimseq.py -- This module contains the low-level functions for loading and codebooks - which define how the presented stimuli will look.
  * `codebooks` - Contains the most common noisetagging codebooks as text files
  * `examples` - contains python based examples for Presentation and Output parts of the BCI

# Quick-start

##Installing mindaffectBCI

That's easy:
`pip install mindaffectBCI`

# Simple *output* module
An output module listens for selections from the mindaffect decoder and acts on them to create some output.  Here we show how to make a simple output module which print's "Hello World" when the presentation 'button' with ID=1 is selected.

```
# Import the utopia2output module
from minaffectBCI.utopia2output import Utopia2Output
```

Now we can create an utopia2output object and connect it to a running mindaffect BCI decoder. 

```
u2o=Utopia2Output()
u2o.connect()
```

(Note: For this to succeed you must have a real or simulated mindaffectBCI decoder running somewhere on your network.)

Now we define a function to print hello-world:
```
def helloworld():
   print("hello world")
```

And connect it so it is run when the object with ID=1 is selected.

```
# set the objectID2Action dictionary to use our helloworld function if 1 is selected 
u2o.objectID2Action={ 1:helloworld }
```

Finally, run the main loop

```
u2o.run()
```

# Simple *presention* module



