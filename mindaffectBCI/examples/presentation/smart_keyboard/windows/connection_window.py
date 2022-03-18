"""
This modules contains a single class displaying the status of connection to UtopiaHub.

It contains the following visual and functional elements:
 * textfield displaying the current status of connection
"""

#  Copyright (c) 2021,
#  Authors: Thomas de Lange, Thomas Jurriaans, Damy Hillen, Joost Vossers, Jort Gutter, Florian Handke, Stijn Boosman
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from mindaffectBCI.examples.presentation.smart_keyboard.windows.window import Window
import threading


class ConnectionWindow(Window):
    """
    Subclass of Window, designed for trying to connect the Noisetag with the Utopia Hub, and in case of a
    failed connection, it will start the application with flickering disabled.

    Args:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style (dict): Style instructions for the keyboard.
        use_flickering (bool): Activates or deactivates flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

    Attributes:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style (dict): Style instructions for the keyboard.
        use_flickering (bool): Activates or deactivates flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

        connection_status (bool): Indicates the status of the connection to the Utopia Hub.
        instructions (dict): Mapping of phase names to instruction texts.
        instruction (Object): Framework dependent text object which holds the instruction.
        current_phase (str): Name of the current phase.
        phase_links (dict): Mapping from phases to their next phase.
        timer (int): Count down used to time the duration of the phases.
        thread (Thread): A python thread for creating a connection to the Hub without freezing the application.
    """

    def __init__(self, parent, facade, style, use_flickering, noisetag=None, host=None):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        self.connection_status = False
        self.host = host

        self.instructions = {
            "waiting": 'Attempting to connect to the Utopia Hub\n\nPlease wait',
            "success": 'Connection established\n\nStarting application...',
            "fail": 'Connection failed\n\n Check if the Utopia Hub is running\n\n'
                    'Starting application without flickering...'
        }

        self.instruction = self.facade.create_text(
            text=self.instructions["waiting"],
            col=style["text_color"],
            # place in the middle of the screen:
            pos=(.5, .5),
        )

        self.logo = self.facade.create_icon(
            "key_icons\MindAffect_Logo.png",
            label_col=self.style.get("MA_orange",(128,128,0)),
            size=(.1,.1),
            pos=(.95,.95),
        )
        
        self.current_phase = "waiting"

        self.phase_links = {
            "waiting": "fail",
            "fail": "end",
            "success": "end"
        }

        self.timer = 0
        self.thread = None

    def activate(self):
        """Activates the visual and functional elements of this Window."""
        self.facade.toggle_text_render(self.instruction, True)
        self.facade.toggle_image_render(self.logo, True)
        self.timer = 600
        if self.use_flickering:
            self.attempt_connection()

    def deactivate(self):
        """Deactivates the visual and functional elements of this Window."""
        self.facade.toggle_text_render(self.instruction, False)
        self.facade.toggle_image_render(self.logo, False)

    def draw(self, noisetag, last_flip_time, target_idx=-1):
        """
        Decides per frame if a phase change should occur.

        Args:
            noisetag (Noisetag): Reference to the noisetag object
            last_flip_time (int): time of the last flip
            target_idx (int): (Optional) index of the target stimulus.
        """
        if self.use_flickering:
            self.connection_status = noisetag.isConnected()

            if self.connection_status and self.current_phase == 'waiting':
                self.next_phase(new_phase="success")
                return
        else:
            # If flickering was set to false, immediately start the application without attempting a connection:
            self.parent.build_windows(self, False)

        self.timer -= 1

        if self.timer == 0:
            self.next_phase()

    def next_phase(self, new_phase=None):
        """
        Handles the phase changing of the ConnectionWindow. If a phase is specified,will directy switch
        to that phase. Otherwise will look up the next phase in self.phase_links.

        Args:
            new_phase (str): Optional name of the next phase
        """
        if not new_phase:
            new_phase = self.phase_links[self.current_phase]
        self.current_phase = new_phase

        if self.current_phase == "end":
            # Instruct the Application to build all the other windows:
            self.parent.build_windows(self, self.connection_status)
        else:
            # After either the connection has timed out or a connection is established,
            # will join the thread and display the result on screen for 3 seconds:
            self.thread.join()
            self.facade.set_text(self.instruction, self.instructions[self.current_phase])
            self.timer = 180

    def attempt_connection(self):
        """Starts a thread which will attempt to connect to the Utopia Hub."""
        self.thread = threading.Thread(target=self.threaded_connect, args=[self.noisetag])
        self.thread.start()

    def threaded_connect(self, noisetag):
        """
        Connects the noisetag to the Utopia Hub.
        Meant to run concurrently in order to not freeze up the application.

        Args:
            noisetag (Noisetag): Reference to the noisetag object
        """
        noisetag.connect(host=self.host, queryifhostnotfound=False, timeout_ms=5000)

    def get_connection_status(self):
        """Returns the connection status of the noisetag"""
        return self.connection_status
