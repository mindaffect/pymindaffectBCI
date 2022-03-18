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
from mindaffectBCI.examples.presentation.smart_keyboard.key_type import KeyType
from mindaffectBCI.examples.presentation.smart_keyboard.key import Key
from math import log10
from collections import deque
from mindaffectBCI.utopiaclient import DataPacket
import time
from statistics import median


class ElectrodeQualityWindow(Window):
    """
    Subclass of Window, designed for displaying the quality of the electrodes connected.

    Args:
        parent (windows.window.Window): A reference to the wrapping application.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style (dict): General style instructions used for coloring.
        use_flickering (bool): Activates or deactivates flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

    Attributes:
        parent (windows.window.Window): A reference to the wrapping application.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style (dict): General style instructions used for coloring.
        use_flickering (bool): Activates or deactivates flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

        texts (dict): Dictionary of different instruction texts that are being cycled through depending on the state of the window.
        phases (list): List of the keys present within the texts dictionary.
        phase (int): The phase we are currently in.
        text (Object): Framework dependent text object which holds our text.
        dataringbuffer (collections.deque): A circular buffer used for holding the signal values we receive from Noisetag.
        datawindow_ms (int): The time window in milliseconds, indicates how long the signal value window is supposed to be.
        isRunning (bool): Indicates whether the quality screen is running or not.
        t0 (float): Time 0, the moment we start recording the EEG signals.
        nch (int): The number of channels that are connected.
        label (list): List of the labels that we receive, they represent signal noise values.
        linebbox (list): A bounding box for the signal graphs, helps in calculating coordinates.
        keys (list): List of keys that are created for the signal quality screen.
    """

    def __init__(self, parent, facade, style, use_flickering, noisetag=None):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        self.texts = {
            'instruction': "Electrode Quality\nAdjust headset until all electrodes are green\n"
                           "(or noise to signal ratio < 5)\nPress any key to abort",
            'abort': 'Press any key on your real keyboard to return to the menu',
            'flickering_off': 'Flickering isn\'t turned on and this screen won\'t work\n'
                              'Press any key on your real keyboard to return to the menu'
        }

        # For handling the different phases of the electrode quality screen
        self.phases = ['instruction', 'abort', 'flickering_off']
        self.phase = 0

        # Text object
        self.text = self.facade.create_text(text=self.texts[self.phases[self.phase]],
                                            pos=(0.5, 0.95),
                                            col=self.style['text_color'],
                                            align_hor='center',
                                            align_vert='top')

        # Starts building the screen upon initialisation:
        self.dataringbuffer = deque()  # deque so efficient sliding data window
        self.datawindow_ms = 5000  # 5seconds data plotted
        self.isRunning = False
        self.t0 = None

        # Creates all the necessary items using standard settings, default nch value is 4:
        self.nch = 4
        self.label = [None] * self.nch
        self.linebbox = [None] * self.nch

        self.keys = self.__build_quality_screen()

        self.windows = None
        self.active_window = None
        self.switched_windows = False

    def activate(self):
        """Activates all visual and functional elements of the Window."""
        self.reset()
        self.phase = 0
        self.facade.set_text(self.text, self.texts[self.phases[self.phase]])
        self.facade.toggle_text_render(self.text, True)
        for key in self.keys:
            key.toggle_render(True)

    def deactivate(self):
        """Deactivates all visual and functional elements of the Window."""
        self.facade.toggle_text_render(self.text, False)
        for key in self.keys:
            key.toggle_render(False)

        # Deactivate some things in the noisetag that had been activated by draw:
        if self.noisetag is not None:
            self.noisetag.removeSubscription("D")
            self.noisetag.modeChange("idle")

    def switch_window(self, window_name):
        """
        Switches to Window specified by window_name.

        Args:
            window_name (str): The name of the Window to switch to.
        """
        if self.windows:
            if window_name in self.windows:
                self.windows[self.active_window].deactivate()
                self.windows[window_name].activate()
                self.active_window = window_name
                self.switched_windows = True
            else:
                self.parent.switch_window(window_name)
                self.switched_windows = True
        else:
            self.parent.switch_window(window_name)
            self.switched_windows = True

    def draw(self, noisetag, last_flip_time, target_idx=-1):
        if self.facade.key_event():
            if self.phase == 0:
                # In quality phase:
                self.phase = 1
                self.facade.set_text(self.text, self.texts[self.phases[self.phase]])
            else:
                # Both phases 1 and 2 indicate that this button press should exit:
                self.switch_window("Menu")
                return

        if self.phase == 0:
            if not self.use_flickering:
                # When flickering is disabled, exit screen upon next button press:
                self.phase = 2
                self.facade.set_text(self.text, self.texts[self.phases[self.phase]])
            else:
                # ### START: Most of this code is from the selectionMatrix.py in Mindaffect's Project,
                # adapted to fit our particular implementation: START ###

                if not self.isRunning:
                    self.isRunning = True  # Mark that we're running.
                    self.t0 = self.__get_time_stamp()
                    self.noisetag.addSubscription("D")  # Subscribe to "Datapacket" messages.
                    self.noisetag.modeChange("ElectrodeQuality")
                    self.dataringbuffer.clear()

                # Get the sig qualities:
                electrode_qualities = self.noisetag.getLastSignalQuality()
                if not electrode_qualities:  # Default to 4 off qualities.
                    electrode_qualities = [.5] * self.nch

                # If default value turns out to be different than actual number of electrodes, update accordingly:
                if len(electrode_qualities) != self.nch:
                    self.__update_nch(len(electrode_qualities))

                # Update the colors:
                issig2noise = True  # any([s>1.5 for s in electrodeQualities])
                for i, qual in enumerate(electrode_qualities):
                    # Change label:
                    self.keys[i].set_label_text("%d: %3.1f" % (i + 1, qual))

                    if issig2noise:
                        qual = log10(qual) / 1  # n2s=50->1 n2s=10->.5 n2s=1->0

                    qual = max(0, min(1, qual))
                    qualcolor = (int(255 * qual), int(255 * (1 - qual)), 0)  # red=bad, green=good

                    # Set color:
                    self.keys[i].change_color(qualcolor)

                # Get raw signals:
                msgs = self.noisetag.getNewMessages()
                for m in msgs:
                    if m.msgID == DataPacket.msgID:
                        self.dataringbuffer.extend(m.samples)
                        if self.__get_time_stamp() > self.t0 + self.datawindow_ms:  # slide buffer
                            # Remove same number of samples we've just added:
                            for i in range(len(m.samples)):
                                self.dataringbuffer.popleft()

                if self.dataringbuffer:
                    if len(self.dataringbuffer[0]) != self.nch:
                        self.__update_nch(len(self.dataringbuffer[0]))

                    # CAR, signal reduction to give better noise readings:
                    dataringbuffer = []
                    for t in self.dataringbuffer:
                        mu = median(t)
                        dataringbuffer.append([c - mu for c in t])
                    
                    # tail centering for plotting
                    # compute average offset over the last 100 samples
                    offset = [0 for _ in range(self.nch)]
                    N = 0
                    for t in dataringbuffer[-100:]:
                        for i,c in enumerate(t):
                            offset[i] = offset[i] + c
                        N = N + 1
                    offset = [ o/N for o in offset ]
                    # subtract this offset
                    for i,t in enumerate(dataringbuffer):
                        dataringbuffer[i] = [c - offset[j] for j,c in enumerate(t)]

                    # ### END: Most of this code is from the selectionMatrix.py in Mindaffect's Project, adapted to fit
                    # our particular implementation: END ###

                    # Part of the preprocessing of the Mindaffect code was removed as it had to with scaling of the
                    # graphs which we solved in a different way, hence we believe it isn't needed here anymore.

                    # Draw the vertices:
                    buffer_length = len(dataringbuffer)
                    vertex_list = [[] for i in range(self.nch)]
                    colors = [(255,0,0), (0,255,0), (0,0,255), (128,128,0)]

                    # Calculate the coordinates for each vertex in our graph:
                    for i, t in enumerate(dataringbuffer):
                        for j in range(self.nch):
                            xscale = self.linebbox[j][0]
                            yscale = self.linebbox[j][1]
                            width = self.linebbox[j][2]
                            height = self.linebbox[j][3]
                            vertex = (xscale + (width / (buffer_length - 1)) * i, yscale + t[j] * (height / 2))
                            vertex_list[j].append(vertex)

                    # Draw all the graphs using the vertex lists:
                    for i, vertices in enumerate(vertex_list):
                        graph = self.facade.create_line(vertices, colors[i % 4])
                        if hasattr(graph,'draw') :
                            graph.draw()
                        else:
                            for g in graph: g.draw()

    # end of inherited functions
    def reset(self):
        """Resets running status"""
        self.isRunning = False

    def __update_nch(self, nch):
        """
        Updates the number of channels we have and updates all objects that are dependent on nch.

        Args:
            nch (int): The number of channels or electrodes.
        """
        self.nch = nch
        self.label = [None] * self.nch
        self.linebbox = [None] * self.nch

        # Deactivate old keys:
        for key in self.keys:
            key.toggle_render(False)

        # Build new screen with updated number of channels:
        self.keys = self.__build_quality_screen()

        # Rerender keys:
        for key in self.keys:
            key.toggle_render(True)

    def __build_quality_screen(self):
        """Constructs the Electrode Quality screen."""
        # Settings for button calculation:
        key_padding = 0.02
        v_ratio = 0.7
        h_ratio = 0.2
        h_padding = 0.02
        v_padding = 0.05
        key_height = (v_ratio - (self.nch - 1) * key_padding - v_padding * 2) / self.nch
        key_width = h_ratio - h_padding * 2

        keys = []

        for i in range(self.nch):
            # Create all buttons:
            y_pos = v_padding + (i + 0.5) * key_height + i * key_padding
            x_pos = h_padding + 0.5 * key_width
            size = (key_width, key_height)
            pos = (x_pos, y_pos)
            key_type = KeyType["BASIC_KEY"]
            key_label = '0.0'
            keys.append(
                Key(self.facade, key_type, size, pos, key_label, self.button_color,
                    self.line_color, self.label_color))
            self.linebbox[i] = (h_ratio, y_pos, 1 - h_ratio - h_padding, key_height / 2)  # x, y, w, h

        return keys

    def __get_time_stamp(self):
        """
        Gets a timestamp

        Returns:
            (int) The timestamp.
        """
        if self.noisetag is not None:
            return self.noisetag.getTimeStamp()

        # If not connected to utopiaclient:
        return int(time.perf_counter() * 1000 % (1 << 31))
