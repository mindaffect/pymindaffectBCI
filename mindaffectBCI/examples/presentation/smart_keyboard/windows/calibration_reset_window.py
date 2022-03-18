"""
This modules contains a single class that shows text that calibration reset has been successful.

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


class CalibrationResetWindow(Window):
	"""
	Subclass of Window, designed for giving the user feedback that resetting the Calibration Model has been successful.

	Args:
		parent (windows.window.Window): The parent of this window.
		facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
		style (dict): Style instructions for the keyboard.
		use_flickering (bool): Activates or deactivates flickering.
		noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

	Attributes:
		self.text (Object): Framework dependent text object which holds the text.
	"""

	def __init__(self, parent, facade, style, use_flickering, noisetag=None):
		super().__init__(
			parent=parent,
			facade=facade,
			style=style,
			use_flickering=use_flickering,
			noisetag=noisetag
		)

		self.text_success = "Successfully reset the Calibration model\nPress any key to go back to the menu"
		self.text_fail = "No noisetag connected, can't reset\nPress any key to go back to the menu"

		self.text_object = self.facade.create_text(text=self.text_success, col=style["text_color"], pos=(.5, .5))

	def activate(self):
		# resets the noisetag model
		if self.noisetag is not None:
			self.noisetag.modeChange("reset")
			self.facade.set_text(self.text_object, self.text_success)
		else:
			self.facade.set_text(self.text_object, self.text_fail)
		self.facade.toggle_text_render(self.text_object, True)

	def deactivate(self):
		self.facade.toggle_text_render(self.text_object, False)

	def draw(self, noisetag, last_flip_time, target_idx=-1):
		if self.facade.key_event():
			self.parent.switch_window("Menu")
