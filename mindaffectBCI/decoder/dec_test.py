#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

# Pre-processing:
#  filter = 1-12Hz -> stop band 0-1, 12-inf
#  downsample = 30hz
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, butterfilt_and_downsample
stopband = ((0, 1), (12, -1))
fs_out = 30
ppfn = butterfilt_and_downsample(order=6, stopband=stopband, fs_out=fs_out)
ui = UtopiaDataInterface(data_preprocessor=ppfn) 

# Classifier:
#   * response length 700ms (as the p300 is from 300-600ms)
tau_ms = 700
#   * start of target stimulus vs. start of any stimuls
#       -> 'rising-edge' and 'non-target rising edge'
evtlabs = ('re', 'ntre')
#   * rank-3 decomposition, as we tend to get multiple component, e.g. perceptual, P3a, P3b
rank = 3
#  CCA classifier
from mindaffectBCI.decoder.model_fitting import MultiCCA
clsfr = MultiCCA(tau=int(fs_out*tau_ms/1000), rank=rank, evtlabs=evtlabs)

from mindaffectBCI.decoder import decoder
decoder.mainloop(ui=ui, clsfr=clsfr, calplots=True)