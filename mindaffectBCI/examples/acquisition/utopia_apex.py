import argparse
import queue
import signal
import os
import time

from mindaffectBCI import utopiaclient
import numpy as np

# set up signal handler
signal.signal(signal.SIGINT, lambda signum, frame: shutdown())
signal.signal(signal.SIGTERM, lambda signum, frame: shutdown())

LOGINTERVAL_S = 3
t0 = None
nextLogTime = None


def printLog(nSamp, nBlock):
    """textual logging of the data arrivals etc."""
    global t0, nextLogTime
    t = time.time()
    if t0 is None:
        t0 = t
        nextLogTime = t
    if t > nextLogTime:
        elapsed = time.time() - t0
        print(
            "%d %d %f %f (samp,blk,s,hz)" % (nSamp, nBlock, elapsed, nSamp / elapsed),
            flush=True,
        )
        nextLogTime = t + LOGINTERVAL_S


def run(host=None, **_):
    """run the acquisition process for the SAGA apex device
    Args:
        host ([type], optional): ip-address of the minaffectBCI Hub, auto-discover
                                 if None. Defaults to None.

    """
    global dev
    from mindaffectBCI.examples.acquisition.apex_sdk.tmsi_sdk import (
        TMSiSDK,
        DeviceType,
        DeviceInterfaceType,
        MeasurementType,
    )
    from mindaffectBCI.examples.acquisition.apex_sdk.sample_data_server.sample_data_server import (
        SampleDataServer,
    )

    # Execute a device discovery. This returns a list of device-objects for every discovered device.
    TMSiSDK().discover(DeviceType.apex, DeviceInterfaceType.usb)
    discoveryList = TMSiSDK().get_device_list()
    assert (
        len(discoveryList) == 1
    ), f"Found {len(discoveryList)} TMSi APEX devices. Makes sure exactly 1 is connected instead."
    # Get the handle to the first discovered device.
    dev = discoveryList[0]

    # Open a connection to APEX
    dev.open()

    # Load the EEG channel set and configuration
    print("load EEG config")
    dev.import_configuration(
        os.path.join(os.path.dirname(__file__), "APEX_config_EEG32.xml")
    )

    # TODO: select electrodes + references
    # # Set reference to first channel only
    # dev.set_device_references(list_references = [1 if i == 0 else 0 for i in range(len(dev.get_device_channels()))],
    #                           list_indices = [i for i in range(len(dev.get_device_channels()))])

    # TODO: set sampling freq type
    # dev.set_device_sampling_config(sampling_frequency = ApexEnums.TMSiBaseSampleRate.Binary)
    # TODO: set sampling freq

    ch_names = ["Fz"]
    print("Active Channel list:\n")
    for idx, ch in enumerate(dev.get_device_channels()):
        print(f"[{idx}] : [{ch.name}] in [{ch.unit_name}]")
        ch_names.append(ch.name)
    nstream = len(ch_names)

    # 1. create the queue-object
    q_sample_sets = queue.Queue(1000)

    # 2. Register at the <sample_data_server> your <queue> and for which device you
    #    want to receive the sample-data
    SampleDataServer().register_consumer(dev.get_id(), q_sample_sets)

    # connect to the utopia client
    client = utopiaclient.UtopiaClient()
    client.disableHeartbeats()  # disable heartbeats as we're a datapacket source
    client.autoconnect(host)

    # don't subscribe to anything
    client.sendMessage(utopiaclient.Subscribe(None, ""))
    fSample = 1000  # Hz, i.e. the maximum sample rate of the APEX
    client.sendMessage(utopiaclient.DataHeader(None, fSample, nstream, ch_names))

    dev.start_measurement(MeasurementType.APEX_EEG)

    nBlock = 0
    sampling = True
    while not q_sample_sets.empty() or sampling:
        samples = []
        sd = q_sample_sets.get()
        q_sample_sets.task_done()
        samples = np.reshape(
            sd.samples, (sd.num_samples_per_sample_set, sd.num_sample_sets), order="F"
        )

        client.sendMessage(utopiaclient.DataPacket(client.getTimeStamp(), samples))
        nBlock = nBlock + 1
        # log
        printLog(len(samples), nBlock)

    # Close the connection to the device
    dev.stop_measurement()
    dev.close()


def shutdown():
    from mindaffectBCI.examples.acquisition.apex_sdk.tmsi_sdk import DeviceState

    # stop measurement if active
    if dev.get_device_state() == DeviceState.sampling:
        dev.stop_measurement()
    # close the connection if active
    if dev.get_device_state() == DeviceState.connected:
        dev.close()

    exit(0)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, help="hub host address", default="localhost"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run(**vars(parse_arguments()))
