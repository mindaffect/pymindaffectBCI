from bluepy.btle import Peripheral
from bluepy import btle
from sphero_constants import *
import struct
import time
import sys


class sphero_mini():
    def __init__(self, MACAddr, verbosity = 4, user_delegate = None):
        '''
        initialize class instance and then build collect BLE sevices and characteristics.
        Also sends text string to Anti-DOS characteristic to prevent returning to sleep,
        and initializes notifications (which is what the sphero uses to send data back to
        the client).
        '''
        self.verbosity = verbosity # 0 = Silent,
                                   # 1 = Connection/disconnection only
                                   # 2 = Init messages
                                   # 3 = Recieved commands
                                   # 4 = Acknowledgements
        self.sequence = 1
        self.v_batt = None # will be updated with battery voltage when sphero.getBatteryVoltage() is called
        self.firmware_version = [] # will be updated with firware version when sphero.returnMainApplicationVersion() is called

        if self.verbosity > 0:
            print("[INFO] Connecting to", MACAddr)
        self.p = Peripheral(MACAddr, "random") #connect

        if self.verbosity > 1:
            print("[INIT] Initializing")

        # Subscribe to notifications
        self.sphero_delegate = MyDelegate(self, user_delegate) # Pass a reference to this instance when initializing
        self.p.setDelegate(self.sphero_delegate)

        if self.verbosity > 1:
            print("[INIT] Read all characteristics and descriptors")
        # Get characteristics and descriptors
        self.API_V2_characteristic = self.p.getCharacteristics(uuid="00010002-574f-4f20-5370-6865726f2121")[0]
        self.AntiDOS_characteristic = self.p.getCharacteristics(uuid="00020005-574f-4f20-5370-6865726f2121")[0]
        self.DFU_characteristic = self.p.getCharacteristics(uuid="00020002-574f-4f20-5370-6865726f2121")[0]
        self.DFU2_characteristic = self.p.getCharacteristics(uuid="00020004-574f-4f20-5370-6865726f2121")[0]
        self.API_descriptor = self.API_V2_characteristic.getDescriptors(forUUID=0x2902)[0]
        self.DFU_descriptor = self.DFU_characteristic.getDescriptors(forUUID=0x2902)[0]

        # The rest of this sequence was observed during bluetooth sniffing:
        # Unlock code: prevent the sphero mini from going to sleep again after 10 seconds
        if self.verbosity > 1:
            print("[INIT] Writing AntiDOS characteristic unlock code")
        self.AntiDOS_characteristic.write("usetheforce...band".encode(), withResponse=True)

        # Enable DFU notifications:
        if self.verbosity > 1:
            print("[INIT] Configuring DFU descriptor")
        self.DFU_descriptor.write(struct.pack('<bb', 0x01, 0x00), withResponse = True)

        # No idea what this is for. Possibly a device ID of sorts? Read request returns '00 00 09 00 0c 00 02 02':
        if self.verbosity > 1:
            print("[INIT] Reading DFU2 characteristic")
        _ = self.DFU2_characteristic.read()

        # Enable API notifications:
        if self.verbosity > 1:
            print("[INIT] Configuring API dectriptor")
        self.API_descriptor.write(struct.pack('<bb', 0x01, 0x00), withResponse = True)

        self.wake()

        # Finished initializing:
        if self.verbosity > 1:
            print("[INIT] Initialization complete\n")

    def disconnect(self):
        if self.verbosity > 0:
            print("[INFO] Disconnecting")
        
        self.p.disconnect()

    def wake(self):
        '''
        Bring device out of sleep mode (can only be done if device was in sleep, not deep sleep).
        If in deep sleep, the device should be connected to USB power to wake.
        '''
        if self.verbosity > 2:
            print("[SEND {}] Waking".format(self.sequence))
        
        self._send(characteristic=self.API_V2_characteristic,
                   devID=deviceID['powerInfo'],
                   commID=powerCommandIDs["wake"],
                   payload=[]) # empty payload

        #self.getAcknowledgement("Wake")

    def sleep(self, deepSleep=False):
        '''
        Put device to sleep or deep sleep (deep sleep needs USB power connected to wake up)
        '''
        if deepSleep:
            sleepCommID=powerCommandIDs["deepSleep"]
            if self.verbosity > 0:
                print("[INFO] Going into deep sleep. Connect USB power to wake.")
        else:
            sleepCommID=powerCommandIDs["sleep"]
        self._send(characteristic=self.API_V2_characteristic,
                   devID=deviceID['powerInfo'],
                   commID=sleepCommID,
                   payload=[]) #empty payload

    def setLEDColor(self, red = None, green = None, blue = None):
        '''
        Set device LED color based on RGB vales (each can  range between 0 and 0xFF)
        '''
        if self.verbosity > 2:
            print("[SEND {}] Setting main LED colour to [{}, {}, {}]".format(self.sequence, red, green, blue))
        
        self._send(characteristic = self.API_V2_characteristic,
                  devID = deviceID['userIO'], # 0x1a
                  commID = userIOCommandIDs["allLEDs"], # 0x0e
                  payload = [0x00, 0x0e, red, green, blue])

        #self.getAcknowledgement("LED/backlight")

    def setBackLEDIntensity(self, brightness=None):
        '''
        Set device LED backlight intensity based on 0-255 values

        NOTE: this is not the same as aiming - it only turns on the LED
        '''
        if self.verbosity > 2:
            print("[SEND {}] Setting backlight intensity to {}".format(self.sequence, brightness))

        self._send(characteristic = self.API_V2_characteristic,
                  devID = deviceID['userIO'],
                  commID = userIOCommandIDs["allLEDs"],
                  payload = [0x00, 0x01, brightness])

        ##self.getAckknowledgement("LED/backlight")

    def roll(self, speed=None, heading=None):
        '''
        Start to move the Sphero at a given direction and speed.
        heading: integer from 0 - 360 (degrees)
        speed: Integer from 0 - 255
        
        Note: the zero heading should be set at startup with the resetHeading method. Otherwise, it may
        seem that the sphero doesn't honor the heading argument
        '''
        if self.verbosity > 2:
            print("[SEND {}] Rolling with speed {} and heading {}".format(self.sequence, speed, heading))
    
        if abs(speed) > 255:
            print("WARNING: roll speed parameter outside of allowed range (-255 to +255)")

        if speed < 0:
            speed = -1*speed+256 # speed values > 256 in the send packet make the spero go in reverse

        speedH = (speed & 0xFF00) >> 8
        speedL = speed & 0xFF
        headingH = (heading & 0xFF00) >> 8
        headingL = heading & 0xFF
        self._send(characteristic = self.API_V2_characteristic,
                  devID = deviceID['driving'],
                  commID = drivingCommands["driveWithHeading"],
                  payload = [speedL, headingH, headingL, speedH])

        #self.getAcknowledgement("Roll")

    def resetHeading(self):
        '''
        Reset the heading zero angle to the current heading (useful during aiming)
        Note: in order to manually rotate the sphero, you need to call stabilization(False).
        Once the heading has been set, call stabilization(True).
        '''
        if self.verbosity > 2:
            print("[SEND {}] Resetting heading".format(self.sequence))
    
        self._send(characteristic = self.API_V2_characteristic,
                  devID = deviceID['driving'],
                  commID = drivingCommands["resetHeading"],
                  payload = []) #empty payload

        #self.getAcknowledgement("Heading")

    def returnMainApplicationVersion(self):
        '''
        Sends command to return application data in a notification
        '''
        if self.verbosity > 2:
            print("[SEND {}] Requesting firmware version".format(self.sequence))
    
        self._send(self.API_V2_characteristic,
                   devID = deviceID['systemInfo'],
                   commID = SystemInfoCommands['mainApplicationVersion'],
                   payload = []) # empty

        #self.getAcknowledgement("Firmware")

    def getBatteryVoltage(self):
        '''
        Sends command to return battery voltage data in a notification.
        Data printed to console screen by the handleNotifications() method in the MyDelegate class.
        '''
        if self.verbosity > 2:
            print("[SEND {}] Requesting battery voltage".format(self.sequence))
    
        self._send(self.API_V2_characteristic,
                   devID=deviceID['powerInfo'],
                   commID=powerCommandIDs['batteryVoltage'],
                   payload=[]) # empty

        #self.getAcknowledgement("Battery")

    def stabilization(self, stab = True):
        '''
        Sends command to turn on/off the motor stabilization system (required when manually turning/aiming the sphero)
        '''
        if stab == True:
            if self.verbosity > 2:
                    print("[SEND {}] Enabling stabilization".format(self.sequence))
            val = 1
        else:
            if self.verbosity > 2:
                    print("[SEND {}] Disabling stabilization".format(self.sequence))
            val = 0
        self._send(self.API_V2_characteristic,
                   devID=deviceID['driving'],
                   commID=drivingCommands['stabilization'],
                   payload=[val])

        #self.getAcknowledgement("Stabilization")

    def wait(self, delay):
        '''
        This is a non-blocking delay command. It is similar to time.sleep(), except it allows asynchronous 
        notification handling to still be performed.
        '''
        start = time.time()
        while(1):
            self.p.waitForNotifications(0.001)
            if time.time() - start > delay:
                break

    def _send(self, characteristic=None, devID=None, commID=None, payload=[]):
        '''
        A generic "send" method, which will be used by other methods to send a command ID, payload and
        appropriate checksum to a specified device ID. Mainly useful because payloads are optional,
        and can be of varying length, to convert packets to binary, and calculate and send the
        checksum. For internal use only.

        Packet structure has the following format (in order):

        - Start byte: always 0x8D
        - Flags byte: indicate response required, etc
        - Virtual device ID: see sphero_constants.py
        - Command ID: see sphero_constants.py
        - Sequence number: Seems to be arbitrary. I suspect it is used to match commands to response packets (in which the number is echoed).
        - Payload: Could be varying number of bytes (incl. none), depending on the command
        - Checksum: See below for calculation
        - End byte: always 0xD8

        '''
        sendBytes = [sendPacketConstants["StartOfPacket"],
                    sum([flags["resetsInactivityTimeout"], flags["requestsResponse"]]),
                    devID,
                    commID,
                    self.sequence] + payload # concatenate payload list

        self.sequence += 1 # Increment sequence number, ensures we can identify response packets are for this command
        if self.sequence > 255:
            self.sequence = 0

        # Compute and append checksum and add EOP byte:
        # From Sphero docs: "The [checksum is the] modulo 256 sum of all the bytes
        #                   from the device ID through the end of the data payload,
        #                   bit inverted (1's complement)"
        # For the sphero mini, the flag bits must be included too:
        checksum = 0
        for num in sendBytes[1:]:
            checksum = (checksum + num) & 0xFF # bitwise "and to get modulo 256 sum of appropriate bytes
        checksum = 0xff - checksum # bitwise 'not' to invert checksum bits
        sendBytes += [checksum, sendPacketConstants["EndOfPacket"]] # concatenate

        # Convert numbers to bytes
        output = b"".join([x.to_bytes(1, byteorder='big') for x in sendBytes])

        #send to specified characteristic:
        characteristic.write(output, withResponse = True)

    def getAcknowledgement(self, ack):
        #wait up to 10 secs for correct acknowledgement to come in, including sequence number!
        start = time.time()
        while(1):
            self.p.waitForNotifications(1)
            if self.sphero_delegate.notification_seq == self.sequence-1: # use one less than sequence, because _send function increments it for next send. 
                if self.verbosity > 3:
                    print("[RESP {}] {}".format(self.sequence-1, self.sphero_delegate.notification_ack))
                self.sphero_delegate.clear_notification()
                break
            elif self.sphero_delegate.notification_seq >= 0:
                print("Unexpected ACK. Expected: {}/{}, received: {}/{}".format(
                    ack, self.sequence, self.sphero_delegate.notification_ack.split()[0],
                    self.sphero_delegate.notification_seq),
                    file=sys.stderr)
            if time.time() > start + 1:
                print("Timeout waiting for acknowledgement: {}/{}".format(ack, self.sequence), file=sys.stderr)
                break

# =======================================================================
# The following functions are experimental:
# =======================================================================

    def configureCollisionDetection(self,
                                     xThreshold = 50, 
                                     yThreshold = 50, 
                                     xSpeed = 50, 
                                     ySpeed = 50, 
                                     deadTime = 50, # in 10 millisecond increments
                                     method = 0x01, # Must be 0x01        
                                     callback = None):
        '''
        Appears to function the same as other Sphero models, however speed settings seem to have no effect. 
        NOTE: Setting to zero seems to cause bluetooth errors with the Sphero Mini/bluepy library - set to 
        255 to make it effectively disabled.

        deadTime disables future collisions for a short period of time to avoid repeat triggering by the same
        event. Set in 10ms increments. So if deadTime = 50, that means the delay will be 500ms, or half a second.
        
        From Sphero docs:
        
            xThreshold/yThreshold: An 8-bit settable threshold for the X (left/right) and Y (front/back) axes 
            of Sphero.

            xSpeed/ySpeed: An 8-bit settable speed value for the X and Y axes. This setting is ranged by the 
            speed, then added to xThreshold, yThreshold to generate the final threshold value.
        '''
        
        if self.verbosity > 2:
            print("[SEND {}] Configuring collision detection".format(self.sequence))
    
        self._send(self.API_V2_characteristic,
                   devID=deviceID['sensor'],
                   commID=sensorCommands['configureCollision'],
                   payload=[method, xThreshold, xSpeed, yThreshold, ySpeed, deadTime])

        self.collision_detection_callback = callback

        #self.getAcknowledgement("Collision")

    def configureSensorStream(self): # Use default values
        '''
        Send command to configure sensor stream using default values as found during bluetooth 
        sniffing of the Sphero Edu app.

        Must be called after calling configureSensorMask()
        '''
        bitfield1 = 0b00000000 # Unknown function - needs experimenting
        bitfield2 = 0b00000000 # Unknown function - needs experimenting
        bitfield3 = 0b00000000 # Unknown function - needs experimenting
        bitfield4 = 0b00000000 # Unknown function - needs experimenting

        if self.verbosity > 2:
            print("[SEND {}] Configuring sensor stream".format(self.sequence))
    
        self._send(self.API_V2_characteristic,
                   devID=deviceID['sensor'],
                   commID=sensorCommands['configureSensorStream'],
                   payload=[bitfield1, bitfield1, bitfield1, bitfield1])

        #self.getAcknowledgement("Sensor")

    def configureSensorMask(self,
                            sample_rate_divisor = 0x25, # Must be > 0
                            packet_count = 0,
                            IMU_pitch = False,
                            IMU_roll = False,
                            IMU_yaw = False,
                            IMU_acc_x = False,
                            IMU_acc_y = False,
                            IMU_acc_z = False,
                            IMU_gyro_x = False,
                            IMU_gyro_y = False,
                            IMU_gyro_z = False):

        '''
        Send command to configure sensor mask using default values as found during bluetooth 
        sniffing of the Sphero Edu app. From experimentation, it seems that these are he functions of each:
        
        Sampling_rate_divisor. Slow data EG: Set to 0x32 to the divide data rate by 50. Setting below 25 (0x19) causes 
                bluetooth errors        
        
        Packet_count: Select the number of packets to transmit before ending the stream. Set to zero to stream infinitely
        
        All IMU bool parameters: Toggle transmission of that value on or off (e.g. set IMU_acc_x = True to include the 
                X-axis accelerometer readings in the sensor stream)
        '''

        # Construct bitfields based on function parameters:
        IMU_bitfield1 = (IMU_pitch<<2) + (IMU_roll<<1) + IMU_yaw
        IMU_bitfield2 = ((IMU_acc_y<<7) + (IMU_acc_z<<6) + (IMU_acc_x<<5) + \
                         (IMU_gyro_y<<4) + (IMU_gyro_x<<2) + (IMU_gyro_z<<2))
        
        if self.verbosity > 2:
            print("[SEND {}] Configuring sensor mask".format(self.sequence))
    
        self._send(self.API_V2_characteristic,
                   devID=deviceID['sensor'],
                   commID=sensorCommands['sensorMask'],
                   payload=[0x00,               # Unknown param - altering it seems to slow data rate. Possibly averages multiple readings?
                            sample_rate_divisor,       
                            packet_count,       # Packet count: select the number of packets to stop streaming after (zero = infinite)
                            0b00,               # Unknown param: seems to be another accelerometer bitfield? Z-acc, Y-acc
                            IMU_bitfield1,
                            IMU_bitfield2,
                            0b00])              # reserved, Position?, Position?, velocity?, velocity?, Y-gyro, timer, reserved

        #self.getAcknowledgement("Mask")

        '''
        Since the sensor values arrive as unlabelled lists in the order that they appear in the bitfields above, we need 
        to create a list of sensors that have been configured.Once we have this list, then in the default_delegate class, 
        we can get sensor values as attributes of the sphero_mini class.
        e.g. print(sphero.IMU_yaw) # displays the current yaw angle
        '''

        # Initialize dictionary with sensor names as keys and their bool values (set by the user) as values:
        availableSensors = {"IMU_pitch" : IMU_pitch,
                            "IMU_roll" : IMU_roll,
                            "IMU_yaw" : IMU_yaw,
                            "IMU_acc_y" : IMU_acc_y,
                            "IMU_acc_z" : IMU_acc_z,
                            "IMU_acc_x" : IMU_acc_x,
                            "IMU_gyro_y" : IMU_gyro_y,
                            "IMU_gyro_x" : IMU_gyro_x,
                            "IMU_gyro_z" : IMU_gyro_z}
        
        # Create list of of only sensors that have been "activated" (set as true in the method arguments):
        self.configured_sensors = [name for name in availableSensors if availableSensors[name] == True]

    def sensor1(self): # Use default values
        '''
        Unknown function. Observed in bluetooth sniffing. 
        '''
        self._send(self.API_V2_characteristic,
                   devID=deviceID['sensor'],
                   commID=sensorCommands['sensor1'],
                   payload=[0x01])

        #self.getAcknowledgement("Sensor1")

    def sensor2(self): # Use default values
        '''
        Unknown function. Observed in bluetooth sniffing. 
        '''
        self._send(self.API_V2_characteristic,
                   devID=deviceID['sensor'],
                   commID=sensorCommands['sensor2'],
                   payload=[0x00])

        #self.getAcknowledgement("Sensor2")

# =======================================================================

class MyDelegate(btle.DefaultDelegate):

    '''
    This class handles notifications (both responses and asynchronous notifications).
    
    Usage of this class is described in the Bluepy documentation
    
    '''

    def __init__(self, sphero_class, user_delegate):
        self.sphero_class = sphero_class # for saving sensor values as attributes of sphero class instance
        self.user_delegate = user_delegate # to directly notify users of callbacks
        btle.DefaultDelegate.__init__(self)
        self.clear_notification()
        self.notificationPacket = []

    def clear_notification(self):
        self.notification_ack = "DEFAULT ACK"
        self.notification_seq = -1

    def bits_to_num(self, bits):
        '''
        This helper function decodes bytes from sensor packets into single precision floats. Encoding follows the
        the IEEE-754 standard.
        '''
        num = int(bits, 2).to_bytes(len(bits) // 8, byteorder='little')
        num = struct.unpack('f', num)[0]
        return num

    def handleNotification(self, cHandle, data):
        '''
        This method acts as an interrupt service routine. When a notification comes in, this
        method is invoked, with the variable 'cHandle' being the handle of the characteristic that
        sent the notification, and 'data' being the payload (sent one byte at a time, so the packet
        needs to be reconstructed)  

        The method keeps appending bytes to the payload packet byte list until end-of-packet byte is
        encountered. Note that this is an issue, because 0xD8 could be sent as part of the payload of,
        say, the battery voltage notification. In future, a more sophisticated method will be required.
        '''
        # Allow the user to intercept and process data first..
        if self.user_delegate != None:
            if self.user_delegate.handleNotification(cHandle, data):
                return

        for data_byte in data: # parse each byte separately (sometimes they arrive simultaneously)

            self.notificationPacket.append(data_byte) # Add new byte to packet list

            # If end of packet (need to find a better way to segment the packets):
            if data_byte == sendPacketConstants['EndOfPacket']:
                # Once full the packet has arrived, parse it:
                # Packet structure is similar to the outgoing send packets (see docstring in sphero_mini._send())
                
                # Attempt to unpack. Might fail if packet is too badly corrupted
                try:
                    start, flags_bits, devid, commcode, seq, *notification_payload, chsum, end = self.notificationPacket
                except ValueError:
                    print("Warning: notification packet unparseable", self.notificationPacket, file=sys.stderr)
                    self.notificationPacket = [] # Discard this packet
                    return # exit

                # Compute and append checksum and add EOP byte:
                # From Sphero docs: "The [checksum is the] modulo 256 sum of all the bytes
                #                   from the device ID through the end of the data payload,
                #                   bit inverted (1's complement)"
                # For the sphero mini, the flag bits must be included too:
                checksum_bytes = [flags_bits, devid, commcode, seq] + notification_payload
                checksum = 0 # init
                for num in checksum_bytes:
                    checksum = (checksum + num) & 0xFF # bitwise "and to get modulo 256 sum of appropriate bytes
                checksum = 0xff - checksum # bitwise 'not' to invert checksum bits
                if checksum != chsum: # check computed checksum against that recieved in the packet
                    print("Warning: notification packet checksum failed", self.notificationPacket, file=sys.stderr)
                    self.notificationPacket = [] # Discard this packet
                    return # exit

                # Check if response packet:
                if flags_bits & flags['isResponse']: # it is a response

                    # Use device ID and command code to determine which command is being acknowledged:
                    if devid == deviceID['powerInfo'] and commcode == powerCommandIDs['wake']:
                        self.notification_ack = "Wake acknowledged" # Acknowledgement after wake command
                        
                    elif devid == deviceID['driving'] and commcode == drivingCommands['driveWithHeading']:
                        self.notification_ack = "Roll command acknowledged"

                    elif devid == deviceID['driving'] and commcode == drivingCommands['stabilization']:
                        self.notification_ack = "Stabilization command acknowledged"

                    elif devid == deviceID['userIO'] and commcode == userIOCommandIDs['allLEDs']:
                        self.notification_ack = "LED/backlight color command acknowledged"

                    elif devid == deviceID['driving'] and commcode == drivingCommands["resetHeading"]:
                        self.notification_ack = "Heading reset command acknowledged"

                    elif devid == deviceID['sensor'] and commcode == sensorCommands["configureCollision"]:
                        self.notification_ack = "Collision detection configuration acknowledged"

                    elif devid == deviceID['sensor'] and commcode == sensorCommands["configureSensorStream"]:
                        self.notification_ack = "Sensor stream configuration acknowledged"

                    elif devid == deviceID['sensor'] and commcode == sensorCommands["sensorMask"]:
                        self.notification_ack = "Mask configuration acknowledged"

                    elif devid == deviceID['sensor'] and commcode == sensorCommands["sensor1"]:
                        self.notification_ack = "Sensor1 acknowledged"

                    elif devid == deviceID['sensor'] and commcode == sensorCommands["sensor2"]:
                        self.notification_ack = "Sensor2 acknowledged"

                    elif devid == deviceID['powerInfo'] and commcode == powerCommandIDs['batteryVoltage']:
                        V_batt = notification_payload[2] + notification_payload[1]*256 + notification_payload[0]*65536
                        V_batt /= 100 # Notification gives V_batt in 10mV increments. Divide by 100 to get to volts.
                        self.notification_ack = "Battery voltage:" + str(V_batt) + "v"
                        self.sphero_class.v_batt = V_batt

                    elif devid == deviceID['systemInfo'] and commcode == SystemInfoCommands['mainApplicationVersion']:
                        version = '.'.join(str(x) for x in notification_payload)
                        self.notification_ack = "Firmware version: " + version
                        self.sphero_class.firmware_version = notification_payload
                                                
                    else:
                        self.notification_ack = "Unknown acknowledgement" #print(self.notificationPacket)
                        print(self.notificationPacket, "===================> Unknown ack packet")

                    self.notification_seq = seq

                else: # Not a response packet - therefore, asynchronous notification (e.g. collision detection, etc):
                    
                    # Collision detection:
                    if devid == deviceID['sensor'] and commcode == sensorCommands['collisionDetectedAsync']:
                        # The first four bytes are data that is still un-parsed. the remaining unsaved bytes are always zeros
                        _, _, _, _, _, _, axis, _, Y_mag, _, X_mag, *_ = notification_payload
                        if axis == 1: 
                            dir = "Left/right"
                        else:
                            dir = 'Forward/back'
                        print("Collision detected:")
                        print("\tAxis:", dir)
                        print("\tX_mag:", X_mag)
                        print("\tY_mag:", Y_mag)

                        #if self.sphero_class.collision_detection_callback is not None:
                         #   self.notificationPacket = [] # need to clear packet, in case new notification comes in during callback
                          #  self.sphero_class.collision_detection_callback()

                    # Sensor response:
                    elif devid == deviceID['sensor'] and commcode == sensorCommands['sensorResponse']:
                        # Convert to binary, pad bytes with leading zeros:
                        val = ''
                        for byte in notification_payload:
                            val += format(int(bin(byte)[2:], 2), '#010b')[2:]
                        
                        # Break into 32-bit chunks
                        nums = []
                        while(len(val) > 0):
                            num, val = val[:32], val[32:] # Slice off first 16 bits
                            nums.append(num)
                        
                        # convert from raw bits to float:
                        nums = [self.bits_to_num(num) for num in nums]

                        # Set sensor values as class attributes:
                        for name, value in zip(self.sphero_class.configured_sensors, nums):
                            setattr(self.sphero_class, name, value)
                        
                    # Unrecognized packet structure:
                    else:
                        self.notification_ack = "Unknown asynchronous notification" #print(self.notificationPacket)
                        print(self.notificationPacket, "===================> Unknown async packet")
                        
                self.notificationPacket = [] # Start new payload after this byte
