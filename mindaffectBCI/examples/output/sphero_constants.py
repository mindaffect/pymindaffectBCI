'''
Known Peripheral UUIDs, obtained by querying using the Bluepy module:
=====================================================================
Anti DOS Characteristic <00020005-574f-4f20-5370-6865726f2121>
Battery Level Characteristic <Battery Level>
Peripheral Preferred Connection Parameters Characteristic <Peripheral Preferred Connection Parameters>
API V2 Characteristic <00010002-574f-4f20-5370-6865726f2121>
DFU Control Characteristic <00020002-574f-4f20-5370-6865726f2121>
Name Characteristic <Device Name>
Appearance Characteristic <Appearance>
DFU Info Characteristic <00020004-574f-4f20-5370-6865726f2121>
Service Changed Characteristic <Service Changed>
Unknown1 Characteristic <00020003-574f-4f20-5370-6865726f2121>
Unknown2 Characteristic <00010003-574f-4f20-5370-6865726f2121>

The rest of the values saved in the dictionaries below, were borrowed from
@igbopie's javacript library, which is available at https://github.com/igbopie/spherov2.js

'''

deviceID = {"apiProcessor": 0x10,                   # 16
            "systemInfo": 0x11,                     # 17
            "powerInfo": 0x13,                      # 19
            "driving": 0x16,                        # 22
            "animatronics": 0x17,                   # 23
            "sensor": 0x18,                         # 24
            "something": 0x19,                      # 25
            "userIO": 0x1a,                         # 26
            "somethingAPI": 0x1f}                   # 31

SystemInfoCommands = {"mainApplicationVersion": 0x00,   # 00
                      "bootloaderVersion": 0x01,    # 01
                      "something": 0x06,            # 06
                      "something2": 0x13,           # 19
                      "something6": 0x12,           # 18    
                      "something7": 0x28}           # 40

sendPacketConstants = {"StartOfPacket": 0x8d,       # 141
                       "EndOfPacket": 0xd8}         # 216

userIOCommandIDs = {"allLEDs": 0x0e}                # 14

flags= {"isResponse": 0x01,                         # 0x01
        "requestsResponse": 0x02,                   # 0x02
        "requestsOnlyErrorResponse": 0x04,          # 0x04
        "resetsInactivityTimeout": 0x08}            # 0x08

powerCommandIDs={"deepSleep": 0x00,                 # 0
                "sleep": 0x01,                      # 01
                "batteryVoltage": 0x03,             # 03
                "wake": 0x0D,                       # 13
                "something": 0x05,                  # 05 
                "something2": 0x10,                 # 16         
                "something3": 0x04,                 # 04
                "something4": 0x1E}                 # 30

drivingCommands={"rawMotor": 0x01,                  # 1
                 "resetHeading": 0x06,              # 6    
                 "driveAsSphero": 0x04,             # 4
                 "driveAsRc": 0x02,                 # 2
                 "driveWithHeading": 0x07,          # 7
                 "stabilization": 0x0C}             # 12

sensorCommands={'sensorMask': 0x00,                 # 00
                'sensorResponse': 0x02,             # 02
                'configureCollision': 0x11,         # 17
                'collisionDetectedAsync': 0x12,     # 18
                'resetLocator': 0x13,               # 19
                'enableCollisionAsync': 0x14,       # 20
                'sensor1': 0x0F,                    # 15
                'sensor2': 0x17,                    # 23
                'configureSensorStream': 0x0C}      # 12
