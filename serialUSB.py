import serial
import time

def runSerial():
    ON = bytes.fromhex("A0 01 01 A2")
    OFF = bytes.fromhex("A0 01 00 A1")
    ser = serial.Serial('COM4', 9600)  # open serial port
    print(ser)
    ser.write(ON)
    time.sleep(0.5)
    ser.write(OFF)
    ser.close()
# Call the subroutine
runSerial()
