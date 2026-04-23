#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

BLEService batService("180D");          // arbitrary UUID
BLECharacteristic dataChar(
  "2A37",
  BLERead | BLENotify,
  64 // max length of string packet
);

const int PIEZO_PIN = A0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  pinMode(PIEZO_PIN, INPUT);

  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("SmartBat");
  BLE.setAdvertisedService(batService);
  batService.addCharacteristic(dataChar);
  BLE.addService(batService);

  dataChar.writeValue("ready");

  BLE.advertise();
  Serial.println("BLE device active, waiting for connections...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    while (central.connected()) {
      sendSensorPacket();
      delay(20); // 50 Hz
    }

    Serial.println("Disconnected");
  }
}

void sendSensorPacket() {
  float ax, ay, az;
  float gx, gy, gz;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    int piezo = analogRead(PIEZO_PIN);
    unsigned long t = millis();

    char buffer[64];
    snprintf(
      buffer, sizeof(buffer),
      "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%lu",
      ax, ay, az, gx, gy, gz, piezo, t
    );

    dataChar.writeValue(buffer);
  }
}
