// ============================================================
// HandGlow Firmware v2.0 — Flex Sensors + Dual MPU6050 (Accel + Gyro)
// ============================================================
// --- Finger-to-Pin Mapping ---
// Right Hand: Thumb=D15, Index=D4,  Middle=D34, Ring=D35, Pinky=D32
// Left  Hand: Thumb=D33, Index=D25, Middle=D26, Ring=D27,  Pinky=D14
//
// --- MPU6050 I2C Addresses ---
// Right Hand MPU: AD0 → 3V3 → address 0x69
// Left  Hand MPU: AD0 → GND → address 0x68
//
// --- Serial Output (22 channels) ---
// DATA,RT:val,RI:val,...,LP:val,aX1,aY1,aZ1,gX1,gY1,gZ1,aX2,aY2,aZ2,gX2,gY2,gZ2
//      └── 10 flex sensors ──┘  └── Right MPU (0x69) ──┘  └── Left MPU (0x68) ──┘
// ============================================================

#include <Wire.h>

// ─── MPU6050 Addresses ───
const uint8_t MPU_RIGHT = 0x69;  // Right hand (AD0 → 3V3)
const uint8_t MPU_LEFT  = 0x68;  // Left hand  (AD0 → GND)

// ─── Flex Sensor Config ───
const int numFlexSensors = 10;

const int flexPins[numFlexSensors] = {
  15,  // Right Thumb
  4,   // Right Index
  34,  // Right Middle
  35,  // Right Ring
  32,  // Right Pinky
  33,  // Left Thumb
  25,  // Left Index
  26,  // Left Middle
  27,  // Left Ring
  14   // Left Pinky
};

const char* fingerNames[numFlexSensors] = {
  "RT", // Right Thumb
  "RI", // Right Index
  "RM", // Right Middle
  "RR", // Right Ring
  "RP", // Right Pinky
  "LT", // Left Thumb
  "LI", // Left Index
  "LM", // Left Middle
  "LR", // Left Ring
  "LP"  // Left Pinky
};

// ─── Smoothing (EMA Filter) ───
const float alpha = 0.15;
float smoothedValues[numFlexSensors];

// ─── Hardware Settings ───
const bool invertData = false;

// ─── Sampling Rate ───
const unsigned long SAMPLE_INTERVAL_MS = 40;  // 25 Hz (was 50ms)
unsigned long lastSampleTime = 0;

// ─── IMU Data Structure ───
struct IMUData {
  float ax, ay, az;  // Accelerometer in g
  float gx, gy, gz;  // Gyroscope in °/s
  bool ok;           // true if read succeeded
};

// ─── MPU6050 Status ───
bool mpuRightOk = false;
bool mpuLeftOk  = false;

// ============================================================
// Initialize a single MPU6050
// ============================================================
bool initMPU(uint8_t addr) {
  // Wake up MPU6050 (clear sleep bit)
  Wire.beginTransmission(addr);
  Wire.write(0x6B);  // PWR_MGMT_1
  Wire.write(0x00);  // Wake up
  if (Wire.endTransmission(true) != 0) return false;

  delay(10);

  // Set accelerometer range to ±4g
  Wire.beginTransmission(addr);
  Wire.write(0x1C);  // ACCEL_CONFIG
  Wire.write(0x08);  // ±4g (AFS_SEL = 1)
  Wire.endTransmission(true);

  // Set gyroscope range to ±250°/s (most sensitive)
  Wire.beginTransmission(addr);
  Wire.write(0x1B);  // GYRO_CONFIG
  Wire.write(0x00);  // ±250°/s (FS_SEL = 0)
  Wire.endTransmission(true);

  // Set Digital Low Pass Filter for smoother readings
  Wire.beginTransmission(addr);
  Wire.write(0x1A);  // CONFIG
  Wire.write(0x03);  // DLPF ~44Hz bandwidth (good for hand gestures)
  Wire.endTransmission(true);

  return true;
}

// ============================================================
// Read full IMU data (accel + gyro) from MPU6050
// Reads 14 bytes: accel(6) + temp(2) + gyro(6)
// ============================================================
IMUData readIMU(uint8_t addr) {
  IMUData d = {0, 0, 0, 0, 0, 0, false};

  Wire.beginTransmission(addr);
  Wire.write(0x3B);  // Start at ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return d;

  Wire.requestFrom(addr, (uint8_t)14, (uint8_t)true);
  if (Wire.available() < 14) return d;

  // Accelerometer raw (±4g range → 8192 LSB/g)
  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();

  // Temperature (skip 2 bytes)
  Wire.read(); Wire.read();

  // Gyroscope raw (±250°/s range → 131 LSB/°/s)
  int16_t gx = (Wire.read() << 8) | Wire.read();
  int16_t gy = (Wire.read() << 8) | Wire.read();
  int16_t gz = (Wire.read() << 8) | Wire.read();

  // Convert to physical units
  d.ax = ax / 8192.0f;   // g
  d.ay = ay / 8192.0f;
  d.az = az / 8192.0f;
  d.gx = gx / 131.0f;    // °/s
  d.gy = gy / 131.0f;
  d.gz = gz / 131.0f;
  d.ok = true;
  return d;
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  // Initialize I2C
  Wire.begin(21, 22);  // SDA=21, SCL=22 (ESP32 default)
  Wire.setClock(400000); // 400kHz fast mode

  // Pre-fill smoothing array
  for (int i = 0; i < numFlexSensors; i++) {
    smoothedValues[i] = analogRead(flexPins[i]);
  }

  // Initialize MPU6050 sensors
  delay(100);  // Let MPUs power up

  mpuRightOk = initMPU(MPU_RIGHT);
  mpuLeftOk  = initMPU(MPU_LEFT);

  Serial.println("HandGlow v2.0 - Flex + Dual MPU6050 (Accel+Gyro)");
  Serial.print("MPU Right (0x69): ");
  Serial.println(mpuRightOk ? "OK" : "FAIL");
  Serial.print("MPU Left  (0x68): ");
  Serial.println(mpuLeftOk ? "OK" : "FAIL");

  // Print header row
  // Format: DATA,RT(D15),...,LP(D14),aX1,aY1,aZ1,gX1,gY1,gZ1,aX2,aY2,aZ2,gX2,gY2,gZ2
  Serial.print("DATA");
  for (int i = 0; i < numFlexSensors; i++) {
    Serial.print(",");
    Serial.print(fingerNames[i]);
    Serial.print("(D");
    Serial.print(flexPins[i]);
    Serial.print(")");
  }
  // Right hand IMU headers (MPU1 = Right = 0x69)
  Serial.print(",aX1,aY1,aZ1,gX1,gY1,gZ1");
  // Left hand IMU headers (MPU2 = Left = 0x68)
  Serial.print(",aX2,aY2,aZ2,gX2,gY2,gZ2");
  Serial.println();

  lastSampleTime = millis();
}

// ============================================================
// LOOP — runs at 25 Hz (40ms intervals)
// ============================================================
void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  // ── Read Flex Sensors ──
  Serial.print("DATA");
  for (int i = 0; i < numFlexSensors; i++) {
    int rawValue = analogRead(flexPins[i]);

    if (invertData) {
      rawValue = 4095 - rawValue;
    }

    // Apply EMA smoothing
    smoothedValues[i] = (alpha * rawValue) + ((1.0 - alpha) * smoothedValues[i]);

    Serial.print(",");
    Serial.print(fingerNames[i]);
    Serial.print(":");
    Serial.print((int)smoothedValues[i]);
  }

  // ── Read Right Hand IMU (MPU1 @ 0x69) ──
  IMUData imu1 = {0, 0, 0, 0, 0, 0, false};
  if (mpuRightOk) {
    imu1 = readIMU(MPU_RIGHT);
  }
  // Print right hand accel + gyro (nan if failed)
  Serial.print(",");
  if (imu1.ok) {
    Serial.print(imu1.ax, 3); Serial.print(",");
    Serial.print(imu1.ay, 3); Serial.print(",");
    Serial.print(imu1.az, 3); Serial.print(",");
    Serial.print(imu1.gx, 2); Serial.print(",");
    Serial.print(imu1.gy, 2); Serial.print(",");
    Serial.print(imu1.gz, 2);
  } else {
    Serial.print("nan,nan,nan,nan,nan,nan");
  }

  // ── Read Left Hand IMU (MPU2 @ 0x68) ──
  IMUData imu2 = {0, 0, 0, 0, 0, 0, false};
  if (mpuLeftOk) {
    imu2 = readIMU(MPU_LEFT);
  }
  // Print left hand accel + gyro (nan if failed)
  Serial.print(",");
  if (imu2.ok) {
    Serial.print(imu2.ax, 3); Serial.print(",");
    Serial.print(imu2.ay, 3); Serial.print(",");
    Serial.print(imu2.az, 3); Serial.print(",");
    Serial.print(imu2.gx, 2); Serial.print(",");
    Serial.print(imu2.gy, 2); Serial.print(",");
    Serial.print(imu2.gz, 2);
  } else {
    Serial.print("nan,nan,nan,nan,nan,nan");
  }

  Serial.println();
}