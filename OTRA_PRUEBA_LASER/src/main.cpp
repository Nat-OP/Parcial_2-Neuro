#include <Arduino.h>
#include "Adafruit_VL53L0X.h"
#include <Wire.h>

// Dirección I2C del multiplexor PCA9548A
#define MUX_ADDR 0x70

// Objetos para los sensores (todos usarán la misma dirección 0x29 internamente)
Adafruit_VL53L0X lox = Adafruit_VL53L0X();

// Función para seleccionar el canal en el PCA9548A (0 a 7)
void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  Serial.println("Iniciando Sensores con Multiplexor...");

  // Inicializar cada sensor en su respectivo canal
  for (uint8_t i = 0; i < 4; i++) {
    tcaselect(i); // Abrir canal i
    if (!lox.begin()) {
      Serial.print("Fallo en canal "); Serial.println(i);
    } else {
      Serial.print("Sensor "); Serial.print(i); Serial.println(" OK");
    }
  }
}

void loop() {
  VL53L0X_RangingMeasurementData_t measure;

  for (uint8_t i = 0; i < 4; i++) {
    tcaselect(i); // Seleccionar canal
    lox.rangingTest(&measure, false); // Leer sensor del canal actual

    Serial.print("S"); Serial.print(i); Serial.print(": ");
    if (measure.RangeStatus != 4) {
      Serial.print(measure.RangeMilliMeter);
    } else {
      Serial.print("Out");
    }
    Serial.print("mm | ");
  }
  Serial.println();
  delay(100);
}
