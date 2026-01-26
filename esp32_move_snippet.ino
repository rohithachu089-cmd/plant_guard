/*
  Updated ESP32/ESP8266 endpoints for Plant Guard with Movement Control
  
  Add these to your existing sketch to handle the Stop-Scan-Move routine.
*/

// Add to your existing includes
#include <ESP8266WebServer.h> // Or <WebServer.h> for ESP32

// Movement Pins (Example)
#define MOTOR_PIN_1 D1 
#define MOTOR_PIN_2 D2

void handle_move() {
  if (!server.hasArg("cmd")) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"cmd arg missing\"}");
    return;
  }
  
  String cmd = server.arg("cmd");
  if (cmd == "start") {
    digitalWrite(MOTOR_PIN_1, HIGH);
    digitalWrite(MOTOR_PIN_2, LOW);
    Serial.println("Bot: Moving...");
  } else if (cmd == "stop") {
    digitalWrite(MOTOR_PIN_1, LOW);
    digitalWrite(MOTOR_PIN_2, LOW);
    Serial.println("Bot: Stopped for scanning.");
  }
  
  server.send(200, "application/json", "{\"ok\":true}");
}

// In setup() add:
// pinMode(MOTOR_PIN_1, OUTPUT);
// pinMode(MOTOR_PIN_2, OUTPUT);
// server.on("/move", HTTP_GET, handle_move);
