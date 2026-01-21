/*
 Example ESP8266 endpoints for Plant Guard
 - GET /status -> {"water_level":0-100, "pump1_count":int, "pump2_count":int}
 - GET /spray?pump=1|2 -> triggers pump 1 or 2

 Adjust pins and logic to your hardware. This example assumes:
  - Pump1 on D6 (12V), Pump2 on D0 (6V)
  - Water level sensor on D5 (LOW means water present) and a simple mapping to percentage.
*/

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// WiFi
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

// Pins (match your ESP sketch)
#define WATER_PIN  D5
#define PUMP_12V   D6   // Pump1 (powdery)
#define PUMP_6V    D0   // Pump2 (rust)

ESP8266WebServer server(80);

volatile unsigned long pump1_count = 0;
volatile unsigned long pump2_count = 0;

int water_level_percent() {
  // Simple example: LOW means water present -> 100%; HIGH -> 0%
  // Replace with your ADC/ultrasonic logic if available
  int present = digitalRead(WATER_PIN) == LOW ? 1 : 0;
  return present ? 100 : 0;
}

void do_spray(uint8_t pump) {
  if (pump == 1) {
    digitalWrite(PUMP_12V, HIGH);
    delay(1500);
    digitalWrite(PUMP_12V, LOW);
    pump1_count++;
  } else if (pump == 2) {
    digitalWrite(PUMP_6V, HIGH);
    delay(1500);
    digitalWrite(PUMP_6V, LOW);
    pump2_count++;
  }
}

void handle_status() {
  char buf[128];
  snprintf(buf, sizeof(buf), "{\"water_level\":%d,\"pump1_count\":%lu,\"pump2_count\":%lu}",
           water_level_percent(), pump1_count, pump2_count);
  server.send(200, "application/json", buf);
}

void handle_spray() {
  if (!server.hasArg("pump")) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"pump arg missing\"}");
    return;
  }
  int pump = server.arg("pump").toInt();
  if (pump != 1 && pump != 2) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"pump must be 1 or 2\"}");
    return;
  }
  do_spray((uint8_t)pump);
  server.send(200, "application/json", "{\"ok\":true}");
}

void setup() {
  pinMode(WATER_PIN, INPUT_PULLUP);
  pinMode(PUMP_12V, OUTPUT);
  pinMode(PUMP_6V, OUTPUT);
  digitalWrite(PUMP_12V, LOW);
  digitalWrite(PUMP_6V, LOW);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.begin(115200);
  Serial.println();
  Serial.print("Connecting to "); Serial.println(ssid);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println();
  Serial.print("IP: "); Serial.println(WiFi.localIP());

  server.on("/status", HTTP_GET, handle_status);
  server.on("/spray", HTTP_GET, handle_spray);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
