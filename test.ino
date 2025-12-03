#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

#define DHTPIN 15
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// Soil Moisture Sensor
#define SOIL_PIN 34   // Analog pin

// CHANGE THESE
const char* WIFI_SSID = "POCO M6 Pro 5G";
const char* WIFI_PASSWORD = "11221111";

// IP of the laptop running Python server.py
const char* SERVER_IP = "10.53.110.213";
const int SERVER_PORT = 5000;

void setup() {
  Serial.begin(115200);

  dht.begin();

  Serial.println("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }

  Serial.println("\nConnected to WiFi!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  int soilRaw = analogRead(SOIL_PIN);
  int soilPercent = map(soilRaw, 4095, 1500, 0, 100);
  soilPercent = constrain(soilPercent, 0, 100);

  // JSON data
  String payload = "{";
  payload += "\"temperature\":" + String(temperature) + ",";
  payload += "\"humidity\":" + String(humidity) + ",";
  payload += "\"soil\":" + String(soilPercent);
  payload += "}";

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + "/data";

    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    int response = http.POST(payload);

    Serial.print("Server Response Code: ");
    Serial.println(response);

    http.end();
  }

  delay(2000);  // send every 2 seconds
}
