/*
 * Edge node: ESP32 + AS7265x (18-ch VIS-NIR) + SHT31 (temp/humidity) + white LED.
 * Pipeline: dark scan -> white ref (stored) -> leaf scans -> reference normalisation -> QC
 *           -> 3-pt moving average -> SNV -> band select -> int8 LDA -> softmax -> confidence gate
 *           -> MQTT (compact result, or raw spectrum for gateway second opinion).
 *
 * Libraries: SparkFun AS7265X, Adafruit SHT31, PubSubClient (Arduino Library Manager).
 * Put edge_model.h (from export_edge.py) next to this file.
 * NOT yet tested on hardware - check pins, LED driver and sensor gain/integration for your build.
 *
 * Serial commands:  'W' = capture and store white reference (place on white tile)
 */
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Preferences.h>
#include <SparkFun_AS7265X.h>
#include <Adafruit_SHT31.h>
#include "edge_model.h"

// ---------- user settings ----------
const char *WIFI_SSID = "YOUR_WIFI";
const char *WIFI_PASS = "YOUR_PASS";
const char *MQTT_HOST = "192.168.1.10";
const char *PLANT_ID  = "plant_001";
const int   LED_PIN   = 25;        // MOSFET gate for broadband white LED
const int   N_SCANS   = 5;         // repeated scans per leaf
const uint32_t SLEEP_S = 900;      // deep-sleep between sessions
// -----------------------------------

AS7265X sensor;
Adafruit_SHT31 sht;
WiFiClient wifi;
PubSubClient mqtt(wifi);
Preferences prefs;

float dark[N_CH], whiteRef[N_CH];

// Reads the 18 calibrated channels in wavelength order (410..940 nm) - matches WL_AS7265X.
void readChannels(float *o) {
  sensor.takeMeasurements();
  o[0]=sensor.getCalibratedA(); o[1]=sensor.getCalibratedB(); o[2]=sensor.getCalibratedC();
  o[3]=sensor.getCalibratedD(); o[4]=sensor.getCalibratedE(); o[5]=sensor.getCalibratedF();
  o[6]=sensor.getCalibratedG(); o[7]=sensor.getCalibratedH(); o[8]=sensor.getCalibratedR();
  o[9]=sensor.getCalibratedI(); o[10]=sensor.getCalibratedS(); o[11]=sensor.getCalibratedJ();
  o[12]=sensor.getCalibratedT(); o[13]=sensor.getCalibratedU(); o[14]=sensor.getCalibratedV();
  o[15]=sensor.getCalibratedW(); o[16]=sensor.getCalibratedK(); o[17]=sensor.getCalibratedL();
}

void averageScan(float *out, bool ledOn) {
  digitalWrite(LED_PIN, ledOn ? HIGH : LOW);
  delay(150);
  for (int i = 0; i < N_CH; i++) out[i] = 0;
  float tmp[N_CH];
  for (int s = 0; s < N_SCANS; s++) {
    readChannels(tmp);
    for (int i = 0; i < N_CH; i++) out[i] += tmp[i] / N_SCANS;
  }
  digitalWrite(LED_PIN, LOW);
}

void movingAverage3(const float *in, float *out) {          // edge replication, same as Python
  for (int i = 0; i < N_CH; i++) {
    float a = in[i > 0 ? i - 1 : 0], b = in[i], c = in[i < N_CH - 1 ? i + 1 : N_CH - 1];
    out[i] = (a + b + c) / 3.0f;
  }
}

void snv(float *x) {                                         // population std, same as Python
  float m = 0, v = 0;
  for (int i = 0; i < N_CH; i++) m += x[i];
  m /= N_CH;
  for (int i = 0; i < N_CH; i++) v += (x[i] - m) * (x[i] - m);
  float sd = sqrtf(v / N_CH) + 1e-9f;
  for (int i = 0; i < N_CH; i++) x[i] = (x[i] - m) / sd;
}

// int8 LDA + softmax. Returns predicted class, writes max probability to *conf.
int classify(const float *snvSpec, float temp, float hum, float *conf) {
  float f[N_FEAT];
  for (int i = 0; i < N_SEL; i++) f[i] = snvSpec[SEL_IDX[i]];
  if (N_CLIM >= 1) f[N_SEL] = temp;
  if (N_CLIM >= 2) f[N_SEL + 1] = hum;
  for (int j = 0; j < N_FEAT; j++) f[j] = (f[j] - F_MEAN[j]) * F_INV_STD[j];   // standardise (float)
  float logit[N_CLASSES], mx = -1e30f;
  for (int c = 0; c < N_CLASSES; c++) {
    float acc = 0;
    for (int j = 0; j < N_FEAT; j++) acc += (float)W_Q[c][j] * f[j];
    logit[c] = acc * W_SCALE[c] + BIAS[c];
    if (logit[c] > mx) mx = logit[c];
  }
  float sum = 0, p[N_CLASSES];
  for (int c = 0; c < N_CLASSES; c++) { p[c] = expf(logit[c] - mx); sum += p[c]; }
  int best = 0;
  for (int c = 0; c < N_CLASSES; c++) { p[c] /= sum; if (p[c] > p[best]) best = c; }
  *conf = p[best];
  return best;
}

void connectNet() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  mqtt.setServer(MQTT_HOST, 1883);
  mqtt.setBufferSize(1024);
  if (WiFi.status() == WL_CONNECTED) mqtt.connect(PLANT_ID);
}

void captureWhite() {
  float w[N_CH], d[N_CH];
  averageScan(d, false);
  averageScan(w, true);
  prefs.putBytes("dark", d, sizeof(d));
  prefs.putBytes("white", w, sizeof(w));
  Serial.println("white/dark reference stored");
}

void runSession() {
  // --- measure ---
  float raw[N_CH];
  averageScan(dark, false);                                  // dark scan each session
  averageScan(raw, true);                                    // leaf scan
  float temp = sht.readTemperature(), hum = sht.readHumidity();

  // --- reference normalisation + QC ---
  float refl[N_CH], sig = 0, darkMean = 0, whiteMean = 0;
  for (int i = 0; i < N_CH; i++) {
    float den = whiteRef[i] - dark[i];
    refl[i] = (raw[i] - dark[i]) / (den > 1e-6f ? den : 1e-6f);
    sig += den; darkMean += dark[i]; whiteMean += whiteRef[i];
  }
  bool qcOk = (sig / N_CH > 0.02f * whiteMean / N_CH) &&       // enough signal
              (darkMean < 0.15f * whiteMean) &&                // no light leak (poor seating)
              !isnan(temp) && !isnan(hum);
  int batt = 100;                                             // TODO: read battery ADC

  if (!qcOk) {                                                // ask for re-measurement
    if (mqtt.connected()) {
      char msg[96]; snprintf(msg, sizeof(msg), "{\"q\":0,\"batt\":%d}", batt);
      mqtt.publish((String("leaf/") + PLANT_ID + "/result").c_str(), msg);
    }
    return;
  }

  // --- smooth -> SNV -> classify (timed) ---
  uint32_t t0 = micros();
  float sm[N_CH]; movingAverage3(refl, sm); snv(sm);
  float conf; int cls = classify(sm, temp, hum, &conf);
  uint32_t us = micros() - t0;
  Serial.printf("class=%d conf=%.2f inference=%u us\n", cls, conf, us);   // latency for the paper

  // --- confidence gate + publish ---
  if (!mqtt.connected()) return;
  String base = String("leaf/") + PLANT_ID;
  char msg[768];
  if (conf >= CONF_THRESHOLD) {
    snprintf(msg, sizeof(msg), "{\"cls\":%d,\"p\":%.3f,\"batt\":%d,\"q\":1,\"temp\":%.1f,\"hum\":%.1f,\"us\":%u}",
             cls, conf, batt, temp, hum, us);
    mqtt.publish((base + "/result").c_str(), msg);
  } else {
    int n = snprintf(msg, sizeof(msg), "{\"spec\":[");       // raw reflectance goes to gateway
    for (int i = 0; i < N_CH; i++) n += snprintf(msg + n, sizeof(msg) - n, "%.4f%s", refl[i], i < N_CH - 1 ? "," : "");
    snprintf(msg + n, sizeof(msg) - n, "],\"temp\":%.1f,\"hum\":%.1f}", temp, hum);
    mqtt.publish((base + "/uncertain").c_str(), msg);
  }
  mqtt.loop(); delay(200);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT); digitalWrite(LED_PIN, LOW);
  Wire.begin();
  if (!sensor.begin()) { Serial.println("AS7265x not found"); while (1) delay(1000); }
  sensor.disableIndicator();
  sensor.setGain(AS7265X_GAIN_16X);
  sensor.setIntegrationCycles(49);
  if (!sht.begin(0x44)) Serial.println("SHT31 not found");

  prefs.begin("ref", false);
  bool haveRef = prefs.getBytesLength("white") == sizeof(whiteRef);
  if (haveRef) { prefs.getBytes("white", whiteRef, sizeof(whiteRef)); }
  Serial.println("Send 'W' within 3 s to capture white reference...");
  uint32_t t = millis();
  while (millis() - t < 3000) if (Serial.available() && Serial.read() == 'W') { captureWhite(); haveRef = false; break; }
  if (!haveRef) { prefs.getBytes("white", whiteRef, sizeof(whiteRef)); }

  connectNet();
  runSession();

  esp_sleep_enable_timer_wakeup((uint64_t)SLEEP_S * 1000000ULL);   // low-power duty cycle
  esp_deep_sleep_start();
}

void loop() {}
