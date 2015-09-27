int sensorPin = A0;
int pwmPin = 9;
int val=0;

void setup() {
  pinMode(pwmPin,OUTPUT);
  Serial.begin(9600);
}
void loop() {
  for (int i=0; i<255; i++) {
    analogWrite(pwmPin,i);
    val = analogRead(sensorPin);
    Serial.println(val);
    delay(10);
  }
}

