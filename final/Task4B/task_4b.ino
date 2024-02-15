//Motor pin connections
#define in1 10
#define in2 11
#define in3 12
#define in4 13

#define buzzer 8

#define led 7

//ir pin connection
#define irPin2 6
#define irPin1 5
#define irPin3 3
#define irPin4 4
int node_count = 0;


void setup() {
  Serial.begin(115200);

  //Motor pins as output
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // pinMode(enB, OUTPUT);
  //sensor pin as input
  pinMode(irPin1, INPUT);
  pinMode(irPin2, INPUT);
  pinMode(irPin3, INPUT);
  pinMode(irPin4, INPUT);
  //led output
  pinMode(led, OUTPUT);
  //buzzer output
  pinMode(buzzer, OUTPUT);
  //Keeping all motors off initially
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  // Initially off
  digitalWrite(buzzer, HIGH);  // Negative logic Buzzer
   digitalWrite(led, HIGH);
   delay(4000);
   digitalWrite(led, LOW);
  // delay(4000);
}

//unsigned long StartTime = 0;

void loop() {
  if (!digitalRead(irPin3) && digitalRead(irPin4) && !digitalRead(irPin1) && !digitalRead(irPin2)) {
    //Right
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
  }
  if (digitalRead(irPin3) && !digitalRead(irPin4) && !digitalRead(irPin1) && !digitalRead(irPin2)) {
    //Left
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
  }
  if (!digitalRead(irPin1) && digitalRead(irPin2)) {
    //Left
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
  } else if (!digitalRead(irPin1) && !digitalRead(irPin2)) {
    //Forward
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
  } else if (digitalRead(irPin1) && !digitalRead(irPin2)) {
    //Right
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
  } else if (digitalRead(irPin1) && digitalRead(irPin2)) {
    delay(100);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);

    digitalWrite(buzzer, LOW);
    //digitalWrite(led, HIGH);
    delay(500);
    digitalWrite(buzzer, HIGH);
    node_count++;
    //Right
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    delay(70);
    if (node_count == 1) {
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
      delay(100);
    }
    if (node_count == 3 || node_count == 5 || node_count == 6 || node_count == 8) {
      // RIGHT TURN FROM NODE
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(150);
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(300);
      while (!digitalRead(irPin1) && !digitalRead(irPin2)) {
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        digitalWrite(in3, HIGH);
        digitalWrite(in4, LOW);
      }
    }
    if (node_count == 7) {
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
      delay(150);
    }
    if (node_count == 4 || node_count == 10) {
      //LEFT TURN FROM NODE
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(150);
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
      delay(300);
      while (!digitalRead(irPin1)) {
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        digitalWrite(in3, LOW);
        digitalWrite(in4, HIGH);
      }
    }
    if (node_count == 11) {
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(500);
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(200);
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(500);
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(270);
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      delay(200);
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);

      digitalWrite(led, HIGH);
      delay(6000);
      digitalWrite(led, LOW);
    }

    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    delay(250);
  }
}