#include <ArduinoJson.h>
#include <WiFi.h>

//Motor pin connections
#define in1 12
#define in2 14
#define in3 27
#define in4 26

#define buzzer 18

#define led 19

//ir pin connection
#define irPin2 17
#define irPin1 16
#define irPin3 4
#define irPin4 5
int node_count = 0, msg_received = 0, flag = 0;

const char* ssid = "SG";
const char* password = "rugved1234";

WiFiServer server(80);

const int MAX_NUMBERS = 30;  // Adjust the size based on your needs
int receivedNumbers[MAX_NUMBERS];
int numberOfReceivedNumbers = 0;
const int MAX_STR_LENGTH = 3;  // Maximum length of the character array
char receivedStrings[MAX_NUMBERS][MAX_STR_LENGTH];
int numberOfReceivedStrings = 0;

char direction = 'U', dest_node;
int node_i = 0, dir_i = 0;
int event_nodes[] = { 1, 4, 10, 8, 15 };

void setup() {
  Serial.begin(115200);

  //Motor pins as output
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

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
  digitalWrite(led, LOW);

  WiFi.begin(ssid, password);
  Serial.println("Establishing connection to WiFi with SSID: " + String(ssid));

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.print("Connected to network with IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();
}

void get_data() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("New client connected");

    while (client.connected()) {
      if (client.available()) {
        msg_received++;
        // Read the JSON data
        DynamicJsonDocument jsonDoc(1024);  // Adjust the size based on your data
        DeserializationError error = deserializeJson(jsonDoc, client);

        if (error) {
          Serial.print("Error decoding JSON: ");
          Serial.println(error.c_str());
          break;
        }

        // Process the JSON data (assuming it's an array of numbers)
        if (jsonDoc.is<JsonArray>()) {
          JsonArray jsonArray = jsonDoc.as<JsonArray>();

          for (int i = 0; i < jsonArray[0].size(); i++) {
            int number = jsonArray[0][i].as<int>();

            // Check if there is enough space in the array
            if (numberOfReceivedNumbers < MAX_NUMBERS) {
              receivedNumbers[numberOfReceivedNumbers] = number;
              numberOfReceivedNumbers++;
            }
            Serial.println("Received numbers: " + String(number));
          }
          for (int i = 0; i < jsonArray[1].size(); i++) {
            const char* str = jsonArray[1][i].as<const char*>();

            // Check if there is enough space in the array
            if (numberOfReceivedStrings < MAX_NUMBERS) {
              strncpy(receivedStrings[numberOfReceivedStrings], str, MAX_STR_LENGTH - 1);
              receivedStrings[numberOfReceivedStrings][MAX_STR_LENGTH - 1] = '\0';  // Null-terminate the string
              numberOfReceivedStrings++;
            }

            // Process each string as needed
            Serial.println("Received string: " + String(str));
          }
        }

        // Send a response back to the Python client
        client.print("Data received successfully!");

        // Close the connection
        client.stop();
        Serial.println("Client disconnected");
      }
    }
  }
  if (numberOfReceivedNumbers > 0) {
    Serial.println("Received numbers array:");
    for (int i = 0; i < numberOfReceivedNumbers; i++) {
      Serial.println(receivedNumbers[i]);
    }
    Serial.println("Received strings array:");
    for (int i = 0; i < numberOfReceivedStrings; i++) {
      Serial.println(receivedStrings[i][0]);
    }
  }
}

void line_following() {
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
    // delay(100);
    // digitalWrite(in3, LOW);
    // digitalWrite(in4, LOW);
    // digitalWrite(in1, LOW);
    // digitalWrite(in2, LOW);

    // digitalWrite(buzzer, LOW);
    // delay(500);
    // digitalWrite(buzzer, HIGH);
    node_count++;
    //Right
    // digitalWrite(in1, LOW);
    // digitalWrite(in2, LOW);
    // digitalWrite(in3, HIGH);
    // digitalWrite(in4, LOW);
    // delay(40);
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    delay(400);
  }
}

void left_turn() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(150);
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  delay(450);
  while (!digitalRead(irPin1) && !digitalRead(irPin2)) {
    flag = 1;
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
  }
}

void right_turn() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(150);
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(500);
  // bool x = true;
  while (!digitalRead(irPin2) && !digitalRead(irPin1)) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    // if () {
    //   x = false;
    // }
  }
  // if (digitalRead(irPin2)) {
  //   flag = 1;
  // }
  // else {
  //   while (!digitalRead(irPin2)) {
  //   digitalWrite(in1, HIGH);
  //   digitalWrite(in2, LOW);
  //   digitalWrite(in3, HIGH);
  //   digitalWrite(in4, LOW);
  // }
  // }
}

void u_turn() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(150);
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(1220);
  // while (!digitalRead(irPin1)) {
  //   digitalWrite(in1, HIGH);
  //   digitalWrite(in2, LOW);
  //   digitalWrite(in3, HIGH);
  //   digitalWrite(in4, LOW);
  // }
}

void loop() {
  if (msg_received == 0) {
    get_data();
  } else if (msg_received > 0) {
    if (node_i == 0) {
      delay(10000);
    }
    if (node_i >= numberOfReceivedNumbers) {
      char next_direction = 'D';
      if (direction == 'L' && next_direction == 'D') {
        left_turn();
      } else if (direction == 'R' && next_direction == 'D') {
        right_turn();
      }
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
      digitalWrite(buzzer, LOW);
      delay(5000);
      digitalWrite(buzzer, HIGH);
      delay(30000);
    }
    dest_node = receivedNumbers[node_i];

    if ((dest_node == 1 || dest_node == 4 || dest_node == 10 || dest_node == 8 || dest_node == 15)) {
      if (receivedStrings[dir_i][0] == 'S') {
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        digitalWrite(in3, LOW);
        digitalWrite(in4, LOW);
        digitalWrite(buzzer, LOW);
        delay(1000);
        digitalWrite(buzzer, HIGH);
      } else if ((direction == 'U' && receivedStrings[dir_i][0] == 'L') || (direction == 'R' && receivedStrings[dir_i][0] == 'U') || (direction == 'L' && receivedStrings[dir_i][0] == 'D') || (direction == 'D' && receivedStrings[dir_i][0] == 'R')) {
        left_turn();
        direction = receivedStrings[dir_i][0];
      } else if ((direction == 'U' && receivedStrings[dir_i][0] == 'R') || (direction == 'R' && receivedStrings[dir_i][0] == 'D') || (direction == 'L' && receivedStrings[dir_i][0] == 'U') || (direction == 'D' && receivedStrings[dir_i][0] == 'L')) {
        right_turn();
        direction = receivedStrings[dir_i][0];
      } else if ((direction == 'U' && receivedStrings[dir_i][0] == 'D') || (direction == 'R' && receivedStrings[dir_i][0] == 'L') || (direction == 'L' && receivedStrings[dir_i][0] == 'R') || (direction == 'D' && receivedStrings[dir_i][0] == 'U')) {
        u_turn();
        direction = receivedStrings[dir_i][0];
      }
      if (dest_node == 1) {
        if (receivedNumbers[node_i - 1] == 0) {
          unsigned long start = millis();
          while (millis() - start <= 1400) {
            line_following();
          }
        } else if (receivedNumbers[node_i - 1] == 2) {
          unsigned long start = millis();
          while (millis() - start <= 3000) {
            line_following();
          }
        }
      } else if (dest_node == 8) {
        if (receivedNumbers[node_i - 1] == 7) {
          unsigned long start = millis();
          while (millis() - start <= 1600) {
            line_following();
          }
        } else if (receivedNumbers[node_i - 1] == 9) {
          unsigned long start = millis();
          while (millis() - start <= 2400) {
            line_following();
          }
        }
      } else if (dest_node == 10) {
        if (receivedNumbers[node_i - 1] == 9) {
          unsigned long start = millis();
          while (millis() - start <= 2600) {
            line_following();
          }
        } else if (receivedNumbers[node_i - 1] == 11) {
          unsigned long start = millis();
          while (millis() - start <= 1800) {
            line_following();
          }
        }
      } else if (dest_node == 4) {
        if (receivedNumbers[node_i - 1] == 3) {
          unsigned long start = millis();
          while (millis() - start <= 1500) {
            line_following();
          }
        } else if (receivedNumbers[node_i - 1] == 5) {
          unsigned long start = millis();
          while (millis() - start <= 2400) {
            line_following();
          }
        }
      } else if (dest_node == 15) {
        if (receivedNumbers[node_i - 1] == 14) {
          unsigned long start = millis();
          while (millis() - start <= 5000) {
            line_following();
          }
          direction = 'R';
        } else if (receivedNumbers[node_i - 1] == 12) {
          unsigned long start = millis();
          while (millis() - start <= 9500) {
            line_following();
          }
          direction = 'L';
        }
      }
    }

    else {
      if (direction == receivedStrings[dir_i][0]) {
        while (node_count == 0) {
          line_following();
        }
        if (dest_node == 3 && receivedNumbers[node_i - 1] == 2) {
          direction = 'U';
        } else if (dest_node == 2 && receivedNumbers[node_i - 1] == 3) {
          direction = 'L';
        }
      } else if ((direction == 'U' && receivedStrings[dir_i][0] == 'L') || (direction == 'R' && receivedStrings[dir_i][0] == 'U') || (direction == 'L' && receivedStrings[dir_i][0] == 'D') || (direction == 'D' && receivedStrings[dir_i][0] == 'R')) {
        //while (flag == 0) {
        left_turn();
        flag = 0;
        //}
        while (node_count == 0) {
          line_following();
        }
        direction = receivedStrings[dir_i][0];
      } else if ((direction == 'U' && receivedStrings[dir_i][0] == 'R') || (direction == 'R' && receivedStrings[dir_i][0] == 'D') || (direction == 'L' && receivedStrings[dir_i][0] == 'U') || (direction == 'D' && receivedStrings[dir_i][0] == 'L')) {
        right_turn();
        while (node_count == 0) {
          line_following();
        }
        direction = receivedStrings[dir_i][0];
      } else if ((direction == 'U' && receivedStrings[dir_i][0] == 'D') || (direction == 'R' && receivedStrings[dir_i][0] == 'L') || (direction == 'L' && receivedStrings[dir_i][0] == 'R') || (direction == 'D' && receivedStrings[dir_i][0] == 'U')) {
        u_turn();
        while (node_count == 0) {
          line_following();
        }
        if ((dest_node == 14 && receivedNumbers[node_i - 1] == 15) || (dest_node == 12 && receivedNumbers[node_i - 1] == 15)) {
          direction = 'D';
        } else {
          direction = receivedStrings[dir_i][0];
        }
      }
    }
    node_count = 0;
    node_i++;
    dir_i++;
  }
}