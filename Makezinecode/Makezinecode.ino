#include <avr/interrupt.h> // Use timer interrupt library

/******** Sine wave parameters ********/
#define PI2 6.283185 // 2*PI saves calculation later
#define AMP 127 // Scaling factor for sine wave
#define OFFSET 128 // Offset shifts wave to all >0 values

/******** Lookup table ********/
#define LENGTH 256 // Length of the wave lookup table
byte wave[LENGTH]; // Storage for waveform

unsigned long time = millis();
int bpm = 240; //number of beats per minute, where each beat is a quarter note
float speed_multiplier; //calculated using input from the tempo pot, used to determine speed of playback
float octave_adjuster; //calculated using input from the pitch pot, used to determine pitch of notes playback

void play_note(int beats, int input);
int little_bee_beats[] = {4,1,1,1,1,4,1,1,1,1,2,2,2,2,1,1,1,1,420};
int little_bee_notes[] = {40,47,255,47,255,45,53,255,60,53,47,45,40,40,40,40};
int lost_woods_beats[] = {/*start first line*/2,2,4,2,2,4,/*end first measure*/
                            2,2,2,2,4,2,2,/*end 2nd measure*/
                            2,2,8,2,2,2,8,/*end of first line*/
                            2,2,4,2,2,4,/*end of 1st measure 2nd line*/ 
                            2,2,2,2,4,2,2,2,2,8,2,2,2,8,/*end of 2nd line*/
                            2,2,4,2,2,4,/*end 1st measure*/
                            2,2,8,/*end 2nd measure*/
                            2,2,4,2,2,4, /*end 3rd measure*/
                            2,2,8,/*end 3rd line*/
                            2,2,4,2,2,4,/*end 1st measure*/
                            2,2,8,/*end 2nd measure*/
                            2,2,2,2,2,2,2,2,/*end 3rd measure*/
                            2,2,2,2,2,2,1,2,1,/*end 4th line*/
                            16,/*end 1st mearues*/
                            4,2,2,2,2,4,/*end last measure last line*/
                            420
                            };

int lost_woods_notes[] = {45, 36, 32, 45, 36, 32, /*end first measure*/
                          45, 36, 32, 24, 27, 32, 30, /*end 2nd measure*/
                          32, 40, 47, 53, 47, 40, 47,/*end of first line*/ 
                          45, 36, 32, 45, 36, 32,/*end of 1st measure 2nd line*/
                          45, 36, 32, 24, 27, 32, 
                          30, 24, 32, 40, 32, 40, 53, 47,/*end of 2nd line*/
                          53, 47, 45, 40, 36, 32,/*end 1st measure*/
                          30, 32, 47, /*end 2nd measure*/
                          45, 40, 36, 32, 30, 27,/*end 3rd measure*/
                          24, 22, 20,/*end 3rd line*/
                          53, 47, 45, 40, 36, 32,/*end 1st measure*/
                          30, 32, 47, /*end 2nd measure*/
                          45, 47, 36, 40, 32, 36, 30, 32, /*end 3rd measure*/
                          27, 30, 24, 27, 22, 24, 32, 30, 36, /*end 4th line*/
                          32, 455, 24, 24, 24, 24, 455
                          }; //67 notes long
                          
int little_bee_array_length = 18; //set to length of the song array
int lost_woods_array_length = 66; //set to length of the song array
int which_song = 1; //this controls what song to play. 0=little bee, 1=lost woods

//setup the analog read pin ints
int tempo_pin = 0;
int pitch_pin = 2;


void setup() {
  Serial.begin(9600);

/* Populate the waveform table with a sine wave */
for (int i=0; i<LENGTH; i++) { // Step across wave table
   float v = (AMP*sin((PI2/LENGTH)*i)); // Compute value
   wave[i] = int(v+OFFSET); // Store value as integer
 }

/****Set timer1 for 8-bit fast PWM output ****/
 pinMode(9, OUTPUT); // Make timer’s PWM pin an output
 TCCR1B = (1 << CS10); // Set prescaler to full 16MHz
 TCCR1A |= (1 << COM1A1); // Pin low when TCNT1=OCR1A
 TCCR1A |= (1 << WGM10); // Use 8-bit fast PWM mode
 TCCR1B |= (1 << WGM12);

/******** Set up timer2 to call ISR ********/
 TCCR2A = 0; // No options in control register A
 TCCR2B = (1 << CS21); // Set prescaler to divide by 8
 TIMSK2 = (1 << OCIE2A); // Call ISR when TCNT2 = OCRA2
 OCR2A = 71; // Set frequency of generated wave
 sei(); // Enable interrupts to generate waveform!
}
int i;

void loop() { 
  i = 0;
if (which_song == 0){
  while (little_bee_beats[i] != 420) {
    //serial read for both tempo and pitch here, will return a val from 0-1023
    int analog_tempo = analogRead(tempo_pin);
    int analog_pitch = analogRead(pitch_pin);
    //need to adjust those analog readings to be reasonable values to input to play_note
    //we want the analog_tempo to be somewhere between 0 and 4, so divide by 256.0
    play_note(little_bee_beats[i], little_bee_notes[i], (analog_tempo/256.0), (analog_pitch/512.0));
    i+=1;
  }
}
else {
  while (lost_woods_beats[i] != 420) {
    //serial read for both tempo and pitch here, will return a val from 0-1023
    int analog_tempo = analogRead(tempo_pin);
    int analog_pitch = analogRead(pitch_pin);
    //need to adjust those analog readings to be reasonable values to input to play_note
    //we want the analog_tempo to be somewhere between 0 and 4, so divide by 256.0
    play_note(lost_woods_beats[i], lost_woods_notes[i], (analog_tempo/256.0), (analog_pitch/512.0));
    i+=1;
  }
}
}

/* This function will play a given note for a given duration of time 
    int beats is how many 16th notes the note will play for
    int input is the OCR2A register value that will play a certain frequency
    float speed_multiplier is a multiplied adjustment to the tempo
    float octave_adjsuter is a multiplied adjustment to the frequency of each note
*/
void play_note(int beats, int input, float speed_multiplier, float octave_adjuster) {
  long duration = (1000L*60*beats/bpm)/speed_multiplier;
  Serial.print("duration = ");
  Serial.println(duration);
  int ocr2aval = input * octave_adjuster;
  OCR2A = ocr2aval;
  unsigned long start = millis();
  while (millis() - start <= duration) {
  }
  OCR2A = 455; //set this to play a very low note in between notes to simulate silence
}

/******** Called every time TCNT2 = OCR2A ********/
ISR(TIMER2_COMPA_vect) { // Called when TCNT2 == OCR2A
 static byte index=0; // Points to each table entry
 OCR1AL = wave[index++]; // Update the PWM output
 //asm(“NOP;NOP”); // Fine tuning -- using this causes error everytime
 TCNT2 = 6; // Timing to compensate for ISR run time
}
