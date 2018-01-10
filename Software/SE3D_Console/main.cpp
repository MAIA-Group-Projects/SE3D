#include <iostream>
#include <stdlib.h>

#include "utility.h"

using namespace std;

int main(int argc, char* argv[])
{
    int key = 99;
    Utility u;

    u.printConsoleInstructions();
    cin >> key;

    while(key != 0) {
        if (key == 1) {
            system("cls");
            u.testRecorder();
        } else if (key == 2) {
            system("cls");
            u.printRegistrationInstructions();
            cin >> key;
            if (key == 1) {
                system("cls");
                string color1, color2, depth1, depth2;
                cout << "Provide filenames for color#1, color#2, depth#1, depth#2" << endl;
                cin >> color1 >> color2 >> depth1 >> depth2;
                u.testPairRegistration(color1, color2, depth1, depth2);
            } else if (key == 2) {
                u.testMultipleRegistration();
            }
        }
        system("cls");
        u.printConsoleInstructions();
        cin >> key;
    }

    return 0;
}
