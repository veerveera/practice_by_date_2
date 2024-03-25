#include <iostream>
#include <string> 
#include <fstream> 
#include <unordered_map> 
#include<algorithm>
using namespace std;
 
char tolower(char c) {
    if (c <= 'Z' && c >= 'A')
        return c - ('Z' - 'z');
    if (c <= 'Я' && c >= 'А')
        return c - ('Я' - 'я');
    return c;
}
 
string getWord(string mixedword) {
 string s = "";
    for (int i = 0; i < mixedword.size(); i++) {
        char t = tolower(mixedword[i]);
        if (t >= 'а' && t <= 'я' || t >= 'a' && t <= 'z') {
 s += t;
        }
    }
    return s;
}
 
int main() {
 setlocale(LC_ALL, "rus");
 ifstream file("data.txt");
 unordered_map<string, int> word_data;
 string word;
 string s = "";
    #pragma omp parallel for
    while (file >> word) {
    #pragma omp critical 
        {
            s = getWord(word);
            word_data[s]++;
            s = "";
        }
    }
    vector<pair<string, int>> sorted_words(word_data.begin(), word_data.end());
    sort(sorted_words.begin(), sorted_words.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
        return a.second > b.second;
        });
 
    for (const auto& pair : sorted_words) {
        cout << pair.first << ": " << pair.second << endl;
    }
    file.close();
 
    return 0;
}
