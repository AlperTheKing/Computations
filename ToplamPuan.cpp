#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>
#include <climits>  // For INT_MAX

using namespace std;
using namespace std::chrono;

int min_max_total_score = INT_MAX;
vector<vector<int>> best_assignment(3, vector<int>(6));

// Function to calculate the total scores for each contestant
void calculate_total_scores(const vector<int>& jury1, const vector<int>& jury2, const vector<int>& jury3) {
    vector<int> total_scores(6);
    
    for (int i = 0; i < 6; i++) {
        total_scores[i] = jury1[i] + jury2[i] + jury3[i];
    }

    // Find the maximum score
    int max_score = *max_element(total_scores.begin(), total_scores.end());
    
    // Update the best assignment if we found a better solution
    if (max_score < min_max_total_score) {
        min_max_total_score = max_score;
        best_assignment = {jury1, jury2, jury3};
    }
}

// Main function to find the best score assignment
void find_best_score_assignment() {
    vector<int> scores = {1, 2, 3, 4, 5, 6};

    // Loop through all 6 contestants
    vector<int> jury1(6), jury2(6), jury3(6);
    for (int i = 0; i < 6; i++) {
        // Assign scores for Jury 1
        for (int j1 = 0; j1 < 6; j1++) {
            jury1[i] = scores[j1];
            
            // Assign scores for Jury 2 (excluding Jury 1's score)
            for (int j2 = 0; j2 < 6; j2++) {
                if (j2 == j1) continue; // Skip if Jury 2's score is the same as Jury 1's
                
                jury2[i] = scores[j2];

                // Assign scores for Jury 3 (excluding Jury 1's and Jury 2's scores)
                for (int j3 = 0; j3 < 6; j3++) {
                    if (j3 == j1 || j3 == j2) continue; // Skip if Jury 3's score matches Jury 1 or 2

                    jury3[i] = scores[j3];

                    // Calculate total scores
                    calculate_total_scores(jury1, jury2, jury3);
                }
            }
        }
    }
}

int main() {
    // Start time measurement
    auto start = high_resolution_clock::now();
    
    // Find the best score assignment
    find_best_score_assignment();

    // End time measurement
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    // Output the result
    cout << "Minimum of the maximum total score: " << min_max_total_score << endl;
    cout << "Jury 1 scores: ";
    for (int i : best_assignment[0]) cout << i << " ";
    cout << endl;
    cout << "Jury 2 scores: ";
    for (int i : best_assignment[1]) cout << i << " ";
    cout << endl;
    cout << "Jury 3 scores: ";
    for (int i : best_assignment[2]) cout << i << " ";
    cout << endl;
    
    // Output execution time
    cout << "Execution time: " << duration.count() << " microseconds" << endl;

    return 0;
}