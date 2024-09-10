#include <iostream>
#include <vector>
#include <chrono>
#include <random123/threefry.h>
#include <random123/philox.h>
#include <thread>  // For hardware concurrency
#include <cstdlib> // For abs()

#define NUM_SIMULATIONS 1000000  // 1 million simulations
#define POINTS_ON_BOARD 24       // Total number of points on the backgammon board
#define TOTAL_PIECES 15          // Each player has 15 pieces
#define HOME_START 18            // Home starts from point 19 to 24 (18 to 23 in zero-indexed)

struct Player {
    std::vector<int> pieces_on_points;  // Keeps track of how many pieces are on each point
    int pieces_off;                     // How many pieces the player has borne off the board
    int pieces_in_home;                 // Number of pieces in the home quadrant

    Player() : pieces_on_points(POINTS_ON_BOARD, 0), pieces_off(0), pieces_in_home(0) {}
};

using rng_t = r123::Philox4x32;  // Use random123's Philox generator
rng_t rng;

// Function to roll the dice
std::pair<int, int> roll_dice(rng_t::ctr_type &c, rng_t::key_type &k) {
    rng_t::ctr_type r = rng(c, k);
    int dice1 = (r[0] % 6) + 1;
    int dice2 = (r[1] % 6) + 1;
    return {dice1, dice2};  // Return both dice
}

// Move pieces based on dice roll
bool move_pieces(Player &player, Player &opponent, int dice_roll1, int dice_roll2) {
    bool moved = false;
    // Try to move using the first dice
    for (int i = 0; i < POINTS_ON_BOARD; ++i) {
        if (player.pieces_on_points[i] > 0) {
            int target_point = i + dice_roll1;
            if (target_point < POINTS_ON_BOARD) {
                // Move piece if the opponent has less than 2 pieces on target point
                if (opponent.pieces_on_points[target_point] < 2) {
                    // If opponent has one piece, hit it
                    if (opponent.pieces_on_points[target_point] == 1) {
                        opponent.pieces_on_points[target_point] = 0;  // Remove opponent's piece
                        opponent.pieces_on_points[0]++;               // Move it to the bar (0 point)
                    }
                    // Move player piece
                    player.pieces_on_points[i]--;
                    player.pieces_on_points[target_point]++;
                    moved = true;
                }
            } else if (target_point >= POINTS_ON_BOARD && i >= HOME_START) {
                // Bear off the piece if it's in the home quadrant
                player.pieces_on_points[i]--;
                player.pieces_off++;
                moved = true;
            }
        }
        if (moved) break;
    }

    // Try to move using the second dice
    moved = false;
    for (int i = 0; i < POINTS_ON_BOARD; ++i) {
        if (player.pieces_on_points[i] > 0) {
            int target_point = i + dice_roll2;
            if (target_point < POINTS_ON_BOARD) {
                if (opponent.pieces_on_points[target_point] < 2) {
                    if (opponent.pieces_on_points[target_point] == 1) {
                        opponent.pieces_on_points[target_point] = 0;
                        opponent.pieces_on_points[0]++;
                    }
                    player.pieces_on_points[i]--;
                    player.pieces_on_points[target_point]++;
                    moved = true;
                }
            } else if (target_point >= POINTS_ON_BOARD && i >= HOME_START) {
                player.pieces_on_points[i]--;
                player.pieces_off++;
                moved = true;
            }
        }
        if (moved) break;
    }

    return moved;  // Return true if at least one move was made
}

// Check if all pieces are borne off
bool all_pieces_borne_off(const Player &player) {
    return player.pieces_off == TOTAL_PIECES;
}

// Simulate one full game
int simulate_game(long thread_id) {
    Player player1, player2;

    // Set up initial board configuration (example setup)
    player1.pieces_on_points[0] = 2;
    player1.pieces_on_points[11] = 5;
    player1.pieces_on_points[16] = 3;
    player1.pieces_on_points[18] = 5;

    player2.pieces_on_points[23] = 2;
    player2.pieces_on_points[12] = 5;
    player2.pieces_on_points[7] = 3;
    player2.pieces_on_points[5] = 5;

    rng_t::ctr_type c = {{0}};
    rng_t::key_type k = {{static_cast<uint32_t>(thread_id)}};

    int player1_wins = 0;
    int player2_wins = 0;

    for (int game = 0; game < NUM_SIMULATIONS; ++game) {
        c[0] = game;  // Counter for random123

        while (!all_pieces_borne_off(player1) && !all_pieces_borne_off(player2)) {
            // Player 1's turn
            auto dice_roll = roll_dice(c, k);
            move_pieces(player1, player2, dice_roll.first, dice_roll.second);

            // Player 2's turn
            dice_roll = roll_dice(c, k);
            move_pieces(player2, player1, dice_roll.first, dice_roll.second);
        }

        // Check who won
        if (all_pieces_borne_off(player1)) {
            player1_wins++;
        } else {
            player2_wins++;
        }
    }

    return player1_wins - player2_wins;  // Return the difference in wins
}

void *run_simulations(void *threadid) {
    long tid = (long) threadid;
    int *result = new int(simulate_game(tid));
    pthread_exit((void *) result);
}

int main() {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;  // Fallback if hardware_concurrency fails
    }

    std::cout << "Using " << num_threads << " threads." << std::endl;

    pthread_t threads[num_threads];
    std::vector<int> thread_results(num_threads, 0);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (long t = 0; t < num_threads; ++t) {
        pthread_create(&threads[t], nullptr, run_simulations, (void *) t);
    }

    long long total_player1_wins = 0;
    long long total_player2_wins = 0;

    for (long t = 0; t < num_threads; ++t) {
        int *thread_result;
        pthread_join(threads[t], (void **) &thread_result);
        int result = *thread_result;
        if (result > 0) {
            total_player1_wins += result;
        } else {
            total_player2_wins -= result;
        }
        delete thread_result;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    long long total_games = NUM_SIMULATIONS * num_threads;
    double player1_win_percentage = (double)total_player1_wins / total_games * 100.0;
    double player2_win_percentage = (double)total_player2_wins / total_games * 100.0;

    std::cout << "Player 1 Win Percentage: " << player1_win_percentage << "%" << std::endl;
    std::cout << "Player 2 Win Percentage: " << player2_win_percentage << "%" << std::endl;
    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}