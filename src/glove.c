// Author: Haozhe An (haozhe@umd.edu)
// Implementation of DD-GloVe, a train-time debiasing algorithm to learn GloVe 
// word embeddings by leveraging dictionary definitions.
// Our work is to appear in Findings of ACL 2022.
// 
// The code is adapted from the original release of GloVe.
// ========== Begin of original message from the GloVe authors ==========
//  GloVe: Global Vectors for Word Representation
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/
// ========== End of original message from the GloVe authors ==========

#define _CRT_SECURE_NO_WARNINGS

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// windows pthread.h is buggy, but this #define fixes it
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>

#include "common.h"

#define _FILE_OFFSET_BITS 64

int write_header = 0; //0=no, 1=yes; writes vocab_size/vector_size as first line for use with some libraries, such as gensim.
int verbose = 2; // 0, 1, or 2
int seed = 0;
int use_unk_vec = 1; // 0 or 1
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 0; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
int checkpoint_every = 0; // checkpoint the model for every checkpoint_every iterations. Do nothing if checkpoint_every <= 0
int load_init_param = 0; // if 1 initial paramters are loaded from -init-param-file
int save_init_param = 0; // if 1 initial paramters are saved (i.e., in the 0 checkpoint)
int load_init_gradsq = 0; // if 1 initial squared gradients are loaded from -init-gradsq-file
real eta = 0.05; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real grad_clip_value = 100.0; // Clipping parameter for gradient components. Values will be clipped to [-grad_clip_value, grad_clip_value] interval.
real *W, *gradsq, *cost, *def_cost, *ortho_cost, *proj_cost, *context_ortho_cost;
long long num_lines, *lines_per_thread, vocab_size;
char vocab_file[MAX_STRING_LENGTH];
char input_file[MAX_STRING_LENGTH];
char save_W_file[MAX_STRING_LENGTH];
char save_gradsq_file[MAX_STRING_LENGTH];
char init_param_file[MAX_STRING_LENGTH];
char init_gradsq_file[MAX_STRING_LENGTH];

char definition_file[MAX_STRING_LENGTH];
int **word_to_def;
real lambda = 0.001, beta = 0.001, gamma_ = 0.001; // Hyper-param to balance weights
int def_word_num, use_def_loss = 1, use_ortho_loss = 1, use_proj_loss = 1; // Indicate if using the loss or not
int SEED_WORD_1 = 19, SEED_WORD_2 = 43; // Hard-coded indices for he and she in our vocab
int seed_1_top_num = 30, seed_2_top_num = 30;
long long cap = 30000; // Must be smaller than vocab size

// For bias-aware weight schedule and finding bias direction from initial seeds
real *bias_word_value, *bias_def_value, *context_bias_word_value, *context_bias_def_value;
real *bias_word_value_temp, *bias_def_value_temp, *context_bias_word_value_temp, *context_bias_def_value_temp;
long long *boy_candidate_words, *girl_candidate_words;
int girl_added = 0, boy_added = 0;

/**
 * Loads a save file for use as the initial values for the parameters or gradsq
 * Return value: 0 if success, -1 if fail
 */
int load_init_file(char *file_name, real *array, long long array_size) {
    FILE *fin;
    long long a;
    fin = fopen(file_name, "rb");
    if (fin == NULL) {
        log_file_loading_error("init file", file_name);
        return -1;
    }
    for (a = 0; a < array_size; a++) {
        if (feof(fin)) {
            fprintf(stderr, "EOF reached before data fully loaded in %s.\n", file_name);
            fclose(fin);
            return -1;
        }
        fread(&array[a], sizeof(real), 1, fin);
    }
    fclose(fin);
    return 0;
}

void initialize_parameters() {
    // TODO: return an error code when an error occurs, clean up in the calling routine
    if (seed == 0) {
        seed = time(0);
    }
    fprintf(stderr, "Using random seed %d\n", seed);
    srand(seed);
    long long a;
    long long W_size = 2 * vocab_size * (vector_size + 1); // +1 to allocate space for bias

    /* Allocate space for word vectors and context word vectors, and correspodning gradsq */
    a = posix_memalign((void **)&W, 128, W_size * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, W_size * sizeof(real)); // Might perform better than malloc
    if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        free(W);
        exit(1);
    }
    if (load_init_param) {
        // Load existing parameters
        fprintf(stderr, "\nLoading initial parameters from %s \n", init_param_file);
        if (load_init_file(init_param_file, W, W_size)) {
            free(W);
            free(gradsq);
            exit(1);
        }
    } else {
        // Initialize new parameters
        for (a = 0; a < W_size; ++a) {
            W[a] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
        }
    }

    if (load_init_gradsq) {
        // Load existing squared gradients
        fprintf(stderr, "\nLoading initial squared gradients from %s \n", init_gradsq_file);
        if (load_init_file(init_gradsq_file, gradsq, W_size)) {
            free(W);
            free(gradsq);
            exit(1);
        }
    } else {
        // Initialize new squared gradients
        for (a = 0; a < W_size; ++a) {
            gradsq[a] = 1.0; // So initial value of eta is equal to initial learning rate
        }
    }
}

inline real check_nan(real update) {
    if (isnan(update) || isinf(update)) {
        fprintf(stderr,"\ncaught NaN in update");
        return 0.;
    } else {
        return update;
    }
}

int compare( const void* a, const void* b) {
    // usage: qsort( a, 6, sizeof(real), compare ) a is array; 6 is array length
    real real_a = * ( (real*) a );
    real real_b = * ( (real*) b );

    // an easy expression for comparing
    return (real_a > real_b) - (real_a < real_b);
}

/* Train the GloVe model */
void *glove_thread(void *vid) {
    long long a, b ,l1, l2;
    long long id = *(long long*)vid;
    CREC cr;
    real diff, fdiff, temp1, temp2, diff_def, diff_ortho, diff_proj;
    FILE *fin;

    fin = fopen(input_file, "rb");
    if (fin == NULL) {
        // TODO: exit all the threads or somehow mark that glove failed
        log_file_loading_error("input file", input_file);
        pthread_exit(NULL);
    }
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET); //Threads spaced roughly equally throughout file
    cost[id] = 0, def_cost[id] = 0, ortho_cost[id] = 0, proj_cost[id] = 0, context_ortho_cost[id] = 0;

    real* W_updates1 = (real*)malloc(vector_size * sizeof(real));
    if (NULL == W_updates1){
        fclose(fin);
        pthread_exit(NULL);
    }
    real* W_updates2 = (real*)malloc(vector_size * sizeof(real));
    if (NULL == W_updates2){
        fclose(fin);
        free(W_updates1);
        pthread_exit(NULL);
    }

    long long fweight_increased = 0, fweight_decreased = 0;
    real fweight_total = 0, fweight_max = -1, fweight_min = 1;
    for (a = 0; a < lines_per_thread[id]; a++) {
        fread(&cr, sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        if (cr.word1 < 1 || cr.word2 < 1) { continue; }

        /* Get location of words in W & gradsq */
        l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words

        /* Calculate cost, save diff for gradients */
        diff = 0;
        for (b = 0; b < vector_size; b++) diff += W[b + l1] * W[b + l2]; // dot product of word and context word vector
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val); // add separate bias for each word
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff

        // Bias-aware co-occurrence weight schedule
        real word1_bias_difference = bias_word_value[cr.word1 - 1LL] - bias_def_value[cr.word1 - 1LL];
        real word2_bias_difference = context_bias_word_value[cr.word2 - 1LL] - context_bias_def_value[cr.word2 - 1LL];
        real fweight = 1, difference_thresh = 0.08;

        if (fabs(word1_bias_difference) > difference_thresh || fabs(word2_bias_difference) > difference_thresh) {
            real sign_word_bias_1 = bias_word_value[cr.word1 - 1LL] > 0 ? 1 : -1;
            real sign_word_bias_2 = context_bias_word_value[cr.word2 - 1LL] > 0 ? 1 : -1;
            real max_difference = fabs(word1_bias_difference) > fabs(word2_bias_difference) ? fabs(word1_bias_difference) : fabs(word2_bias_difference);
            fweight = 1 - sign_word_bias_1 * sign_word_bias_2 * max_difference * 0.4;
        }

        fdiff *= fweight;
        if (fweight > 1) fweight_increased++;
        if (fweight < 1) fweight_decreased++;
        fweight_total += fweight;
        if (fweight > fweight_max) fweight_max = fweight;
        if (fweight < fweight_min) fweight_min = fweight;

        // Check for NaN and inf() in the diffs.
        if (isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)) {
            fprintf(stderr,"Caught NaN in diff for kdiff for thread. Skipping update\n");
            fprintf(stderr, "l1 %lld, l2 %lld\n", l1, l2);
            continue;
        }
        cost[id] += 0.5 * fdiff * diff; // weighted squared error

        /* Adaptive gradient updates*/
        fdiff *= eta;
        for(b = 0; b < vector_size; b++) {
            // learning rate times gradient for word vectors
            temp1 = fdiff * W[b + l2];
            temp2 = fdiff * W[b + l1];
            // adaptive updates
            W[b + l1] -= temp1 / sqrt(gradsq[b + l1]);
            W[b + l2] -= temp2 / sqrt(gradsq[b + l2]);
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;

            // Clip values to avoid numerical difficulties
            if (W[b + l1] > 1) W[b + l1] = 1;
            if (W[b + l1] < -1) W[b + l1] = -1;
            if (W[b + l2] > 1) W[b + l2] = 1;
            if (W[b + l2] < -1) W[b + l2] = -1;
        }

        // updates for bias terms
        W[vector_size + l1] -= fdiff / sqrt(gradsq[vector_size + l1]);
        W[vector_size + l2] -= fdiff / sqrt(gradsq[vector_size + l2]);
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
    } // Finish original GloVe cost updates

    if (id == 0) {
        fprintf(stderr, "%d girls words (id): ", girl_added);
        for (int c = 0; c < girl_added; c++) {
            fprintf(stderr, "%lld, ", girl_candidate_words[c]);
        }
        fprintf(stderr, "\n");

        fprintf(stderr, "%d boys words (id): ", boy_added);
        for (int c = 0; c < boy_added; c++) {
            fprintf(stderr, "%lld, ", boy_candidate_words[c]);
        }
        fprintf(stderr, "\n");
    }

    real debias_direction[vector_size], context_debias_direction[vector_size];
    for (int c = 0; c < vector_size; c++) {
        debias_direction[c] = 0;
        context_debias_direction[c] = 0;

        // Use candidates words to approximate bias direction
        for (int d = 0; d < boy_added; d++) {
            debias_direction[c] += W[c + boy_candidate_words[d] * (vector_size + 1)] / (boy_added + 1);
            context_debias_direction[c] += W[c + (boy_candidate_words[d] + vocab_size) * (vector_size + 1)] / (boy_added + 1);
        }
        debias_direction[c] += W[c + SEED_WORD_1 * (vector_size + 1)] / (boy_added + 1);
        context_debias_direction[c] += W[c + (SEED_WORD_1 + vocab_size) * (vector_size + 1)] / (boy_added + 1);
        for (int d = 0; d < girl_added; d++) {
            debias_direction[c] -= W[c + girl_candidate_words[d] * (vector_size + 1)] / (girl_added + 1);
            context_debias_direction[c] -= W[c + (girl_candidate_words[d] + vocab_size) * (vector_size + 1)] / (girl_added + 1);
        }
        debias_direction[c] -= W[c + SEED_WORD_2 * (vector_size + 1)] / (girl_added + 1);
        context_debias_direction[c] -= W[c + (SEED_WORD_2 + vocab_size) * (vector_size + 1)] / (girl_added + 1);

    }

    // other loss updates were here
    def_cost[id] = 0;
    ortho_cost[id] = 0;
    context_ortho_cost[id] = 0;
    proj_cost[id] = 0;

    real def_loss = 0, ortho_loss = 0, context_ortho_loss = 0, proj_loss = 0;
    real s_dot_s = 0, w_dot_s = 0, s_dot_g = 0, g_dot_g = 0, w_dot_g = 0;
    real proj_w_s[vector_size], proj_s_g[vector_size], proj_w_g[vector_size];  // projection of w onto s etc.
    real proj_difference[vector_size], context_proj_difference[vector_size];
    real context_s_dot_s = 0, context_w_dot_s = 0, context_s_dot_g = 0, context_g_dot_g = 0, context_w_dot_g = 0;
    real context_proj_w_s[vector_size], context_proj_s_g[vector_size], context_proj_w_g[vector_size];  // projection of w onto s etc.
    real temp;

    // Store the products for computing gradients later
    real debias_direction_products[vector_size*vector_size], context_debias_direction_products[vector_size*vector_size];
    for (b = 0; b < vector_size; b++) {
        for (int j = 0; j < vector_size; j++) {
            debias_direction_products[b * vector_size + j] = debias_direction[b] * debias_direction[j];
            context_debias_direction_products[b * vector_size + j] = context_debias_direction[b] * context_debias_direction[j];
        }
    }


    // Optimize additional losses
    for (long long word_id = 0; word_id < vocab_size; word_id++) {
        l1 = word_id * (vector_size + 1);
        l2 = (word_id + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words

        // content_len indicates the number of words in the definition
        int content_len = word_to_def[word_id][0];
        int context_content_len = word_to_def[word_id][0];

        // Begin computing other losses and update accordingly
        real def_content_sum[vector_size], context_def_content_sum[vector_size];

        // Initialize all values to zeros
        for (b = 0; b < vector_size; b++) {
            def_content_sum[b] = 0;
            context_def_content_sum[b] = 0;
        }

        // Getting s(w) for focal word if it has definitions
        for (int l = 0; l < content_len; l++) {
            long long content_id = word_to_def[word_id][l+1] * (vector_size + 1);
            for (b = 0; b < vector_size; b++) {
                def_content_sum[b] += W[b + content_id] / content_len;
            }
        }

        // Getting s(w) for context word if it has definitions
        for (int l = 0; l < context_content_len; l++) {
            long long context_content_id = (word_to_def[word_id][l+1] + vocab_size) * (vector_size + 1);
            for (b = 0; b < vector_size; b++) {
                context_def_content_sum[b] += W[b + context_content_id]/ context_content_len;
            }
        }

        def_loss = 0; ortho_loss = 0; context_ortho_loss = 0; proj_loss = 0;
        s_dot_s = 0; w_dot_s = 0; s_dot_g = 0; g_dot_g = 0; w_dot_g = 0;
        context_s_dot_s = 0; context_w_dot_s = 0; context_s_dot_g = 0; context_g_dot_g = 0; context_w_dot_g = 0;

        if (content_len > 0) {
            for (b = 0; b < vector_size; b++) {
                temp = (W[b + l1] - def_content_sum[b]) > 0 ? (W[b + l1] - def_content_sum[b]) : -(W[b + l1] - def_content_sum[b]);
                if (!isnan(temp) && !isinf(temp)) {
                    def_loss += temp;
                    // Update for definition loss
                    if (use_def_loss) {
                        diff_def = (W[b + l1] - def_content_sum[b]) > 0 ? 1 : -1;
                        W[b + l1] -= diff_def * eta * lambda / sqrt(gradsq[b + l1]);
                    }
                } else {
                    fprintf(stderr, "Caught nan or inf for definition loss update l1.\n");
                    continue;
                }

                s_dot_s += def_content_sum[b] * def_content_sum[b];
                w_dot_s +=  W[b + l1] * def_content_sum[b];
                s_dot_g += def_content_sum[b] * debias_direction[b];
                g_dot_g += debias_direction[b] * debias_direction[b];
                w_dot_g += W[b + l1] * debias_direction[b];
            }

            for (b = 0; b < vector_size; b++) {
                proj_w_s[b] = W[b + l1] - w_dot_s / s_dot_s * def_content_sum[b];
                if (!isnan(proj_w_s[b]) && !isinf(proj_w_s[b])) {
                    ortho_loss += proj_w_s[b] * W[b + l1];
                } else {
                    fprintf(stderr, "Caught nan or inf for ortho loss l1.\n");
                    continue;
                }

                proj_w_g[b] = w_dot_g / g_dot_g * debias_direction[b];
                proj_s_g[b] = s_dot_g / g_dot_g * debias_direction[b];
                proj_difference[b] = (proj_w_g[b] - proj_s_g[b]) > 0 ? g_dot_g : -g_dot_g;

            }

            for (b = 0; b < vector_size; b++) {
                temp = (proj_w_g[b] - proj_s_g[b]) > 0 ? (proj_w_g[b] - proj_s_g[b]) : -((proj_w_g[b] - proj_s_g[b]));
                if (!isnan(temp) && !isinf(temp)) {
                    proj_loss += temp;
                    // Update for projection loss
                    if (use_proj_loss) {
                        diff_proj = 0;
                        for (int j = 0; j < vector_size; j++) {
                            diff_proj += debias_direction_products[b * vector_size + j] / proj_difference[j];
                        }

                        W[b + l1] -= diff_proj * eta * gamma_ / sqrt(gradsq[b + l1]);
                        if (isnan(W[b + l1]) || isinf(W[b + l1])) {
                            fprintf(stderr, "Projection loss update for l1 caught nan or inf\n");
                            continue;
                        }
                        if (W[b + l1] > 1) W[b + l1] = 1;
                        if (W[b + l1] < -1) W[b + l1] = -1;
                    }
                } else {
                    fprintf(stderr, "Caught nan or inf for projection loss update l1.\n");
                    continue;
                }
            }

            // Update for ortho loss
            if (use_ortho_loss && content_len > 0) {
                for (b = 0; b < vector_size; b++) {
                    diff_ortho = 2 * ortho_loss * proj_w_s[b];
                    if (!isnan(diff_ortho) && !isinf(diff_ortho)) {
                        W[b + l1] -= diff_ortho * eta * beta / sqrt(gradsq[b + l1]);
                    } else {
                        fprintf(stderr, "Caught nan or inf for ortho loss update l1.\n");
                        continue;
                    }
                }
            }
            def_cost[id] += def_loss;
            ortho_cost[id] += pow(ortho_loss, 2);
            proj_cost[id] += proj_loss;
        }
        else {
            // No definiton, update for proj_loss
            if (use_proj_loss) {
                for (b = 0; b < vector_size; b++) {
                    g_dot_g += debias_direction[b] * debias_direction[b];
                    w_dot_g += W[b + l1] * debias_direction[b];
                }

                for (b = 0; b < vector_size; b++) {
                    proj_w_g[b] = w_dot_g / g_dot_g * debias_direction[b];
                    proj_w_g[b] = proj_w_g[b] > 0 ? g_dot_g : - g_dot_g;
                }


                for (b = 0; b < vector_size; b++) {
                    diff_proj = 0;
                    for (int j = 0; j < vector_size; j++) {
                        diff_proj += debias_direction_products[b * vector_size + j] / proj_w_g[j];
                    }
                    W[b + l1] -= diff_proj * eta * gamma_ / sqrt(gradsq[b + l1]);
                    if (W[b + l1] > 1) W[b + l1] = 1;
                    if (W[b + l1] < -1) W[b + l1] = -1;
                }
            }

            def_cost[id] += def_loss;
            ortho_cost[id] += pow(ortho_loss, 2);
            proj_cost[id] += proj_loss;
        }

        // Now for context word vectors
        if (context_content_len > 0) {
            for (b = 0; b < vector_size; b++) {
                temp = (W[b + l2] - context_def_content_sum[b]) > 0 ? (W[b + l2] - context_def_content_sum[b]) : -(W[b + l2] - context_def_content_sum[b]);
                if (!isnan(temp) && !isinf(temp)) {
                    def_loss += temp;
                    // Update for definition loss
                    if (use_def_loss) {
                        diff_def = (W[b + l2] - context_def_content_sum[b]) > 0 ? 1 : -1;
                        W[b + l2] -= diff_def * eta * lambda / sqrt(gradsq[b + l2]);
                    }
                } else {
                    fprintf(stderr, "Caught nan or inf for definition loss update l2.\n");
                    continue;
                }

                context_s_dot_s += context_def_content_sum[b] * context_def_content_sum[b];
                context_w_dot_s +=  W[b + l2] * context_def_content_sum[b];
                context_s_dot_g += context_def_content_sum[b] * context_debias_direction[b];
                context_g_dot_g += context_debias_direction[b] * context_debias_direction[b];
                context_w_dot_g += W[b + l2] * context_debias_direction[b];
            }

            for (b = 0; b < vector_size; b++) {
                context_proj_w_s[b] = W[b + l2] - context_w_dot_s / context_s_dot_s * context_def_content_sum[b];
                if (!isnan(context_proj_w_s[b]) && !isinf(context_proj_w_s[b])) {
                    context_ortho_loss += context_proj_w_s[b] * W[b + l2]; // for computing gradients later
                } else {
                    fprintf(stderr, "Caught nan or inf for ortho loss l2.\n");
                    continue;
                }

                // fprintf(stderr, "debugging: context_proj_w_s at %d is %lf\n", b, context_proj_w_s[b]);
                context_proj_w_g[b] = context_w_dot_g / context_g_dot_g * context_debias_direction[b];
                context_proj_s_g[b] = context_s_dot_g / context_g_dot_g * context_debias_direction[b];
                context_proj_difference[b] = (context_proj_w_g[b] - context_proj_s_g[b]) > 0 ? context_g_dot_g : -context_g_dot_g;

            }

            for (b = 0; b < vector_size; b++) {
                temp = (context_proj_w_g[b] - context_proj_s_g[b]) > 0 ? (context_proj_w_g[b] - context_proj_s_g[b]) : -((context_proj_w_g[b] - context_proj_s_g[b]));
                if (!isnan(temp) && !isinf(temp)) {
                    proj_loss += temp;
                    // Update for projection loss
                    if (use_proj_loss) {
                        diff_proj = 0;
                        for (int j = 0; j < vector_size; j++) {
                            diff_proj += context_debias_direction_products[b * vector_size + j] / context_proj_difference[j];
                        }

                        W[b + l2] -= diff_proj * eta * gamma_ / sqrt(gradsq[b + l2]);
                        if (W[b + l2] > 1) W[b + l2] = 1;
                        if (W[b + l2] < -1) W[b + l2] = -1;
                    }
                } else {
                    fprintf(stderr, "Caught nan or inf for projection loss update l2.\n");
                    continue;
                }
            }

            // Update for ortho loss
            if (use_ortho_loss && context_content_len > 0) {
                for (b = 0; b < vector_size; b++) {
                    diff_ortho = 2 * context_ortho_loss * context_proj_w_s[b];
                    if (!isnan(diff_ortho) && !isinf(diff_ortho)) {
                        W[b + l2] -= diff_ortho * eta * beta / sqrt(gradsq[b + l2]);
                    } else {
                        fprintf(stderr, "Caught nan or inf for ortho loss update l2.\n");
                        continue;
                    }
                }
            }

            def_cost[id] += def_loss;
            context_ortho_cost[id] += pow(context_ortho_loss, 2);
            proj_cost[id] += proj_loss;
        }
        else {
            // No definiton, update for proj_loss
            if (use_proj_loss) {
                for (b = 0; b < vector_size; b++) {
                    context_g_dot_g += debias_direction[b] * debias_direction[b];
                    context_w_dot_g += W[b + l1] * debias_direction[b];
                }

                for (b = 0; b < vector_size; b++) {
                    context_proj_w_g[b] = w_dot_g / g_dot_g * debias_direction[b];
                    context_proj_w_g[b] = proj_w_g[b] > 0 ? g_dot_g : - g_dot_g;
                }

                for (b = 0; b < vector_size; b++)  {
                    diff_proj = 0;
                    for (int j = 0; j < vector_size; j++) {
                        diff_proj += context_debias_direction_products[b * vector_size + j] / context_proj_w_g[j];
                    }

                    W[b + l1] -= diff_proj * eta * gamma_ / sqrt(gradsq[b + l1]);
                    if (W[b + l1] > 1) W[b + l1] = 1;
                    if (W[b + l1] < -1) W[b + l1] = -1;
                }
            }
            def_cost[id] += def_loss;
            ortho_cost[id] += pow(context_ortho_loss, 2);
            proj_cost[id] += proj_loss;
        }
    }
    
    free(W_updates1);
    free(W_updates2);

    fclose(fin);
    pthread_exit(NULL);
}

/* Save params to file */
int save_params(int nb_iter) {
    /*
     * nb_iter is the number of iteration (= a full pass through the cooccurrence matrix).
     *   nb_iter  > 0 => checkpointing the intermediate parameters, so nb_iter is in the filename of output file.
     *   nb_iter == 0 => checkpointing the initial parameters
     *   else         => saving the final paramters, so nb_iter is ignored.
     */

    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH+20], output_file_gsq[MAX_STRING_LENGTH+20];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH + 1);
    if (NULL == word) {
        return 1;
    }
    FILE *fid, *fout;
    FILE *fgs = NULL;

    if (use_binary > 0 || nb_iter == 0) {
        // Save parameters in binary file
        // note: always save initial parameters in binary, as the reading code expects binary
        if (nb_iter < 0)
            sprintf(output_file,"%s.bin",save_W_file);
        else
            sprintf(output_file,"%s.%03d.bin",save_W_file,nb_iter);

        fout = fopen(output_file,"wb");
        if (fout == NULL) {log_file_loading_error("weights file", save_W_file); free(word); return 1;}
        for (a = 0; a < 2 * vocab_size * (vector_size + 1); a++) fwrite(&W[a], sizeof(real), 1,fout);
        fclose(fout);
        if (save_gradsq > 0) {
            if (nb_iter < 0)
                sprintf(output_file_gsq,"%s.bin",save_gradsq_file);
            else
                sprintf(output_file_gsq,"%s.%03d.bin",save_gradsq_file,nb_iter);

            fgs = fopen(output_file_gsq,"wb");
            if (fgs == NULL) {log_file_loading_error("gradsq file", save_gradsq_file); free(word); return 1;}
            for (a = 0; a < 2 * vocab_size * (vector_size + 1); a++) fwrite(&gradsq[a], sizeof(real), 1,fgs);
            fclose(fgs);
        }
    }
    if (use_binary != 1) { // Save parameters in text file
        if (nb_iter < 0)
            sprintf(output_file,"%s.txt",save_W_file);
        else
            sprintf(output_file,"%s.%03d.txt",save_W_file,nb_iter);
        if (save_gradsq > 0) {
            if (nb_iter < 0)
                sprintf(output_file_gsq,"%s.txt",save_gradsq_file);
            else
                sprintf(output_file_gsq,"%s.%03d.txt",save_gradsq_file,nb_iter);

            fgs = fopen(output_file_gsq,"wb");
            if (fgs == NULL) {log_file_loading_error("gradsq file", save_gradsq_file); free(word); return 1;}
        }
        fout = fopen(output_file,"wb");
        if (fout == NULL) {log_file_loading_error("weights file", save_W_file); free(word); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if (fid == NULL) {log_file_loading_error("vocab file", vocab_file); free(word); fclose(fout); return 1;}
        if (write_header) fprintf(fout, "%lld %d\n", vocab_size, vector_size);
        for (a = 0; a < vocab_size; a++) {
            if (fscanf(fid,format,word) == 0) {free(word); fclose(fid); fclose(fout); return 1;}
            // input vocab cannot contain special <unk> keyword
            if (strcmp(word, "<unk>") == 0) {free(word); fclose(fid); fclose(fout);  return 1;}
            fprintf(fout, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if (save_gradsq > 0) { // Save gradsq
                fprintf(fgs, "%s",word);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[a * (vector_size + 1) + b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                fprintf(fgs,"\n");
            }
            if (fscanf(fid,format,word) == 0) {
                // Eat irrelevant frequency entry
                fclose(fout);
                fclose(fid);
                free(word);
                return 1;
                }
        }

        if (use_unk_vec) {
            real* unk_vec = (real*)calloc((vector_size + 1), sizeof(real));
            real* unk_context = (real*)calloc((vector_size + 1), sizeof(real));
            strcpy(word, "<unk>");

            long long num_rare_words = vocab_size < 100 ? vocab_size : 100;

            for (a = vocab_size - num_rare_words; a < vocab_size; a++) {
                for (b = 0; b < (vector_size + 1); b++) {
                    unk_vec[b] += W[a * (vector_size + 1) + b] / num_rare_words;
                    unk_context[b] += W[(vocab_size + a) * (vector_size + 1) + b] / num_rare_words;
                }
            }

            fprintf(fout, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_vec[b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_context[b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b]);
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b] + unk_context[b]);
            fprintf(fout,"\n");

            free(unk_vec);
            free(unk_context);
        }

        fclose(fid);
        fclose(fout);
        if (save_gradsq > 0) fclose(fgs);
    }
    free(word);
    return 0;
}

/* Train model */
int train_glove() {
    long long a, file_size;
    int save_params_return_code;
    int b;
    FILE *fin, *fdef;
    real total_cost = 0, total_def_cost = 0, total_ortho_cost = 0, total_proj_cost = 0, total_context_ortho_cost = 0;

    fprintf(stderr, "TRAINING MODEL\n");

    fin = fopen(input_file, "rb");
    if (fin == NULL) {log_file_loading_error("cooccurrence file", input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);

    fdef = fopen(definition_file, "rb");
    fprintf(stderr, "Opened definition file\n");
    if (fdef == NULL) {log_file_loading_error("definition file", definition_file); return 1;}

    fprintf(stderr,"Building definition lists...\n");
    fread(&def_word_num, sizeof(int), 1, fdef);
    word_to_def = (int **)malloc(vocab_size * sizeof(int *));
    int curr_id, curr_size;
    int def_word_ids[def_word_num];  //for free memory later

    for (a = 0; a < def_word_num; a++) {
        if (feof(fdef)) {
            fprintf(stderr, "EOF reached before data fully loaded in %s.\n", definition_file);
            fclose(fdef);
            return -1;
        }
        fread(&curr_id, sizeof(int), 1, fdef);
        fread(&curr_size, sizeof(int), 1, fdef);
        word_to_def[curr_id] =(int *)malloc(sizeof(int) * (curr_size+1));
        def_word_ids[a] = curr_id;
        word_to_def[curr_id][0] = curr_size;
        for (int i = 0; i < curr_size; i++){
              fread(&(word_to_def[curr_id][i+1]), sizeof(int), 1, fdef);
        }
    }
    fclose(fdef);

    if (verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if (verbose > 1) fprintf(stderr,"done.\n");
    if (save_init_param) {
        if (verbose > 1) fprintf(stderr,"Saving initial parameters... ");
        save_params_return_code = save_params(0);
        if (save_params_return_code != 0)
            return save_params_return_code;
        if (verbose > 1) fprintf(stderr,"done.\n");
    }
    if (verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if (verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if (verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if (verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    if (verbose > 0) fprintf(stderr,"use_def_loss: %d\n", use_def_loss);
    if (verbose > 0) fprintf(stderr,"use_ortho_loss: %d\n", use_ortho_loss);
    if (verbose > 0) fprintf(stderr,"use_proj_loss: %d\n", use_proj_loss);
    if (verbose > 0) fprintf(stderr,"lambda: %0.9lf\n", lambda);
    if (verbose > 0) fprintf(stderr,"beta: %0.9lf\n", beta);
    if (verbose > 0) fprintf(stderr,"gamma: %0.9lf\n", gamma_);
    if (verbose > 0) fprintf(stderr,"MALE index: %d\n", SEED_WORD_1);
    if (verbose > 0) fprintf(stderr,"FEMALE index: %d\n", SEED_WORD_2);

    bias_word_value = (real *)malloc(sizeof(real) * (vocab_size));
    bias_def_value = (real *)malloc(sizeof(real) * (vocab_size));
    memset(bias_word_value, 0, sizeof(real) * (vocab_size));
    memset(bias_def_value, 0, sizeof(real) * (vocab_size));

    context_bias_word_value = (real *)malloc(sizeof(real) * (vocab_size));
    context_bias_def_value = (real *)malloc(sizeof(real) * (vocab_size));
    memset(context_bias_word_value, 0, sizeof(real) * (vocab_size));
    memset(context_bias_def_value, 0, sizeof(real) * (vocab_size));

    bias_word_value_temp = (real *)malloc(sizeof(real) * (cap));
    bias_def_value_temp = (real *)malloc(sizeof(real) * (cap));
    memset(bias_word_value_temp, 0, sizeof(real) * (cap));
    memset(bias_def_value_temp, 0, sizeof(real) * (cap));

    context_bias_word_value_temp = (real *)malloc(sizeof(real) * (cap));
    context_bias_def_value_temp = (real *)malloc(sizeof(real) * (cap));
    memset(context_bias_word_value_temp, 0, sizeof(real) * (cap));
    memset(context_bias_def_value_temp, 0, sizeof(real) * (cap));

    boy_candidate_words = (long long *)malloc(sizeof(long long) * (seed_1_top_num));
    girl_candidate_words = (long long *)malloc(sizeof(long long) * (seed_2_top_num));

    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long));

    time_t rawtime;
    struct tm *info;
    char time_buffer[80];
    // Lock-free asynchronous SGD
    for (b = 0; b < num_iter; b++) {

        // Get s(w) for boy and girl
        int boy_def_len = 11; // HARD-CODED to use dominant gloss only; original: word_to_def[SEED_WORD_1][0];
        int girl_def_len = word_to_def[SEED_WORD_2][0];
        real boy_def_sum[vector_size], girl_def_sum[vector_size];
        long long content_id;
        // Initialize all values to zeros
        for (int c = 0; c < vector_size; c++) {
            boy_def_sum[c] = 0;
            girl_def_sum[c] = 0;
        }
        for (int l = 0; l < boy_def_len; l++) {
            content_id = word_to_def[SEED_WORD_1][l+1] * (vector_size + 1);
            for (int c = 0; c < vector_size; c++) {
                boy_def_sum[c] += W[c + content_id] / boy_def_len;
            }
        }
        for (int l = 0; l < girl_def_len; l++) {
            content_id = word_to_def[SEED_WORD_2][l+1] * (vector_size + 1);
            for (int c = 0; c < vector_size; c++) {
                girl_def_sum[c] += W[c + content_id] / girl_def_len;
            }
        }
        // Get s(w) for context boy and girl
        real context_boy_def_sum[vector_size], context_girl_def_sum[vector_size];
        // Initialize all values to zeros
        for (int c = 0; c < vector_size; c++) {
            context_boy_def_sum[c] = 0;
            context_girl_def_sum[c] = 0;
        }
        for (int l = 0; l < boy_def_len; l++) {
            content_id = (word_to_def[SEED_WORD_1][l+1] + vocab_size) * (vector_size + 1);
            for (int c = 0; c < vector_size; c++) {
                context_boy_def_sum[c] += W[c + content_id] / boy_def_len;
            }
        }
        for (int l = 0; l < girl_def_len; l++) {
            content_id = (word_to_def[SEED_WORD_2][l+1] + vocab_size) * (vector_size + 1);
            for (int c = 0; c < vector_size; c++) {
                context_girl_def_sum[c] += W[c + content_id] / girl_def_len;
            }
        }


        // Get norm of boy, girl and their definition sum
        real boy_norm = 0, girl_norm = 0, boy_def_norm = 0, girl_def_norm = 0;
        real context_boy_norm = 0, context_girl_norm = 0, context_boy_def_norm = 0, context_girl_def_norm = 0;
        for (int c = 0; c < vector_size; c++) {
            boy_norm += pow(W[c + SEED_WORD_1 * (vector_size + 1)], 2);
            girl_norm += pow(W[c + SEED_WORD_2 * (vector_size + 1)], 2);
            boy_def_norm += pow(boy_def_sum[c], 2);
            girl_def_norm += pow(girl_def_sum[c], 2);
            context_boy_norm += pow(W[c + (SEED_WORD_1 + vocab_size) * (vector_size + 1)], 2);
            context_girl_norm += pow(W[c + (SEED_WORD_2 + vocab_size) * (vector_size + 1)], 2);
            context_boy_def_norm += pow(context_boy_def_sum[c], 2);
            context_girl_def_norm += pow(context_girl_def_sum[c], 2);
        }
        boy_norm = sqrt(boy_norm);
        girl_norm = sqrt(girl_norm);
        boy_def_norm = sqrt(boy_def_norm);
        girl_def_norm = sqrt(girl_def_norm);
        context_boy_norm = sqrt(context_boy_norm);
        context_girl_norm = sqrt(context_girl_norm);
        context_boy_def_norm = sqrt(context_boy_def_norm);
        context_girl_def_norm = sqrt(context_girl_def_norm);

        // Iterate through all words to get bias word value, bias def value, and same for context words
        for (long long word_id = 0; word_id < vocab_size; word_id++){
            real word_bias = 0, context_word_bias = 0, def_bias = 0, context_def_bias = 0;

            // Find word bias
            real word_dot_boy = 0, word_dot_girl = 0, word_norm = 0;
            for (int c = 0; c < vector_size; c++) {
                real curr = W[c + word_id * (vector_size + 1)];
                word_dot_boy += curr * W[c + SEED_WORD_1 * (vector_size + 1)];
                word_dot_girl += curr * W[c + SEED_WORD_2 * (vector_size + 1)];
                word_norm += pow(curr, 2);
            }
            word_norm = sqrt(word_norm);
            word_bias = word_dot_boy / (word_norm * boy_norm) - word_dot_girl / (word_norm * girl_norm);
            bias_word_value[word_id] = word_bias;
            if (word_id < cap && (b < 10 || b % 10 == 0)) bias_word_value_temp[word_id] = word_bias;

            real context_word_dot_boy = 0, context_word_dot_girl = 0, context_word_norm = 0;
            for (int c = 0; c < vector_size; c++) {
                real curr = W[c + (word_id + vocab_size) * (vector_size + 1)];
                context_word_dot_boy += curr * W[c + (SEED_WORD_1 + vocab_size) * (vector_size + 1)];
                context_word_dot_girl += curr * W[c + (SEED_WORD_2 + vocab_size) * (vector_size + 1)];
                context_word_norm += pow(curr, 2);
            }
            context_word_norm = sqrt(context_word_norm);
            context_word_bias = context_word_dot_boy / (context_word_norm * context_boy_norm) -
                                     context_word_dot_girl / (context_word_norm * context_girl_norm);
            context_bias_word_value[word_id] = context_word_bias;
            if (word_id < cap && (b < 10 || b % 10 == 0)) context_bias_word_value_temp[word_id] = context_word_bias;

            // Find definition bias
            int content_len = word_to_def[word_id][0];
            real def_content_sum[vector_size], context_def_content_sum[vector_size];
            for (int c = 0; c < vector_size; c++) {
                def_content_sum[c] = 0;
                context_def_content_sum[c] = 0;
            }
            // s(w) for word
            for (int l = 0; l < content_len; l++) {
                long long content_id = word_to_def[word_id][l+1] * (vector_size + 1);
                for (int c = 0; c < vector_size; c++) {
                    def_content_sum[c] += W[c + content_id]/ content_len;
                }
            }
            // s(w) for context word
            for (int l = 0; l < content_len; l++) {
                long long content_id = (word_to_def[word_id][l+1] + vocab_size) * (vector_size + 1);
                for (int c = 0; c < vector_size; c++) {
                    context_def_content_sum[c] += W[c + content_id] / content_len;
                }
            }

            // If no definition, 0 bias
            if (content_len == 0) {
                bias_def_value[word_id] = 0;
                context_bias_def_value[word_id] = 0;
                if (word_id < cap && (b < 10 || b % 10 == 0)) bias_def_value_temp[word_id] = 0;
                if (word_id < cap && (b < 10 || b % 10 == 0)) context_bias_def_value_temp[word_id] = 0;
            }
            else {
                real def_dot_boy = 0, def_dot_girl = 0, def_norm = 0;
                for (int c = 0; c < vector_size; c++) {
                    def_dot_boy += def_content_sum[c] * boy_def_sum[c];
                    def_dot_girl += def_content_sum[c] * girl_def_sum[c];
                    def_norm += pow(def_content_sum[c], 2);
                }
                def_norm = sqrt(def_norm);
                def_bias = def_dot_boy / (def_norm * boy_def_norm) - def_dot_girl / (def_norm * girl_def_norm);
                bias_def_value[word_id]  = def_bias;
                if (word_id < cap && (b < 10 || b % 10 == 0)) bias_def_value_temp[word_id]  = def_bias;

                real context_def_dot_boy = 0, context_def_dot_girl = 0, context_def_norm = 0;
                for (int c = 0; c < vector_size; c++) {
                    context_def_dot_boy += context_def_content_sum[c] * context_boy_def_sum[c];
                    context_def_dot_girl += context_def_content_sum[c] * context_girl_def_sum[c];
                    context_def_norm += pow(context_def_content_sum[c], 2);
                }
                context_def_norm = sqrt(context_def_norm);
                context_def_bias = context_def_dot_boy / (context_def_norm * context_boy_def_norm) -
                                        context_def_dot_girl / (context_def_norm * context_girl_def_norm);
                context_bias_def_value[word_id] = context_def_bias;
                if (word_id < cap && (b < 10 || b % 10 == 0)) context_bias_def_value_temp[word_id] = context_def_bias;
            }
            

        } // End of iterating all vocab to find biases

        // Find most attribute-specific words
        if (b == 0) {
            qsort(bias_word_value_temp, cap, sizeof(real), compare);
            qsort(context_bias_word_value_temp, cap, sizeof(real), compare);
            qsort(bias_def_value_temp, cap, sizeof(real), compare);
            qsort(context_bias_def_value_temp, cap, sizeof(real), compare);

            boy_added = 0, girl_added = 0;
            for (long long word_id = 0; word_id < cap; word_id++) {
                if (bias_def_value[word_id] < bias_def_value_temp[seed_2_top_num]){
                    girl_candidate_words[girl_added] = word_id;
                    girl_added++;
                }
                else if (bias_def_value[word_id] > bias_def_value_temp[cap - seed_1_top_num - 1]) {
                    boy_candidate_words[boy_added] = word_id;
                    boy_added++;
                }

            }
        }

        total_cost = 0;
        total_def_cost = 0;
        total_ortho_cost = 0;
        total_context_ortho_cost = 0;
        total_proj_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        long long *thread_ids = (long long*)malloc(sizeof(long long) * num_threads);
        for (a = 0; a < num_threads; a++) thread_ids[a] = a;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)&thread_ids[a]);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        for (a = 0; a < num_threads; a++) {
            total_cost += cost[a];
            total_def_cost += def_cost[a];
            total_ortho_cost += ortho_cost[a];
            total_context_ortho_cost += context_ortho_cost[a];
            total_proj_cost += proj_cost[a];
        }
        free(thread_ids);

        time(&rawtime);
        info = localtime(&rawtime);
        strftime(time_buffer,80,"%x - %I:%M.%S%p", info);
        fprintf(stderr, "%s, iter: %03d, glove_cost: %lf, def_cost: %lf, ortho_cost: %lf, context_ortho_cost: %lf, proj_cost: %lf\n\n\n", time_buffer,
               b+1, total_cost/num_lines, total_def_cost/(vocab_size * num_threads), total_ortho_cost/(vocab_size * num_threads),
               total_context_ortho_cost/(vocab_size * num_threads), total_proj_cost/(vocab_size * num_threads));


        if (checkpoint_every > 0 && (b + 1) % checkpoint_every == 0) {
            fprintf(stderr,"    saving intermediate parameters for iter %03d...", b+1);
            save_params_return_code = save_params(b+1);
            if (save_params_return_code != 0) {
                free(pt);
                free(lines_per_thread);
                return save_params_return_code;
            }
            fprintf(stderr,"done.\n");
        }
    }
    free(pt);
    free(lines_per_thread);

    for (a = 0; a < def_word_num; a++) {
        free(word_to_def[def_word_ids[a]]);
    }
    free(word_to_def);
    free(bias_word_value);
    free(bias_def_value);
    free(context_bias_word_value);
    free(context_bias_def_value);
    free(boy_candidate_words);
    free(girl_candidate_words);

    return save_params(-1);
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    int result = 0;

    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-write-header <int>\n");
        printf("\t\tIf 1, write vocab_size/vector_size as first line. Do nothing if 0 (default).\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-grad-clip\n");
        printf("\t\tGradient components clipping parameter. Values will be clipped to [-grad-clip, grad-clip] interval\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-model <int>\n");
        printf("\t\tModel for word vector output (for text output only); default 2\n");
        printf("\t\t   0: output all data, for both word and context word vectors, including bias terms\n");
        printf("\t\t   1: output word vectors, excluding bias terms\n");
        printf("\t\t   2: output word vectors + context word vectors, excluding bias terms\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-gradsq-file <file>\n");
        printf("\t\tFilename, excluding extension, for squared gradient output; default gradsq\n");
        printf("\t-save-gradsq <int>\n");
        printf("\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified\n");
        printf("\t-checkpoint-every <int>\n");
        printf("\t\tCheckpoint a  model every <int> iterations; default 0 (off)\n");
        printf("\t-load-init-param <int>\n");
        printf("\t\tLoad initial parameters from -init-param-file; default 0 (false)\n");
        printf("\t-save-init-param <int>\n");
        printf("\t\tSave initial parameters (i.e., checkpoint the model before any training); default 0 (false)\n");
        printf("\t-init-param-file <file>\n");
        printf("\t\tBinary initial parameters file to be loaded if -load-init-params is 1; (default is to look for vectors.000.bin)\n");
        printf("\t-load-init-gradsq <int>\n");
        printf("\t\tLoad initial squared gradients from -init-gradsq-file; default 0 (false)\n");
        printf("\t-init-gradsq-file <file>\n");
        printf("\t\tBinary initial squared gradients file to be loaded if -load-init-gradsq is 1; (default is to look for gradsq.000.bin)\n");
        printf("\t-seed <int>\n");
        printf("\t\tRandom seed to use.  If not set, will be randomized using current time.");
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
        result = 0;
    } else {
        if ((i = find_arg((char *)"-write-header", argc, argv)) > 0) write_header = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        cost = malloc(sizeof(real) * num_threads);
        def_cost = malloc(sizeof(real) * num_threads);
        ortho_cost = malloc(sizeof(real) * num_threads);
        context_ortho_cost = malloc(sizeof(real) * num_threads);
        proj_cost = malloc(sizeof(real) * num_threads);
        if ((i = find_arg((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-x-max", argc, argv)) > 0) x_max = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-grad-clip", argc, argv)) > 0) grad_clip_value = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
        if (model != 0 && model != 1) model = 2;
        if ((i = find_arg((char *)"-save-gradsq", argc, argv)) > 0) save_gradsq = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
        else strcpy(vocab_file, (char *)"vocab.txt");
        if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(save_W_file, argv[i + 1]);
        else strcpy(save_W_file, (char *)"vectors");
        if ((i = find_arg((char *)"-gradsq-file", argc, argv)) > 0) {
            strcpy(save_gradsq_file, argv[i + 1]);
            save_gradsq = 1;
        }
        else if (save_gradsq > 0) strcpy(save_gradsq_file, (char *)"gradsq");
        if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
        else strcpy(input_file, (char *)"cooccurrence.shuf.bin");
        if ((i = find_arg((char *)"-checkpoint-every", argc, argv)) > 0) checkpoint_every = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-init-param-file", argc, argv)) > 0) strcpy(init_param_file, argv[i + 1]);
        else strcpy(init_param_file, (char *)"vectors.000.bin");
        if ((i = find_arg((char *)"-load-init-param", argc, argv)) > 0) load_init_param = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-save-init-param", argc, argv)) > 0) save_init_param = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-init-gradsq-file", argc, argv)) > 0) strcpy(init_gradsq_file, argv[i + 1]);
        else strcpy(init_gradsq_file, (char *)"gradsq.000.bin");
        if ((i = find_arg((char *)"-load-init-gradsq", argc, argv)) > 0) load_init_gradsq = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-seed", argc, argv)) > 0) seed = atoi(argv[i + 1]);

        strcpy(definition_file, "definitions.dat"); // change file name here

        if ((i = find_arg((char *)"-use-def-loss", argc, argv)) > 0) use_def_loss = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-use-ortho-loss", argc, argv)) > 0) use_ortho_loss = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-use-proj-loss", argc, argv)) > 0) use_proj_loss = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-beta", argc, argv)) > 0) beta = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-gamma", argc, argv)) > 0) gamma_ = atof(argv[i + 1]);

        vocab_size = 0;
        fid = fopen(vocab_file, "r");
        if (fid == NULL) {log_file_loading_error("vocab file", vocab_file); free(cost); return 1;}
        while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
        fclose(fid);
        if (vocab_size == 0) {fprintf(stderr, "Unable to find any vocab entries in vocab file %s.\n", vocab_file); free(cost); return 1;}
        result = train_glove();
        free(cost);
        free(def_cost);
        free(ortho_cost);
        free(context_ortho_cost);
        free(proj_cost);
    }
    free(W);
    free(gradsq);

    return result;
}