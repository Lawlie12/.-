// ============================================================================
// ⚡ TITAN AEGIS V9 - SPECTRE INFERENCE ENGINE (C++)
// Sparse Predictive Ternary Residual Architecture
// ============================================================================

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

const int SCALE = 3;
const int CIN = 3;
const int MID = 24;  // Must match Config.MID_CH from Python!
const int COUT = 27; // 3 * 3^2

// Global Weight Tensors & Dynamic Scales
float s1, s2, s3, s4;
int8_t w1[MID][CIN][3][3];
int8_t w2[MID][MID][3][3];
int8_t w3[MID][MID][3][3];
int8_t w4[COUT][MID][3][3];

// Flat memory indexer (CHW format)
inline size_t idx3(int c, int y, int x, int H, int W) {
    return (size_t)c * H * W + (size_t)y * W + x;
}

// ----------------------------------------------------------------------------
// 🚀 TERNARY MATRIX CONVOLUTION KERNEL
// Processes {-1, 0, 1} math and restores float ranges via the dynamic scale.
// ----------------------------------------------------------------------------
void conv3x3(const vector<float>& in, vector<float>& out, const int8_t* weight, 
             float scale, int in_c, int out_c, int H, int W, bool do_relu) {
    
    #pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_c; ++oc) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            int py = y + ky;
                            int px = x + kx;
                            if (py >= 0 && py < H && px >= 0 && px < W) {
                                // Direct memory access to 4D tensor flat pointer
                                int8_t wt = weight[((oc * in_c + ic) * 3 + (ky + 1)) * 3 + (kx + 1)];
                                if (wt == 1) sum += in[idx3(ic, py, px, H, W)];
                                else if (wt == -1) sum -= in[idx3(ic, py, px, H, W)];
                            }
                        }
                    }
                }
                sum *= scale; // Restore color dynamics using Python's exported mean scale
                
                // Activation
                if (do_relu) out[idx3(oc, y, x, H, W)] = (sum > 0.0f) ? sum : sum * 0.1f;
                else         out[idx3(oc, y, x, H, W)] = sum;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 🚀 BASELINE BILINEAR UPSCALER
// Creates the smooth 1080p canvas for the AI to inject sharp residuals into.
// ----------------------------------------------------------------------------
void upscaleBilinear(const vector<float>& in, int H, int W, vector<float>& out) {
    int Hout = H * SCALE;
    int Wout = W * SCALE;
    
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < CIN; ++c) {
        for (int y = 0; y < Hout; ++y) {
            for (int x = 0; x < Wout; ++x) {
                float srcX = (x + 0.5f) / SCALE - 0.5f;
                float srcY = (y + 0.5f) / SCALE - 0.5f;
                
                int x1 = max(0, min(W - 1, (int)floor(srcX)));
                int y1 = max(0, min(H - 1, (int)floor(srcY)));
                int x2 = max(0, min(W - 1, x1 + 1));
                int y2 = max(0, min(H - 1, y1 + 1));
                
                float dx = srcX - x1;
                float dy = srcY - y1;

                float p11 = in[idx3(c, y1, x1, H, W)];
                float p12 = in[idx3(c, y1, x2, H, W)];
                float p21 = in[idx3(c, y2, x1, H, W)];
                float p22 = in[idx3(c, y2, x2, H, W)];

                out[idx3(c, y, x, Hout, Wout)] = (1.0f - dx) * (1.0f - dy) * p11 +
                                                 dx * (1.0f - dy) * p12 +
                                                 (1.0f - dx) * dy * p21 +
                                                 dx * dy * p22;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// MAIN EXECUTION
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        cout << "========================================================\n";
        cout << " ⚡ TITAN AEGIS V9 (SPECTRE) | 4-LAYER TERNARY ENGINE ⚡ \n";
        cout << "========================================================\n\n";

        if (argc < 3) {
            cout << "❌ ERROR: Missing inputs.\n";
            cout << "Usage: aegis_upscaler.exe <input_image.jpg> <output_image.jpg>\n";
            cout << "Press ENTER to exit...";
            cin.get();
            return -1;
        }

        string input_path = argv[1];
        string output_path = argv[2];

        // 1. INGEST 16.8 KB SPECTRE WEIGHTS
        ifstream file("spectre_weights.bin", ios::binary | ios::ate);
        if (!file) {
            cout << "❌ FATAL: Could not find 'spectre_weights.bin'.\n";
            cin.get(); return -1;
        }
        
        size_t fileSize = file.tellg();
        size_t expectedSize = (4 * 4) + 648 + 5184 + 5184 + 5832; // 16,864 bytes
        if (fileSize != expectedSize) {
            cout << "❌ FATAL: Binary mismatch. Expected " << expectedSize << " bytes, got " << fileSize << ".\n";
            cin.get(); return -1;
        }
        
        file.seekg(0, ios::beg);
        
        // Read Scale Float -> Read Int8 Array (For all 4 layers)
        file.read(reinterpret_cast<char*>(&s1), 4); file.read(reinterpret_cast<char*>(w1), sizeof(w1));
        file.read(reinterpret_cast<char*>(&s2), 4); file.read(reinterpret_cast<char*>(w2), sizeof(w2));
        file.read(reinterpret_cast<char*>(&s3), 4); file.read(reinterpret_cast<char*>(w3), sizeof(w3));
        file.read(reinterpret_cast<char*>(&s4), 4); file.read(reinterpret_cast<char*>(w4), sizeof(w4));
        cout << "[OK] SPECTRE 4-Layer matrices and dynamic float scales loaded.\n";

        // 2. LOAD TARGET IMAGE
        int W, H, channels;
        unsigned char* img_data = stbi_load(input_path.c_str(), &W, &H, &channels, 3);
        if (!img_data) {
            cout << "❌ FATAL: Failed to read image: " << input_path << "\n";
            cin.get(); return -1;
        }
        cout << "[OK] Target frame parsed: " << W << "x" << H << "\n";

        auto t_start = chrono::high_resolution_clock::now();

        vector<float> input(CIN * H * W, 0.0f);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int src_idx = (y * W + x) * 3;
                input[idx3(0, y, x, H, W)] = img_data[src_idx] / 255.0f;
                input[idx3(1, y, x, H, W)] = img_data[src_idx + 1] / 255.0f;
                input[idx3(2, y, x, H, W)] = img_data[src_idx + 2] / 255.0f;
            }
        }
        stbi_image_free(img_data);

        // 3. COMPUTE BILINEAR BASELINE (1080p Canvas)
        int Hout = H * SCALE;
        int Wout = W * SCALE;
        vector<float> base_hr(CIN * Hout * Wout, 0.0f);
        upscaleBilinear(input, H, W, base_hr);

        // 4. EXECUTE TERNARY AI CORE (Generates the high-frequency residual)
        cout << ">> Engaging SPECTRE 4-Layer Residual Prediction...\n";
        vector<float> x1(MID * H * W, 0.0f);
        vector<float> x2(MID * H * W, 0.0f);
        vector<float> x3(MID * H * W, 0.0f);
        vector<float> x4(COUT * H * W, 0.0f);

        conv3x3(input, x1, (int8_t*)w1, s1, CIN, MID, H, W, true);
        conv3x3(x1, x2, (int8_t*)w2, s2, MID, MID, H, W, true);
        conv3x3(x2, x3, (int8_t*)w3, s3, MID, MID, H, W, true);
        conv3x3(x3, x4, (int8_t*)w4, s4, MID, COUT, H, W, false); // No activation on output

        // 5. FUSE RESIDUAL + BASELINE (PixelShuffle & Tanh bounded math)
        cout << ">> Fusing sharp geometry into output tensor...\n";
        vector<unsigned char> final_img(Hout * Wout * CIN);
        
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int c_out = 0; c_out < CIN; ++c_out) {
                    for (int sy = 0; sy < SCALE; ++sy) {
                        for (int sx = 0; sx < SCALE; ++sx) {
                            int c_in = c_out * 9 + sy * SCALE + sx;
                            
                            // Calculate bounded AI detail
                            float val = x4[idx3(c_in, y, x, H, W)];
                            float residual = std::tanh(val) * 0.15f; 
                            
                            // Pull baseline smooth pixel
                            int out_y = y * SCALE + sy;
                            int out_x = x * SCALE + sx;
                            float base_val = base_hr[idx3(c_out, out_y, out_x, Hout, Wout)];
                            
                            // Combine and clamp
                            float final_val = std::max(0.0f, std::min(1.0f, base_val + residual));
                            final_img[((out_y * Wout) + out_x) * CIN + c_out] = static_cast<unsigned char>(final_val * 255.0f);
                        }
                    }
                }
            }
        }

        // 6. SAVE IMAGE
        stbi_write_jpg(output_path.c_str(), Wout, Hout, CIN, final_img.data(), 98);
        
        auto t_end = chrono::high_resolution_clock::now();
        double elapsed_ms = chrono::duration<double, std::milli>(t_end - t_start).count();

        cout << "[OK] 1080p Image Restoration Complete.\n";
        cout << ">> Output saved to: " << output_path << "\n";
        cout << ">> Total Matrix Latency: " << elapsed_ms << " ms\n\n";

    } catch (const std::exception& e) {
        cout << "\n❌ RUNTIME CRASH: " << e.what() << "\n"; cin.get(); return -1;
    }
    return 0;
}
