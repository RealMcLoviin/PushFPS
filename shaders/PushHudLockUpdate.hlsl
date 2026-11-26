// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 DosukaSOL and the PushFPS contributors. Part of PushFPS, a modified version of Veyra.
// PushFPS Live HUD Lock, step 1 (pipeline/PushHudLock.h): per pixel, how long it has stayed the
// same (x, 0..1) and how much contrast it has with its neighbours (y, a held maximum that fades).
// PushFPS 4.0: beside them, whether the pixel is the same in this pair of real frames at the tight
// tolerance the still rule uses (z, 1 or 0; pipeline/PushHudStill.h).
Texture2D<float4> currentTex : register(t0);
Texture2D<float4> previousTex : register(t1);
RWTexture2D<float4> stateTex : register(u0);
cbuffer Constants : register(b0) {
    uint width, height, reset, outputMode; // 0 sRGB, 1 scRGB (linear), 2 PQ
    uint lockOn, pad1, pad2, pad3;         // lockOn 0: only the still rule runs, and only z is worked out
};
// A roughly perceptual value: sRGB and PQ are already encoded; scRGB is linear (1 = 80 nits).
float3 perceptual(float3 v) { return outputMode == 1 ? sqrt(max(v, 0) / 2.5) : v; }
float luma(float3 v) { return dot(v, float3(0.2126, 0.7152, 0.0722)); }
float3 at(int2 p) { return perceptual(currentTex[clamp(p, int2(0, 0), int2(width - 1, height - 1))].rgb); }

// 0.35 reaches the compose gate in about eight quiet frames (~270 ms at 30 FPS).
// One frame stays near 0.35, so a single pair does not lock.
// Quiet is 0.035, not wider: the moving-picture test leaked 117 pixels at 0.055.
static const float kRise = 0.35;
static const float kQuiet = 0.035;  // luma steps below this rise; Elgato noise on a flat bar
static const float kMoved = 0.090;  // a real luma step drops the pixel on that same frame
static const float kFade = 0.97;    // held contrast fades about 3 % per frame
static const float kPairStill = 2.0 / 255.0 + 1e-4;   // hudstill::kPairStill

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    if (reset != 0) { stateTex[id.xy] = float4(0, 0, 0, 0); return; }
    const int2 p = int2(id.xy);
    const float3 c = at(p);
    const float3 d = abs(c - perceptual(previousTex[id.xy].rgb));
    const float change = max(d.r, max(d.g, d.b));
    // z stays on the tight pair test. The 16x16 still rule reads it and must not grow.
    const float same = change <= kPairStill ? 1.0 : 0.0;
    if (lockOn == 0) { stateTex[id.xy] = float4(0, 0, same, 0); return; }
    const float l = luma(c);
    const float lumaChange = abs(l - luma(perceptual(previousTex[id.xy].rgb)));
    const float contrast = max(max(abs(l - luma(at(p + int2(1, 0)))), abs(l - luma(at(p - int2(1, 0))))),
                               max(abs(l - luma(at(p + int2(0, 1)))), abs(l - luma(at(p - int2(0, 1))))));
    float2 s = stateTex[id.xy].xy;
    // Elgato NV12 chroma leaks into R/B while the HUD luma stays put. A hard reset on any
    // channel never lets a bar reach confidence. Real motion is a luma step. Steps between
    // the noise floor and real motion hold confidence: decaying them is what left the bar
    // centre under the lock while the camera pulled it.
    const bool moved = lumaChange > kMoved;
    if (moved) s.x = 0;
    else if (lumaChange < kQuiet) s.x = s.x + (1 - s.x) * kRise;
    s.y = moved ? contrast : max(contrast, s.y * kFade);
    stateTex[id.xy] = float4(s, same, 0);
}
