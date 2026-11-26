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

// 1/0.06: about 38 frames (1.3 s at 30 FPS) to reach 0.9 from zero.
static const float kRise = 0.06;
static const float kStill = 0.02;   // largest channel change still counted as the same pixel
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
    // channel never lets a bar reach a second of confidence. Real motion is a luma step.
    const bool moved = lumaChange > 0.055;
    const bool hold = lumaChange < 0.012 || (lumaChange < 0.028 && change < 0.07);
    if (moved) s.x = 0;
    else if (hold) s.x = s.x + (1 - s.x) * kRise;
    else s.x *= 0.82;
    s.y = moved ? contrast : max(contrast, s.y * kFade);
    stateTex[id.xy] = float4(s, same, 0);
}
