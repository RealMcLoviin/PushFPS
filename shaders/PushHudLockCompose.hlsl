// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 DosukaSOL and the PushFPS contributors. Part of PushFPS, a modified version of Veyra.
// PushFPS Live HUD Lock, step 2 (pipeline/PushHudLock.h): a generated frame takes its locked
// pixels from the held real frame. A pixel is locked when it has been still for about a second
// itself and it, or a neighbour that is just as still, has an edge: the inside of a thin stroke
// and the pixel beside an edge belong to the overlay too. A pixel that moves is never locked.
Texture2D<float4> generatedTex : register(t0);
Texture2D<float4> heldTex : register(t1);
Texture2D<float4> stateTex : register(t2);
// PushFPS 4.0, the still rule (pipeline/PushHudStill.h): per block of 16 x 16 pixels, 0 when the
// surrounding (the 5 x 5 blocks around it) mostly moves, 1 when half or more of it stands still.
Texture2D<float4> shareTex : register(t3);
RWTexture2D<float4> outputTex : register(u0);
cbuffer Constants : register(b0) {
    // flags: 1 Live HUD Lock, 2 the still rule, 4 in place: outputTex is the generated frame itself,
    // only the pixels the still rule keeps are written and generatedTex is never read (the still rule
    // alone: one read of the state per pixel instead of a whole frame read, written and copied back).
    uint width, height, outputMode, flags;
    uint pad1, pad2, pad3, pad4;
};
// Four rings, eight taps. Radii 4, 8, 16, 32 would be too wide; 4/8/16/24 stay inside a bar.
static const int2 kSpread[32] = {
    int2(4, 0), int2(-4, 0), int2(0, 4), int2(0, -4), int2(3, 3), int2(-3, 3), int2(3, -3), int2(-3, -3),
    int2(8, 0), int2(-8, 0), int2(0, 8), int2(0, -8), int2(6, 6), int2(-6, 6), int2(6, -6), int2(-6, -6),
    int2(16, 0), int2(-16, 0), int2(0, 16), int2(0, -16), int2(11, 11), int2(-11, 11), int2(11, -11), int2(-11, -11),
    int2(24, 0), int2(-24, 0), int2(0, 24), int2(0, -24), int2(17, 17), int2(-17, 17), int2(17, -17), int2(-17, -17)
};
static const float kSpreadR[32] = {
    4, 4, 4, 4, 4.24, 4.24, 4.24, 4.24,
    8, 8, 8, 8, 8.49, 8.49, 8.49, 8.49,
    16, 16, 16, 16, 15.56, 15.56, 15.56, 15.56,
    24, 24, 24, 24, 24.04, 24.04, 24.04, 24.04
};
float stillOf(float2 s) { return smoothstep(0.80, 0.95, s.x); }
float edgeOf(float2 s) {
    const float edge = outputMode == 1 ? 0.03 : 0.04;
    return smoothstep(edge, edge * 2.5, s.y);
}
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    const float4 here = stateTex[id.xy];
    const int2 last = int2(width - 1, height - 1);
    // Live HUD Lock. The matte is per pixel: this pixel must be still, and a still edge
    // must exist on it or along a short ring. Falloff is continuous, so a flat bar center
    // is covered without a rectangle and without a 16x16 decision.
    float w = 0;
    if ((flags & 1) != 0) {
        const float still = stillOf(here.xy);
        float influence = 0;
        if (still > 0) {
            [unroll] for (int y = -1; y <= 1; ++y)
                [unroll] for (int x = -1; x <= 1; ++x) {
                    const float2 q = stateTex[clamp(int2(id.xy) + int2(x, y), int2(0, 0), last)].xy;
                    influence = max(influence, edgeOf(q) * stillOf(q));
                }
            [unroll] for (int k = 0; k < 32; ++k) {
                const float fall = saturate(1.0 - kSpreadR[k] / 28.0);
                const float2 q = stateTex[clamp(int2(id.xy) + kSpread[k], int2(0, 0), last)].xy;
                influence = max(influence, edgeOf(q) * stillOf(q) * fall);
            }
        }
        w = still * smoothstep(0.04, 0.22, influence);
    }
    // The still rule: the same in both real frames, with company, in a surrounding that mostly moves.
    // The surrounding is asked first: over a still picture (a menu) the test ends there for every pixel.
    bool keep = false;
    if ((flags & 2) != 0 && here.z > 0.5 && shareTex[id.xy / 16].x < 0.5) {
        uint beside = 0, around = 0;
        [unroll] for (int y = -2; y <= 2; ++y)
            [unroll] for (int x = -2; x <= 2; ++x) {
                const uint same = stateTex[clamp(int2(id.xy) + int2(x, y), int2(0, 0), last)].z > 0.5 ? 1 : 0;
                around += same;
                if (abs(x) <= 1 && abs(y) <= 1) beside += same;
            }
        keep = beside >= 3 && around >= 5;
    }
    // A kept pixel is the real frame's, whole (its own alpha with its own colour).
    if ((flags & 4) != 0) { if (keep) outputTex[id.xy] = heldTex[id.xy]; return; }
    const float4 g = generatedTex[id.xy];
    if (keep) outputTex[id.xy] = heldTex[id.xy];
    else if (w > 0) {
        // Premultiplied matte. w is the extracted HUD coverage, not a region mask.
        // High frequency stays on the held real pixel. The plate is not blurred in.
        outputTex[id.xy] = float4(heldTex[id.xy].rgb * w + g.rgb * (1.0 - w), g.a);
    } else outputTex[id.xy] = g;
}
