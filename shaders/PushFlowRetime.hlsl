// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 DosukaSOL and the PushFPS contributors. Part of PushFPS, a modified version of Veyra.
// One network run at t=0.5 (PushFlowReconstruct owns that picture). This pass builds another time in (0,1)
// from those same grids, sampling the full-resolution real frames. It does not forecast past the newer frame.
Texture2D<float4> previous : register(t0);
Texture2D<float4> current : register(t1);
RWStructuredBuffer<float> grid0 : register(u0);
RWStructuredBuffer<float> grid1 : register(u1);
RWStructuredBuffer<float> mask : register(u2);
RWTexture2D<float4> output : register(u3);
cbuffer Constants : register(b0) {
    uint srcW, srcH, padW, padH;
    uint modelW, modelH, encoding, alignCorners;
    uint maskForFirst, unused0, unused1, unused2;
    float timestep, padT0, padT1, padT2;
};

float3 encodeColor(float3 c) {
    if (encoding == 1) {
        c = saturate(c);
        return lerp(c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055, step(0.0031308, c));
    }
    if (encoding == 2) {
        const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
        const float3 y = pow(saturate(max(c, 0.0) * (80.0 / 10000.0)), m1);
        return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
    }
    return saturate(c);
}
float3 decodeColor(float3 e) {
    e = saturate(e);
    if (encoding == 1) return lerp(e / 12.92, pow((e + 0.055) / 1.055, 2.4), step(0.04045, e));
    if (encoding == 2) {
        const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
        const float3 p = pow(e, 1.0 / m2);
        return pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1) * (10000.0 / 80.0);
    }
    return e;
}
float fade(float x, float2 range) {
    const float t = saturate((x - range.x) / (range.y - range.x));
    return 1.0 - t * t * (3.0 - 2.0 * t);
}
float3 tapFrame(Texture2D<float4> t, float2 q) {
    q = clamp(q, 0.0, float2(srcW - 1, srcH - 1));
    const uint2 a = uint2(q);
    const uint2 b = min(a + 1, uint2(srcW - 1, srcH - 1));
    const float2 f = q - float2(a);
    return lerp(lerp(t[a].rgb, t[uint2(b.x, a.y)].rgb, f.x), lerp(t[uint2(a.x, b.y)].rgb, t[b].rgb, f.x), f.y);
}
float tapMask(float2 q) {
    q = clamp(q, 0.0, float2(modelW - 1, modelH - 1));
    const uint2 a = uint2(q);
    const uint2 b = min(a + 1, uint2(modelW - 1, modelH - 1));
    const float2 f = q - float2(a);
    return lerp(lerp(mask[a.y * padW + a.x], mask[a.y * padW + b.x], f.x), lerp(mask[b.y * padW + a.x], mask[b.y * padW + b.x], f.x), f.y);
}
float2 modelToSource(float2 q) { return (q + 0.5) * float2(srcW, srcH) / float2(modelW, modelH) - 0.5; }
float2 gridToModel(float2 g) {
    const float2 pad = float2(padW, padH);
    return alignCorners != 0 ? (g + 1.0) * 0.5 * (pad - 1.0) : ((g + 1.0) * pad - 1.0) * 0.5;
}
// Step from a model pixel to where the half-way grid reads a frame (PushPredict's stepAt).
float2 stepAt(RWStructuredBuffer<float> grid, float2 s) {
    const float2 c = clamp(s, 0.0, float2(modelW - 1, modelH - 1));
    const uint2 a = uint2(c);
    const uint2 b = min(a + 1, uint2(modelW - 1, modelH - 1));
    const float2 f = c - float2(a);
    const uint i00 = (a.y * padW + a.x) * 2, i10 = (a.y * padW + b.x) * 2, i01 = (b.y * padW + a.x) * 2, i11 = (b.y * padW + b.x) * 2;
    const float2 g00 = float2(grid[i00], grid[i00 + 1]), g10 = float2(grid[i10], grid[i10 + 1]);
    const float2 g01 = float2(grid[i01], grid[i01 + 1]), g11 = float2(grid[i11], grid[i11 + 1]);
    const float2 top = g00 + (g10 - g00) * f.x, bottom = g01 + (g11 - g01) * f.x;
    return gridToModel(top + (bottom - top) * f.y) - c;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= srcW || id.y >= srcH) return;
    const float t = saturate(timestep);
    const float2 y = (float2(id.xy) + 0.5) * float2(modelW, modelH) / float2(srcW, srcH) - 0.5;
    // y = x + (t-0.5)*V(x), V = d1-d0, three fixed-point steps.
    float2 x = y;
    [unroll]
    for (int i = 0; i < 3; ++i) {
        const float2 d0s = stepAt(grid0, x);
        const float2 d1s = stepAt(grid1, x);
        x = y - (t - 0.5) * (d1s - d0s);
    }
    const float2 d0 = stepAt(grid0, x);
    const float2 d1 = stepAt(grid1, x);
    const float2 V = d1 - d0;
    const float2 pA = y - t * V;
    const float2 pB = y + (1.0 - t) * V;
    const float m = saturate(tapMask(x));
    const float mA = maskForFirst != 0 ? m : (1.0 - m);
    const float denom = mA * (1.0 - t) + (1.0 - mA) * t;
    const float wA = denom > 1e-4 ? (mA * (1.0 - t) / denom) : (1.0 - t);
    const float3 ca = encodeColor(tapFrame(previous, modelToSource(pA)));
    const float3 cb = encodeColor(tapFrame(current, modelToSource(pB)));
    const float3 warped = ca * wA + cb * (1.0 - wA);
    float trust = fade(length(x + (t - 0.5) * V - y), float2(0.5, 1.5)) * fade(length(d0 + d1), float2(0.75, 1.5));
    if (x.x < -1.0 || x.y < -1.0 || x.x > float(modelW) || x.y > float(modelH)) trust = 0.0;
    const float3 ea = encodeColor(previous[id.xy].rgb);
    const float3 eb = encodeColor(current[id.xy].rgb);
    const float3 nearer = t <= 0.5 ? ea : eb;
    const float3 oneSide = mA >= 0.5 ? ca : cb;
    const float unsure = fade(abs(mA - 0.5), float2(0.08, 0.20));
    const float3 fallback = lerp(oneSide, nearer, unsure);
    float3 shown = lerp(fallback, warped, trust);
    const float apart = max(abs(ea.r - eb.r), max(abs(ea.g - eb.g), abs(ea.b - eb.b)));
    const float still = fade(length(V), float2(0.25, 0.5)) * fade(apart, float2(0.02, 0.08));
    shown = lerp(shown, nearer, still);
    output[id.xy] = float4(decodeColor(shown), 1.0);
}
