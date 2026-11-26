#pragma once
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 DosukaSOL and the PushFPS contributors. Part of PushFPS, a modified version of Veyra.
//
// PushFPS Live HUD Lock (EnhancementSettings::hudLock): the HUD found pixel by
// pixel on every real frame and kept out of frame generation.
//
// Frame generation interpolates every pixel between two real frames. Over a
// moving picture it pulls a still, sharp overlay (text, crosshair, health bar,
// minimap frame) along with the motion around it, so the overlay smears or
// wobbles in the generated frames. HUD protection fixes this with boxes drawn
// in advance; HUD Lock finds those pixels itself, every frame:
//   1. update (one dispatch per real frame): a per-pixel stillness level grows
//      while the pixel stays the same from frame to frame and drops to zero the
//      moment it changes; beside it, how much contrast the pixel has with its
//      neighbours (a held maximum that fades).
//   2. compose (one dispatch per generated frame): where a pixel's luma has been
//      quiet for a few frames and it, or a short ring around it, has an edge, the
//      generated frame takes that pixel from the held real frame. A confident
//      pixel is replaced whole, so capture noise cannot leave a warped mix in the
//      middle of a bar. The rim falls off. A pixel that moves is released then.
// A still pixel looks the same in the real frame, so locking one that is not
// HUD (a still wall) changes nothing visible; a pixel that starts to move is
// released on that same frame. Capture noise that never lets a pixel settle
// leaves it to frame generation, as without HUD Lock.
#include "veyra/pipeline/GpuPassUtils.h"
#include <algorithm>
#include <array>
#include <string>

namespace veyra::pipeline {

class PushHudLock {
public:
    static constexpr UINT kConstants = 8;
    static constexpr UINT kMaxGenerated = 32;
    // outputMode: how the frames are encoded, as EffectExclusion (0 sRGB, 1 scRGB, 2 PQ).
    // lock: Live HUD Lock (what this pass did alone until 4.0). still: the still rule
    // (PushHudStill.h). With lock alone every generated pixel is what it was before 4.0.
    // If the still rule cannot be set up it is left out and the lock, when asked for, still runs.
    bool initialize(ID3D12Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, unsigned outputMode,
                    bool lock = true, bool still = false);
    bool lockActive() const { return lock_; }
    bool stillActive() const { return still_; }
    // The still rule alone writes its pixels straight into the generated frame where that texture
    // allows it (the graph's do): nothing is read from it, written whole or copied back. With the lock
    // on, with a texture that cannot be written this way, or with VEYRA_HUD_STILL_COPY=1 (for
    // comparison), the frame goes through the scratch copy as before 4.0.
    bool composesInPlace(unsigned slot) const { return slot < generatedCount_ && inPlace_[slot]; }
    // real[0..1]: the graph's two real-frame textures by parity; generated: the
    // generated-frame textures, in the graph's slot order.
    void createViews(ID3D12Device* device, DescriptorStager& stager, ID3D12Resource* const real[2],
                     ID3D12Resource* const* generated, unsigned generatedCount);
    // After real frame `parity` is final: learn from real[1-parity] -> real[parity].
    // reset: no valid previous frame (start, seek, settings change): start over.
    bool update(ID3D12GraphicsCommandList* list, StateTracker& tracker, unsigned parity, bool reset);
    // Locked pixels of generated[slot] come from the held real frame real[heldParity].
    bool compose(ID3D12GraphicsCommandList* list, StateTracker& tracker, unsigned slot, unsigned heldParity);
    std::string describe() const;

private:
    uint32_t width_ = 0, height_ = 0;
    unsigned outputMode_ = 0, generatedCount_ = 0;
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN;
    ComputePass update_, compose_, surround_;
    ComPtr<ID3D12Resource> state_, scratch_;
    // The still rule: per block of 16 x 16 pixels, how many of its pixels stand still, and whether
    // the 5 x 5 blocks around it mostly stand still (what the compose step reads: 0 = mostly moves).
    ComPtr<ID3D12Resource> blocks_, share_;
    uint32_t blocksWide_ = 0, blocksHigh_ = 0;
    bool lock_ = true, still_ = false, copyOnly_ = false;
    bool inPlace_[kMaxGenerated]{};
    ID3D12Resource* real_[2]{};
    ID3D12Resource* generated_[kMaxGenerated]{};
};
}
