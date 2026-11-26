#pragma once
#include "veyra/engine/PresentationSettings.h"
#include "veyra/gfx/NvidiaVrr.h"
namespace veyra::engine {
// Level 3 means variable refresh may be requested. Level 4 means the driver
// reports it active. AMD FreeSync never reports active, so this panel stays
// on Automatic Present(0,0) instead of waiting 7-15 ms for a vblank.
inline bool variableRefreshRequested(const PresentationSettings& requested,const gfx::VrrState& vrr){
    return requested.variableRefresh&&requested.display==DisplaySync::Automatic&&gfx::vrrLevel(vrr)>=3;
}
inline bool variableRefreshPaired(const PresentationSettings& requested,const gfx::VrrState& vrr){
    return variableRefreshRequested(requested,vrr)&&gfx::vrrLevel(vrr)>=4;
}
}
