#ifndef TIE_RUNTIME_DISPLAY_SCREENSHOT_H
#define TIE_RUNTIME_DISPLAY_SCREENSHOT_H

#include "aeron/input.h"

/* Edge-detects the gamepad guide (Fn) button on the raw snapshot; call once
 * per frame before Aeron_Present(). Works with menus open and never consults
 * the config bindings. */
void TieScreenshot_Update(const AeronInputSnapshot* input);

/* Immediate capture of the classic framebuffer (Alt+O Landru hook path).
 * Returns 1 when a BMP was written, 0 otherwise. */
int TieScreenshot_CaptureNow(void);

#endif
