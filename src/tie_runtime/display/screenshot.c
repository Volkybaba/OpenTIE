/* Flight screenshots via GPU readback (VolkyTFColl v4).
 *
 * The two retail capture paths are stubs in this build
 * (FrontendDisplay_CaptureScreenshot returns DX_DD_OK immediately;
 * rtsvga2_takeScreenshot returns 0 — the PCX writer was never recovered),
 * and this aeron has no Aeron_VideoCopyCurrentFrame. Capture therefore reads
 * the pixels from wherever the current image actually lives:
 *
 *   - Flight, HD (remaster) view: the 3D scene never touches the classic
 *     framebuffer — it is rendered into the offscreen RGBA16F flight present
 *     target (RenderFlightTarget runs in both presentation branches, before
 *     the direct-to-swapchain submit). Read it back with
 *     Aeron_ReadRenderTargetRawPixels (synchronous fence wait, ~1 frame
 *     stall), apply the live on-screen tonemap (v6: AgX log map / ACES
 *     fit, parameters read from the present controls), convert to sRGB,
 *     and write a BMP. The v2/v3 classic-buffer read showed only the 2D
 *     cockpit over a black viewport here; v4 read the target but applied
 *     only the sRGB EOTF and crushed the shadows.
 *   - Flight, classic (Tab) view: the DX5 render-target surface that was
 *     last presented, read back through
 *     AeronDx5_ReadbackPresentedFlightFrame.
 *   - Frontend / cutscene: the classic 8-bit paletted framebuffer, as in v2.
 *
 * Either GPU path failing falls back to the classic framebuffer (v2/v3
 * behavior) instead of producing nothing.
 *
 * Trigger: the gamepad guide (Fn) button, edge-detected straight from the
 * raw AeronInputSnapshot — no config binding is consulted, so it works with
 * menus open. Alt+O keeps working through the same capture function.
 *
 * Files land in the user directory (conf/TotallyOpen/OpenTIE/):
 * frontscreen%d.bmp (frontend), moviescreen%d.bmp (cutscene),
 * flightscreen%d.bmp (flight).
 */

#include "tie_runtime/display/screenshot.h"

#include "aeron/aeron.h"
#include "aeron/compat/host.h"
#include "aeron/paths.h"
#include "aeron/render.h"
#include "aeron/scene/present.h"
#include "aeron/vfs.h"
#include "tie/frontend_display_tie98.h"
#include "tie_remaster/remaster.h"
#include "tie_runtime/display/classic_framebuffer.h"
#include "tie_runtime/snapshot/snapshot.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool s_guide_down;

/* Same conversion as aeron/tools/scene_capture/capture.c: the RGBA16F
 * present target holds linear HDR values; the BMP preview is sRGB. */
static float Screenshot_HalfToFloat(uint16_t half) {
	uint32_t exponent = (half >> 10) & 31u, mantissa = half & 1023u;
	if (!exponent) {
		float value = mantissa ? ldexpf((float)mantissa, -24) : 0;
		return half & 0x8000u ? -value : value;
	}
	uint32_t bits = ((uint32_t)(half & 0x8000u) << 16) |
			(exponent == 31 ? 0x7f800000u : (exponent + 112u) << 23) | (mantissa << 13);
	float result;
	memcpy(&result, &bits, sizeof result);
	return result;
}

static uint8_t Screenshot_LinearToSrgb8(float value) {
	float srgb;
	if (!(value > 0.0f))
		return 0;
	if (value > 1.0f)
		value = 1.0f;
	srgb = value <= 0.0031308f ? 12.92f * value : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
	if (srgb < 0.0f)
		srgb = 0.0f;
	if (srgb > 1.0f)
		srgb = 1.0f;
	return (uint8_t)(srgb * 255.0f + 0.5f);
}

/* VolkyTFColl v11: inverse of Screenshot_LinearToSrgb8 (piecewise sRGB
 * curve). The on-screen classic layer applies the user brightness in
 * linear space, so the palette brightness below decodes to linear,
 * applies the same curve, and re-encodes — matching the screen.
 * VolkyTFColl v15: the curve is a gamma lift now (v14's linear factor
 * barely lifted the shadows): out_linear = powf(in_linear, 1.0f / b).
 * Same curve as DDShim_BrightenRetainedView and the HD flight path. */
static float Screenshot_Srgb8ToLinear(uint8_t value) {
	float srgb = (float)value / 255.0f;
	return srgb <= 0.04045f ? srgb / 12.92f : powf((srgb + 0.055f) / 1.055f, 2.4f);
}

static uint8_t Screenshot_BrightenedPaletteByte(uint8_t srgb, float brightness) {
	if (brightness == 1.0f)
		return srgb;
	/* VolkyTFColl v15: gamma lift (see the comment above) — same curve
	 * as the on-screen CPU path, so the screenshot keeps matching the
	 * screen. powf of 0 is 0; the decode never yields a negative. */
	return Screenshot_LinearToSrgb8(powf(Screenshot_Srgb8ToLinear(srgb), 1.0f / brightness));
}

/* VolkyTFColl v6: apply the on-screen tonemap to the HD flight capture.
 * The present target holds linear HDR; v4 applied only the sRGB EOTF, which
 * crushed the shadows the AgX log map lifts (space at linear ~0.0015 reads
 * ~8/255 instead of ~33/255 on screen). Formulas below are verbatim copies
 * of aeron/shaders/scene_tonemap.frag.hlsl + scene_tonemap_agx.hlsli
 * (HLSL mul(M, v) = row-major M times column vector v), with one
 * evidence-driven deviation: the AgX helper does NOT apply the final
 * pow(color, eotf). Numeric check of the shader chain against the measured
 * screen bytes (host test, same constants): linear 0.0015 -> plain sRGB 5
 * (v4: ~5-8), pow-2.2 + sRGB encode 0, sigmoid-only + sRGB encode 31
 * (measured screen ~33). The pow step + store encode double-encodes and
 * would darken the capture below v4; the screen matches the sigmoid
 * output sRGB-encoded. The EOTF exponent is still read live (see below)
 * but not applied. Bloom is not in the target and is not added; the fade
 * tint is (1,1,1,1) and PMA coverage is 1 for a full-frame present, so
 * neither is replicated. No dither step: the shader's encode/decode
 * round-trip is net identity. */

static float Screenshot_Clampf(float v, float lo, float hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

static uint8_t Screenshot_DisplayByteDirect(float value) {
	if (!(value > 0.0f))
		return 0;
	if (value > 1.0f)
		value = 1.0f;
	return (uint8_t)(value * 255.0f + 0.5f);
}

static void Screenshot_MulMat3(const float m[3][3], const float v[3], float out[3]) {
	int i;
	for (i = 0; i < 3; ++i)
		out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
}

static void Screenshot_RrtOdtFit(const float v[3], float out[3]) {
	int i;
	for (i = 0; i < 3; ++i) {
		float a = v[i] * (v[i] + 0.0245786f) - 0.000090537f;
		float b = v[i] * (0.983729f * v[i] + 0.4329510f) + 0.238081f;
		out[i] = a / b;
	}
}

/* ACES Fitted (Stephen Hill 2017). Output is LINEAR sRGB display-referred. */
static void Screenshot_TonemapAces(float rgb[3], float pre_exposure) {
	static const float input_matrix[3][3] = {
		{ 0.59719f, 0.35458f, 0.04823f },
		{ 0.07600f, 0.90834f, 0.01566f },
		{ 0.02840f, 0.13383f, 0.83777f },
	};
	static const float output_matrix[3][3] = {
		{ 1.60475f, -0.53108f, -0.07367f },
		{ -0.10208f, 1.10813f, -0.00605f },
		{ -0.00327f, -0.07276f, 1.07602f },
	};
	float tmp[3];
	int i;
	/* Scene exposure is the v11 user brightness (present.c u[4]); it is
	 * applied once, before the operator, in Screenshot_CaptureFlightHd —
	 * matching the shader's (scene + bloom) * exposure. Pre-exposure
	 * lifts the dark toe before the fit. */
	for (i = 0; i < 3; ++i)
		rgb[i] = rgb[i] * pre_exposure;
	for (i = 0; i < 3; ++i)
		rgb[i] = rgb[i] > 0.0f ? rgb[i] : 0.0f;
	Screenshot_MulMat3(input_matrix, rgb, tmp);
	Screenshot_RrtOdtFit(tmp, rgb);
	Screenshot_MulMat3(output_matrix, rgb, tmp);
	for (i = 0; i < 3; ++i)
		rgb[i] = Screenshot_Clampf(tmp[i], 0.0f, 1.0f);
}

static void Screenshot_AgxContrast(const float x[3], float out[3]) {
	int i;
	for (i = 0; i < 3; ++i) {
		float x2 = x[i] * x[i];
		float x4 = x2 * x2;
		out[i] = 15.5f * x4 * x2 - 40.14f * x4 * x[i] + 31.96f * x4
			- 6.868f * x2 * x[i] + 0.4298f * x2 + 0.1191f * x[i] - 0.00232f;
	}
}

static void Screenshot_AgxApplyLook(float color[3], float look, float power, float saturation) {
	float luma;
	int i;
	if (look < 0.5f)
		return;
	for (i = 0; i < 3; ++i)
		color[i] = powf(color[i] > 0.0f ? color[i] : 0.0f, power);
	luma = color[0] * 0.2126f + color[1] * 0.7152f + color[2] * 0.0722f;
	for (i = 0; i < 3; ++i)
		color[i] = luma + saturation * (color[i] - luma);
}

/* AgX parametric (scene_tonemap_agx.hlsli AeronAgxToneMap) up to the outset
 * matrix. The shader's final pow(color, eotf) is deliberately NOT applied
 * (see the v6 note above): the measured screen bytes match the sigmoid
 * output sRGB-encoded, and the pow would crush space to byte 0-1. The
 * returned value is therefore the linear display-referred value, exactly
 * like the ACES branch. */
static void Screenshot_TonemapAgx(float rgb[3], float look, float power, float saturation) {
	static const float inset_matrix[3][3] = {
		{ 0.842479062253094f, 0.0784335999999992f, 0.0792237451477643f },
		{ 0.0423282422610123f, 0.878468636469772f, 0.0784336f },
		{ 0.0423756549057051f, 0.0784336f, 0.879142973793104f },
	};
	static const float outset_matrix[3][3] = {
		{ 1.19687900512017f, -0.0980208811401368f, -0.0990297440797205f },
		{ -0.0528968517574562f, 1.15190312990417f, -0.0989611768433f },
		{ -0.0529716355144438f, -0.0980434501171241f, 1.15107367264116f },
	};
	float tmp[3];
	int i;
	for (i = 0; i < 3; ++i)
		rgb[i] = rgb[i] > 0.0f ? rgb[i] : 0.0f;
	Screenshot_MulMat3(inset_matrix, rgb, tmp);
	for (i = 0; i < 3; ++i) {
		float v = tmp[i] > 1e-10f ? tmp[i] : 1e-10f;
		rgb[i] = log2f(v);
	}
	for (i = 0; i < 3; ++i)
		rgb[i] = Screenshot_Clampf((rgb[i] - (-12.47393f)) / (4.026069f - (-12.47393f)),
					    0.0f, 1.0f);
	Screenshot_AgxContrast(rgb, tmp);
	for (i = 0; i < 3; ++i)
		rgb[i] = Screenshot_Clampf(tmp[i], 0.0f, 1.0f);
	Screenshot_AgxApplyLook(rgb, look, power, saturation);
	Screenshot_MulMat3(outset_matrix, rgb, tmp);
	for (i = 0; i < 3; ++i)
		rgb[i] = tmp[i];
}

/* Apply the live on-screen tonemap operator to one linear HDR pixel. The
 * operator id and all parameters are read live from the present controls
 * (Cmd+T can change the operator at runtime); nothing is hardcoded. Both
 * branches return a linear display-referred value. */
static void Screenshot_TonemapPixel(float rgb[3], int op, float look, float power,
				    float saturation, float aces_exposure) {
	if (op == AERON_SCENE_TONEMAP_AGX_PARAMETRIC)
		Screenshot_TonemapAgx(rgb, look, power, saturation);
	else
		Screenshot_TonemapAces(rgb, aces_exposure);
}

static bool Screenshot_GuideDown(const AeronInputSnapshot* input) {
	int i;

	if (!input)
		return false;
	for (i = 0; i < AERON_CONTROLLER_MAX; ++i) {
		if (input->controllers[i].connected &&
		    (input->controllers[i].gamepad_buttons & (1u << AERON_GAMEPAD_BUTTON_GUIDE)))
			return true;
	}
	return false;
}

void TieScreenshot_Update(const AeronInputSnapshot* input) {
	bool down = Screenshot_GuideDown(input);

	if (down && !s_guide_down)
		(void)TieScreenshot_CaptureNow();
	s_guide_down = down;
}

/* HD flight: read the RGBA16F present target, convert to sRGB RGB565, write
 * through the existing 16bpp BMP writer. Returns 0 on any failure so the
 * caller can fall back to the classic framebuffer. */
static int Screenshot_CaptureFlightHd(const char* full_path, const TieSnapshot* snapshot) {
	AeronRenderTarget* target = TieRemaster_GetFlightPresentTarget(snapshot);
	AeronTexture* texture;
	int width, height;
	size_t pixels, i;
	uint16_t* staging = NULL;
	uint16_t* rgb565 = NULL;
	int ok = 0;

	if (!target)
		return 0;
	texture = Aeron_RenderTargetGetTexture(target);
	if (!texture || Aeron_TextureGetFormat(texture) != AERON_TEXTURE_FORMAT_RGBA16_FLOAT)
		return 0;
	width = Aeron_TextureGetWidth(texture);
	height = Aeron_TextureGetHeight(texture);
	if (width <= 0 || height <= 0)
		return 0;
	pixels = (size_t)width * (size_t)height;
	staging = (uint16_t*)malloc(pixels * 8);
	rgb565 = (uint16_t*)malloc(pixels * 2);
	if (!staging || !rgb565)
		goto done;
	if (!Aeron_ReadRenderTargetRawPixels(target, staging, pixels * 8, (size_t)width * 8))
		goto done;
	{
		/* Live tonemap state (operator + parameters); the final encoding
		 * depends on the swapchain format: an _SRGB swapchain store
		 * applies the sRGB encode, so both operators go through
		 * LinearToSrgb8; on a _UNORM swapchain the AgX output is written
		 * as-is, while linear ACES output still needs the sRGB encode.
		 * The EOTF exponent accessor exists but is deliberately not read
		 * here: the AgX pow step is omitted (see the v6 note above). */
		int tonemap_op = AeronScenePresent_TonemapOp();
		float agx_look = (float)AeronScenePresent_AgxLook();
		float agx_power = AeronScenePresent_AgxPunchyPower();
		float agx_sat = AeronScenePresent_AgxPunchySaturation();
		float aces_exp = AeronScenePresent_AcesExposure();
		/* VolkyTFColl v11: user brightness = scene exposure. The shader
		 * multiplies (scene + bloom) by it in linear space BEFORE the
		 * tonemap operator, so do the same here — before TonemapPixel,
		 * for both operators. VolkyTFColl v15: the curve is a gamma lift
		 * now (out_linear = powf(in_linear, 1.0f / b)), same as the
		 * on-screen CPU path and the palette path — the linear factor
		 * barely lifted the shadows. Guarded for non-positive values
		 * (powf of a negative is NaN). */
		float brightness = AeronScenePresent_Brightness();
		AeronTextureFormat swap_format = Aeron_SwapchainFormat();
		bool swap_srgb = swap_format == AERON_TEXTURE_FORMAT_RGBA8_SRGB ||
				 swap_format == AERON_TEXTURE_FORMAT_BGRA8_SRGB;
		bool agx_direct = tonemap_op == AERON_SCENE_TONEMAP_AGX_PARAMETRIC && !swap_srgb;
		for (i = 0; i < pixels; ++i) {
			float rgb[3];
			uint8_t r, g, b;
			rgb[0] = Screenshot_HalfToFloat(staging[i * 4 + 0]);
			rgb[1] = Screenshot_HalfToFloat(staging[i * 4 + 1]);
			rgb[2] = Screenshot_HalfToFloat(staging[i * 4 + 2]);
			if (brightness != 1.0f) {
				float inv_brightness = 1.0f / brightness;
				if (rgb[0] > 0.0f)
					rgb[0] = powf(rgb[0], inv_brightness);
				if (rgb[1] > 0.0f)
					rgb[1] = powf(rgb[1], inv_brightness);
				if (rgb[2] > 0.0f)
					rgb[2] = powf(rgb[2], inv_brightness);
			}
			Screenshot_TonemapPixel(rgb, tonemap_op, agx_look, agx_power, agx_sat, aces_exp);
			if (agx_direct) {
				r = Screenshot_DisplayByteDirect(rgb[0]);
				g = Screenshot_DisplayByteDirect(rgb[1]);
				b = Screenshot_DisplayByteDirect(rgb[2]);
			} else {
				r = Screenshot_LinearToSrgb8(rgb[0]);
				g = Screenshot_LinearToSrgb8(rgb[1]);
				b = Screenshot_LinearToSrgb8(rgb[2]);
			}
			rgb565[i] = (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
		}
	}
	ok = Bitmap_WriteBmp24(full_path, rgb565, width, height, width * 2, 16, 0, NULL);
done:
	free(staging);
	free(rgb565);
	return ok;
}

/* VolkyTFColl v12: the on-screen classic flight frame is the DX5 texture
 * layer, which multiplies the v11 brightness tint in linear space — so the
 * read-back buffer gets the same per-channel treatment as the palette
 * path (decode to linear, scale, re-encode). RGB565 and RGB555. */
static void Screenshot_BrightenDx5Buffer(uint16_t* buffer, int width, int height, AeronPixelFormat format,
					 float brightness) {
	int x, y;
	if (!buffer || width <= 0 || height <= 0 || brightness == 1.0f)
		return;
	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			uint16_t px = buffer[(size_t)y * (size_t)width + (size_t)x];
			uint8_t r8, g8, b8;
			if (format == AERON_PIXEL_FORMAT_RGB555) {
				r8 = (uint8_t)(((px >> 10) & 31u) << 3 | ((px >> 10) & 31u) >> 2);
				g8 = (uint8_t)(((px >> 5) & 31u) << 3 | ((px >> 5) & 31u) >> 2);
				b8 = (uint8_t)((px & 31u) << 3 | (px & 31u) >> 2);
				r8 = Screenshot_BrightenedPaletteByte(r8, brightness);
				g8 = Screenshot_BrightenedPaletteByte(g8, brightness);
				b8 = Screenshot_BrightenedPaletteByte(b8, brightness);
				buffer[(size_t)y * (size_t)width + (size_t)x] =
					(uint16_t)(((r8 >> 3) << 10) | ((g8 >> 3) << 5) | (b8 >> 3));
			} else {
				r8 = (uint8_t)(((px >> 11) & 31u) << 3 | ((px >> 11) & 31u) >> 2);
				g8 = (uint8_t)(((px >> 5) & 63u) << 2 | ((px >> 5) & 63u) >> 4);
				b8 = (uint8_t)((px & 31u) << 3 | (px & 31u) >> 2);
				r8 = Screenshot_BrightenedPaletteByte(r8, brightness);
				g8 = Screenshot_BrightenedPaletteByte(g8, brightness);
				b8 = Screenshot_BrightenedPaletteByte(b8, brightness);
				buffer[(size_t)y * (size_t)width + (size_t)x] =
					(uint16_t)(((r8 >> 3) << 11) | ((g8 >> 2) << 5) | (b8 >> 3));
			}
		}
	}
}

/* Classic flight: read back the last presented DX5 render-target surface. */
#define SCREENSHOT_DX5_MAX_WIDTH 1920
#define SCREENSHOT_DX5_MAX_HEIGHT 1200

static int Screenshot_CaptureFlightDx5(const char* full_path) {
	uint16_t* buffer = (uint16_t*)malloc((size_t)SCREENSHOT_DX5_MAX_WIDTH * SCREENSHOT_DX5_MAX_HEIGHT * 2);
	int width = 0, height = 0;
	AeronPixelFormat format = AERON_PIXEL_FORMAT_UNKNOWN;
	int ok;

	if (!buffer)
		return 0;
	ok = AeronDx5_ReadbackPresentedFlightFrame(buffer, SCREENSHOT_DX5_MAX_WIDTH * 2, &width, &height, &format);
	if (ok)
		Screenshot_BrightenDx5Buffer(buffer, width, height, format, AeronScenePresent_Brightness());
	if (ok)
		ok = Bitmap_WriteBmp24(full_path, buffer, width, height, SCREENSHOT_DX5_MAX_WIDTH * 2, 16,
				       format == AERON_PIXEL_FORMAT_RGB555, NULL);
	free(buffer);
	return ok;
}

int TieScreenshot_CaptureNow(void) {
	const TieSnapshot* snapshot = TieSnapshot_Current();
	const TieFramebuffer* framebuffer;
	const char* prefix;
	bool is_flight;
	char name[64];
	char full_path[512];
	uint8_t palette_bgra[256 * 4];
	AeronVfs* vfs;
	const char* user_root;
	int sequence;
	int i;

	is_flight = snapshot && (snapshot->scene_kind == TIE_SCENE_FLIGHT ||
				 snapshot->scene_kind == TIE_SCENE_FLIGHT_LOADING);
	if (is_flight)
		prefix = "flightscreen";
	else if (snapshot && snapshot->scene_kind == TIE_SCENE_CUTSCENE)
		prefix = "moviescreen";
	else
		prefix = "frontscreen";

	vfs = Aeron_GetVfs();
	for (sequence = 0; sequence < 10000; ++sequence) {
		snprintf(name, sizeof name, "%s%d.bmp", prefix, sequence);
		if (!vfs || !AeronVfs_Exists(vfs, AERON_VFS_ROOT_USER, name))
			break;
	}
	if (sequence >= 10000)
		return 0;

	user_root = Aeron_UserPath();
	if (user_root && user_root[0]) {
		size_t len = strlen(user_root);
		snprintf(full_path, sizeof full_path, "%s%s%s", user_root,
			 (len > 0 && user_root[len - 1] == '/') ? "" : "/", name);
	} else {
		snprintf(full_path, sizeof full_path, "%s", name);
	}

	if (is_flight && snapshot) {
		/* HD: 3D lives only on the GPU present target; classic: the DX5
		 * presented frame. Either failing falls through to the classic
		 * framebuffer below (v2/v3 behavior). */
		if (TieRemaster_SuppressesClassicFlight(snapshot)) {
			if (Screenshot_CaptureFlightHd(full_path, snapshot)) {
				Aeron_LogInfo("tie.screenshot", "Saved %s", name);
				return 1;
			}
		} else if (Screenshot_CaptureFlightDx5(full_path)) {
			Aeron_LogInfo("tie.screenshot", "Saved %s", name);
			return 1;
		}
	}

	framebuffer = TieClassicFramebuffer_PresentedVga();
	if (!framebuffer)
		framebuffer = TieClassicFramebuffer_Current();
	if (!framebuffer || !framebuffer->pixels || framebuffer->width <= 0 ||
	    framebuffer->height <= 0 || !framebuffer->palette)
		return 0;

	/* The classic palette is 256 6-bit RGB triplets (values 0-63); the
	 * screen side scales them with << 2 (FrontendDisplay_UpdatePalette),
	 * so do the same here — max 252, matching the display exactly. The
	 * BMP writer wants BGRA quads. VolkyTFColl v11: the on-screen classic
	 * layer also multiplies the v11 brightness tint in linear space, so
	 * the palette entries get the same treatment here. */
	{
		float brightness = AeronScenePresent_Brightness();
		for (i = 0; i < 256; ++i) {
			palette_bgra[4 * i + 0] =
				Screenshot_BrightenedPaletteByte((uint8_t)(framebuffer->palette[3 * i + 2] << 2), brightness);
			palette_bgra[4 * i + 1] =
				Screenshot_BrightenedPaletteByte((uint8_t)(framebuffer->palette[3 * i + 1] << 2), brightness);
			palette_bgra[4 * i + 2] =
				Screenshot_BrightenedPaletteByte((uint8_t)(framebuffer->palette[3 * i + 0] << 2), brightness);
			palette_bgra[4 * i + 3] = 0;
		}
	}

	if (!Bitmap_WriteBmp24(full_path, framebuffer->pixels, framebuffer->width,
			       framebuffer->height, framebuffer->pitch, 8, 0, palette_bgra))
		return 0;

	Aeron_LogInfo("tie.screenshot", "Saved %s", name);
	return 1;
}
