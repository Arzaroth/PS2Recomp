#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    // PS2X_PAD_SCRIPT="25:SQUARE,30:START,60.5:CROSS/300" presses buttons at the given times
    // (seconds since process start) for 500 ms or the given milliseconds, without a keyboard or
    // window focus.
    struct ScriptedPress
    {
        double time = 0.0;
        double hold = 0.5;
        uint16_t mask = 0;
    };

    const auto g_processStart = std::chrono::steady_clock::now();

    uint16_t scriptButton(const std::string &name)
    {
        static const struct { const char *name; uint16_t mask; } kButtons[] = {
            {"LEFT", PAD_LEFT}, {"DOWN", PAD_DOWN}, {"RIGHT", PAD_RIGHT}, {"UP", PAD_UP}, {"START", PAD_START},
            {"R3", PAD_R3}, {"L3", PAD_L3}, {"SELECT", PAD_SELECT}, {"SQUARE", PAD_SQUARE}, {"CROSS", PAD_CROSS},
            {"CIRCLE", PAD_CIRCLE}, {"TRIANGLE", PAD_TRIANGLE}, {"R1", PAD_R1}, {"L1", PAD_L1}, {"R2", PAD_R2},
            {"L2", PAD_L2}};
        for (const auto &button : kButtons)
            if (name == button.name)
                return button.mask;
        return 0;
    }

    std::vector<ScriptedPress> parsePadScript(const char *text)
    {
        std::vector<ScriptedPress> presses;
        if (!text)
            return presses;
        std::string entries(text);
        size_t start = 0;
        while (start < entries.size())
        {
            size_t end = entries.find(',', start);
            if (end == std::string::npos)
                end = entries.size();
            const std::string entry = entries.substr(start, end - start);
            const size_t colon = entry.find(':');
            if (colon != std::string::npos)
            {
                std::string button = entry.substr(colon + 1);
                double hold = 0.5;
                if (const size_t slash = button.find('/'); slash != std::string::npos)
                {
                    hold = std::atof(button.substr(slash + 1).c_str()) / 1000.0;
                    button.resize(slash);
                }
                presses.push_back({std::atof(entry.substr(0, colon).c_str()), hold, scriptButton(button)});
            }
            start = end + 1;
        }
        return presses;
    }

    uint16_t buttonsDownAt(const std::vector<ScriptedPress> &script, double seconds)
    {
        uint16_t down = 0;
        for (const ScriptedPress &press : script)
            if (seconds >= press.time && seconds < press.time + press.hold)
                down |= press.mask;
        return down;
    }

    uint16_t scriptedButtonsDown()
    {
        static const std::vector<ScriptedPress> script = parsePadScript(std::getenv("PS2X_PAD_SCRIPT"));
        if (script.empty())
            return 0;
        return buttonsDownAt(script, std::chrono::duration<double>(std::chrono::steady_clock::now() - g_processStart).count());
    }
}

uint16_t ps2PadScriptButtonsAt(const char *script, double seconds)
{
    return buttonsDownAt(parsePadScript(script), seconds);
}

bool PSPadBackend::readState(int /*port*/, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    uint16_t btns = 0xFFFFu;
    constexpr int kGamepad = 0;
    const bool useGamepad = IsGamepadAvailable(kGamepad);
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };

    if (useGamepad)
    {
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_UP))
            clearBit(PAD_UP);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN))
            clearBit(PAD_DOWN);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT))
            clearBit(PAD_LEFT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT))
            clearBit(PAD_RIGHT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN))
            clearBit(PAD_CROSS);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT))
            clearBit(PAD_CIRCLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT))
            clearBit(PAD_SQUARE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP))
            clearBit(PAD_TRIANGLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1))
            clearBit(PAD_L1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1))
            clearBit(PAD_R1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2))
            clearBit(PAD_L2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2))
            clearBit(PAD_R2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            clearBit(PAD_START);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_LEFT))
            clearBit(PAD_SELECT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_THUMB))
            clearBit(PAD_L3);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_THUMB))
            clearBit(PAD_R3);

        float lx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_X);
        float ly = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_Y);
        float rx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_X);
        float ry = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    else
    {
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
            clearBit(PAD_UP);
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
            clearBit(PAD_DOWN);
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
            clearBit(PAD_LEFT);
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
            clearBit(PAD_RIGHT);
        if (IsKeyDown(KEY_X) || IsKeyDown(KEY_SPACE))
            clearBit(PAD_CROSS);
        if (IsKeyDown(KEY_C) || IsKeyDown(KEY_ESCAPE))
            clearBit(PAD_CIRCLE);
        if (IsKeyDown(KEY_Z) || IsKeyDown(KEY_KP_0))
            clearBit(PAD_SQUARE);
        if (IsKeyDown(KEY_V) || IsKeyDown(KEY_KP_1))
            clearBit(PAD_TRIANGLE);
        if (IsKeyDown(KEY_Q))
            clearBit(PAD_L1);
        if (IsKeyDown(KEY_E))
            clearBit(PAD_R1);
        if (IsKeyDown(KEY_LEFT_SHIFT))
            clearBit(PAD_L2);
        if (IsKeyDown(KEY_RIGHT_SHIFT))
            clearBit(PAD_R2);
        if (IsKeyDown(KEY_ENTER))
            clearBit(PAD_START);
        if (IsKeyDown(KEY_TAB))
            clearBit(PAD_SELECT);
    }

    btns &= static_cast<uint16_t>(~scriptedButtonsDown());

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
