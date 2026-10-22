#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <cstddef>
#include <cstdint>

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    bool readState(int port, int slot, uint8_t *data, size_t size);
};

// Buttons a PS2X_PAD_SCRIPT string holds down at a time (seconds since process start), as a mask of
// pad button bits (1 = pressed).
uint16_t ps2PadScriptButtonsAt(const char *script, double seconds);

#endif
