#ifndef PS2_OVERLAY_MODULES_H
#define PS2_OVERLAY_MODULES_H

#include "ps2_runtime.h"

#include <cstdint>
#include <vector>

// Code modules (overlays) recompiled separately from the main ELF with
// `module_name` set in the recompiler config. Several modules may share a load
// address; the runtime picks the resident one at dispatch time by comparing the
// module's guard bytes with guest memory at its load address.
namespace ps2_overlays
{
    struct Module
    {
        const char *name = "";
        uint32_t loadAddress = 0;  // guest address of the module image (guard is read here)
        uint32_t tableBase = 0;    // dense function table covers [tableBase, tableEnd)
        uint32_t tableEnd = 0;
        uint32_t slotCount = 0;
        PS2Runtime::RecompiledFunction *table = nullptr;
        const uint8_t *guard = nullptr; // bytes expected at loadAddress while resident
        uint32_t guardSize = 0;
    };

    // Picks the resident module among candidates whose table covers `address`.
    // Return nullptr when none is resident.
    using ResidentResolver = const Module *(*)(const uint8_t *rdram, uint32_t address,
                                               const std::vector<const Module *> &candidates);

    // Called from generated registration code during static initialization.
    void registerModule(const Module &module);
    // Drops every registered module and restores the default resolver (tests).
    void clearModules();

    const std::vector<Module> &modules();
    const Module *findModule(const char *name);

    // Default: first candidate whose guard bytes match guest memory.
    const Module *guardResolver(const uint8_t *rdram, uint32_t address,
                                const std::vector<const Module *> &candidates);
    // Game overrides may install a cheaper or more specific resolver; nullptr restores the default.
    void setResidentResolver(ResidentResolver resolver);

    bool coversAddress(uint32_t address);
    const Module *residentModuleFor(const uint8_t *rdram, uint32_t address);
    // Table entry for `address` in the resident module, or nullptr.
    PS2Runtime::RecompiledFunction *residentEntry(const uint8_t *rdram, uint32_t address);

    // Replace a function of a specific module, resident or not (for game overrides).
    bool replaceModuleFunction(const char *moduleName, uint32_t address, PS2Runtime::RecompiledFunction func);
}

#endif // PS2_OVERLAY_MODULES_H
