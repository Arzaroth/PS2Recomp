#include "ps2_overlay_modules.h"

#include <cstring>
#include <iostream>

namespace ps2_overlays
{
    namespace
    {
        // Function-local statics: generated modules register during static
        // initialization, before any other global of this file is guaranteed to exist.
        std::vector<Module> &registry()
        {
            static std::vector<Module> modules;
            return modules;
        }

        ResidentResolver &resolverSlot()
        {
            static ResidentResolver resolver = nullptr;
            return resolver;
        }

        bool tableSlot(const Module &module, uint32_t address, uint32_t &slot)
        {
            if ((address & 3u) != 0u || address < module.tableBase || address >= module.tableEnd)
            {
                return false;
            }
            slot = (address - module.tableBase) >> 2;
            return slot < module.slotCount;
        }

        std::vector<const Module *> candidatesFor(uint32_t address)
        {
            std::vector<const Module *> candidates;
            for (const Module &module : registry())
            {
                uint32_t slot = 0u;
                if (tableSlot(module, address, slot))
                {
                    candidates.push_back(&module);
                }
            }
            return candidates;
        }
    }

    void registerModule(const Module &module)
    {
        for (const Module &existing : registry())
        {
            if (std::strcmp(existing.name, module.name) == 0)
            {
                std::cerr << "[overlay] duplicate module registration: " << module.name << std::endl;
                return;
            }
        }
        registry().push_back(module);
    }

    void clearModules()
    {
        registry().clear();
        resolverSlot() = nullptr;
    }

    const std::vector<Module> &modules()
    {
        return registry();
    }

    const Module *findModule(const char *name)
    {
        for (const Module &module : registry())
        {
            if (std::strcmp(module.name, name) == 0)
            {
                return &module;
            }
        }
        return nullptr;
    }

    const Module *guardResolver(const uint8_t *rdram, uint32_t address,
                                const std::vector<const Module *> &candidates)
    {
        (void)address;
        if (rdram == nullptr)
        {
            return nullptr;
        }
        for (const Module *module : candidates)
        {
            const uint32_t offset = module->loadAddress & PS2_RAM_MASK;
            if (module->guard == nullptr || module->guardSize == 0u ||
                offset + module->guardSize > PS2_RAM_SIZE)
            {
                continue;
            }
            if (std::memcmp(rdram + offset, module->guard, module->guardSize) == 0)
            {
                return module;
            }
        }
        return nullptr;
    }

    void setResidentResolver(ResidentResolver resolver)
    {
        resolverSlot() = resolver;
    }

    bool coversAddress(uint32_t address)
    {
        for (const Module &module : registry())
        {
            uint32_t slot = 0u;
            if (tableSlot(module, address, slot))
            {
                return true;
            }
        }
        return false;
    }

    const Module *residentModuleFor(const uint8_t *rdram, uint32_t address)
    {
        const std::vector<const Module *> candidates = candidatesFor(address);
        if (candidates.empty())
        {
            return nullptr;
        }
        const ResidentResolver resolver = resolverSlot() ? resolverSlot() : &guardResolver;
        return resolver(rdram, address, candidates);
    }

    PS2Runtime::RecompiledFunction *residentEntry(const uint8_t *rdram, uint32_t address)
    {
        const Module *module = residentModuleFor(rdram, address);
        uint32_t slot = 0u;
        if (module == nullptr || !tableSlot(*module, address, slot))
        {
            return nullptr;
        }
        return &module->table[slot];
    }

    bool replaceModuleFunction(const char *moduleName, uint32_t address, PS2Runtime::RecompiledFunction func)
    {
        const Module *module = findModule(moduleName);
        uint32_t slot = 0u;
        if (module == nullptr || !tableSlot(*module, address, slot))
        {
            std::cerr << "[overlay] cannot replace 0x" << std::hex << address << std::dec
                      << " in module " << moduleName << std::endl;
            return false;
        }
        module->table[slot] = func;
        return true;
    }
}
