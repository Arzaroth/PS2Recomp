#include "MiniTest.h"
#include "ps2_overlay_modules.h"
#include "ps2_runtime.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/config_manager.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <string>

using namespace ps2recomp;

namespace
{
    // Two overlay modules sharing one slot, as in games that swap code into a
    // fixed address. Addresses sit high in RAM, away from other runtime tests.
    constexpr uint32_t kSlot = 0x01F00000u;
    constexpr uint32_t kEntry = kSlot + 0x80u;

    const uint8_t kGuardA[8] = {'M', 'O', 'D', 'A', 1, 0, 0, 0};
    const uint8_t kGuardB[8] = {'M', 'O', 'D', 'B', 2, 0, 0, 0};

    PS2Runtime::RecompiledFunction g_tableA[64] = {};
    PS2Runtime::RecompiledFunction g_tableB[64] = {};

    void setReturn(R5900Context *ctx, uint32_t value)
    {
        ctx->r[2] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    uint32_t getReturn(const R5900Context &ctx)
    {
        return static_cast<uint32_t>(_mm_cvtsi128_si32(ctx.r[2]));
    }

    void moduleAFunction(uint8_t *, R5900Context *ctx, PS2Runtime *) { setReturn(ctx, 0xAu); }
    void moduleBFunction(uint8_t *, R5900Context *ctx, PS2Runtime *) { setReturn(ctx, 0xBu); }
    void replacementFunction(uint8_t *, R5900Context *ctx, PS2Runtime *) { setReturn(ctx, 0xCu); }
    void mainFunction(uint8_t *, R5900Context *ctx, PS2Runtime *) { setReturn(ctx, 0xEu); }

    ps2_overlays::Module makeModule(const char *name, PS2Runtime::RecompiledFunction *table, const uint8_t *guard)
    {
        ps2_overlays::Module module;
        module.name = name;
        module.loadAddress = kSlot;
        module.tableBase = kSlot;
        module.tableEnd = kSlot + 64u * 4u;
        module.slotCount = 64u;
        module.table = table;
        module.guard = guard;
        module.guardSize = 8u;
        return module;
    }

    void registerTwoModules()
    {
        ps2_overlays::clearModules();
        std::fill(std::begin(g_tableA), std::end(g_tableA), nullptr);
        std::fill(std::begin(g_tableB), std::end(g_tableB), nullptr);
        g_tableA[(kEntry - kSlot) >> 2] = &moduleAFunction;
        g_tableB[(kEntry - kSlot) >> 2] = &moduleBFunction;
        ps2_overlays::registerModule(makeModule("modA", g_tableA, kGuardA));
        ps2_overlays::registerModule(makeModule("modB", g_tableB, kGuardB));
    }

    uint32_t callAt(PS2Runtime &runtime, uint32_t address)
    {
        R5900Context ctx{};
        ctx.pc = address;
        runtime.lookupFunction(address)(runtime.memory().getRDRAM(), &ctx, &runtime);
        return getReturn(ctx);
    }

    const ps2_overlays::Module *alwaysSecond(const uint8_t *, uint32_t,
                                             const std::vector<const ps2_overlays::Module *> &candidates)
    {
        return candidates.size() > 1 ? candidates[1] : nullptr;
    }

    std::filesystem::path writeTempConfig(const std::string &body)
    {
        const auto suffix = std::to_string(
            static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
        const std::filesystem::path path =
            std::filesystem::temp_directory_path() / ("ps2recomp-module-" + suffix + ".toml");
        std::ofstream file(path);
        file << "[general]\ninput = \"dummy.elf\"\noutput = \"out\"\n" << body;
        return path;
    }
}

void register_ps2_overlay_module_tests()
{
    MiniTest::Case("OverlayModules", [](TestCase &tc)
                   {
        tc.Run("resident module is chosen by its guard bytes in guest memory", [](TestCase &t) {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::Stop);
            registerTwoModules();
            uint8_t *rdram = runtime.memory().getRDRAM();

            std::memcpy(rdram + kSlot, kGuardB, sizeof(kGuardB));
            t.IsTrue(runtime.hasFunction(kEntry), "module B should be resident");
            t.Equals(callAt(runtime, kEntry), 0xBu, "dispatch should reach module B");

            std::memcpy(rdram + kSlot, kGuardA, sizeof(kGuardA));
            t.Equals(callAt(runtime, kEntry), 0xAu, "swapping the image should switch dispatch to module A");

            std::memset(rdram + kSlot, 0xFF, sizeof(kGuardA));
            t.IsFalse(runtime.hasFunction(kEntry), "no module should be resident over foreign bytes");
            t.Equals(ps2_overlays::residentModuleFor(rdram, kEntry), static_cast<const ps2_overlays::Module *>(nullptr),
                     "resident lookup should report no module");
            ps2_overlays::clearModules();
        });

        tc.Run("a custom resolver replaces guard matching", [](TestCase &t) {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            registerTwoModules();
            std::memcpy(runtime.memory().getRDRAM() + kSlot, kGuardA, sizeof(kGuardA));

            ps2_overlays::setResidentResolver(&alwaysSecond);
            t.Equals(callAt(runtime, kEntry), 0xBu, "the installed resolver should pick module B despite A's guard");
            ps2_overlays::setResidentResolver(nullptr);
            t.Equals(callAt(runtime, kEntry), 0xAu, "clearing the resolver should restore guard matching");
            ps2_overlays::clearModules();
        });

        tc.Run("replaceFunction targets the resident module only", [](TestCase &t) {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            registerTwoModules();
            uint8_t *rdram = runtime.memory().getRDRAM();
            std::memcpy(rdram + kSlot, kGuardA, sizeof(kGuardA));

            t.IsTrue(runtime.replaceFunction(kEntry, &replacementFunction), "replacing in the resident module should succeed");
            t.Equals(callAt(runtime, kEntry), 0xCu, "module A should run the replacement");

            std::memcpy(rdram + kSlot, kGuardB, sizeof(kGuardB));
            t.Equals(callAt(runtime, kEntry), 0xBu, "module B must keep its own function");

            t.IsTrue(ps2_overlays::replaceModuleFunction("modB", kEntry, &replacementFunction),
                     "replacing by module name should succeed");
            t.Equals(callAt(runtime, kEntry), 0xCu, "module B should now run the replacement");
            t.IsFalse(ps2_overlays::replaceModuleFunction("missing", kEntry, &replacementFunction),
                      "unknown module names should be rejected");
            ps2_overlays::clearModules();
        });

        tc.Run("main ELF functions keep priority over overlay modules", [](TestCase &t) {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            registerTwoModules();
            std::memcpy(runtime.memory().getRDRAM() + kSlot, kGuardA, sizeof(kGuardA));
            const uint32_t shared = kEntry + 4u;
            g_tableA[(shared - kSlot) >> 2] = &moduleAFunction;

            // The unit-test main table spans all of RAM; write it directly as generated
            // main ELF registration would (replaceFunction would target the resident module).
            PS2Runtime::RecompiledFunction &mainSlot = g_ps2RecompiledFunctionTable[shared >> 2];
            mainSlot = &mainFunction;
            t.Equals(callAt(runtime, shared), 0xEu, "a main table entry should win over the resident module");
            mainSlot = nullptr;
            t.Equals(callAt(runtime, shared), 0xAu, "an empty main slot should fall back to the resident module");
            ps2_overlays::clearModules();
        });

        tc.Run("module mode scopes the registration source and headers", [](TestCase &t) {
            Function func;
            func.name = "FUN_00001080";
            func.start = 0x1080;
            func.end = 0x1084;
            func.isRecompiled = true;

            CodeGenerator gen({}, {});
            gen.setRenamedFunctions({{func.start, "mod7_FUN_00001080_0x1080"}});
            CodeGenerator::ModuleInfo info;
            info.name = "mod7";
            info.loadAddress = 0x1000;
            info.guard = {0x00, 0x07, 0xAB, 0xCD};
            gen.setModuleInfo(info);

            Instruction nop{};
            nop.address = func.start;
            nop.opcode = OPCODE_ADDIU;
            const std::string generated = gen.generateFunction(func, {nop}, true);
            const std::string registration = gen.generateFunctionRegistration({func}, {});

            t.IsTrue(generated.find("#include <mod7_functions.h>") != std::string::npos,
                     "function sources should include the module header");
            t.IsTrue(generated.find("#include <mod7_stubs.h>") != std::string::npos,
                     "function sources should include the module stub header");
            t.IsTrue(registration.find("#include \"ps2_overlay_modules.h\"") != std::string::npos,
                     "registration should include the overlay registry");
            t.IsTrue(registration.find("g_ps2RecompiledFunctionTable") == std::string::npos,
                     "modules must not define the main ELF table");
            t.IsTrue(registration.find("mod7_FunctionTable[0] = mod7_FUN_00001080_0x1080;") != std::string::npos,
                     "the module table should hold the scoped function");
            t.IsTrue(registration.find("module.loadAddress = 0x1000u;") != std::string::npos,
                     "registration should carry the load address");
            t.IsTrue(registration.find("mod7_Guard[4u] = {\n    0x0, 0x7, 0xab, 0xcd,") != std::string::npos,
                     "registration should embed the guard bytes");
            t.IsTrue(registration.find("g_module") == std::string::npos,
                     "registration names must be module-prefixed so unity builds can merge modules");
            t.IsTrue(registration.find("ps2_overlays::registerModule(module);") != std::string::npos,
                     "registration should register the module");
        });

        tc.Run("config manager parses and validates module settings", [](TestCase &t) {
            const auto path = writeTempConfig("module_name = \"mod003\"\nmodule_load = \"0x35ec80\"\nmodule_guard_size = 32\n");
            const RecompilerConfig config = ConfigManager(path.string()).loadConfig();
            t.Equals(config.moduleName, std::string("mod003"), "module name should parse");
            t.Equals(config.moduleLoadAddress, 0x35ec80u, "module load address should parse from hex");
            t.Equals(config.moduleGuardSize, 32u, "guard size should parse");
            std::filesystem::remove(path);

            const auto bad = writeTempConfig("module_name = \"3bad-name\"\n");
            bool threw = false;
            try
            {
                (void)ConfigManager(bad.string()).loadConfig();
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            t.IsTrue(threw, "module names that are not C identifiers should be rejected");
            std::filesystem::remove(bad);
        }); });
}
