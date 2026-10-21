// SPU2 driven through its registers and DMA channels 4 / 7, as Sony's libsd does.
#include "emulator/core/iop_memory.h"
#include "emulator/core/iop_spu2.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace ps2x::iop::detail;

namespace
{
    constexpr uint32_t kBase = 0x1F900000u;
    constexpr uint32_t kMmix = kBase + 0x198u;
    constexpr uint32_t kAttr = kBase + 0x19Au;
    constexpr uint32_t kIrqaH = kBase + 0x19Cu;
    constexpr uint32_t kIrqaL = kBase + 0x19Eu;
    constexpr uint32_t kKon = kBase + 0x1A0u;
    constexpr uint32_t kKoff = kBase + 0x1A4u;
    constexpr uint32_t kTsaH = kBase + 0x1A8u;
    constexpr uint32_t kTsaL = kBase + 0x1AAu;
    constexpr uint32_t kStd = kBase + 0x1ACu;
    constexpr uint32_t kAdmas = kBase + 0x1B0u;
    constexpr uint32_t kVmixL = kBase + 0x188u;
    constexpr uint32_t kVmixR = kBase + 0x190u;
    constexpr uint32_t kEndx = kBase + 0x340u;
    constexpr uint32_t kMvolL = kBase + 0x760u;
    constexpr uint32_t kMvolR = kBase + 0x762u;
    constexpr uint32_t kBvolL = kBase + 0x76Cu;
    constexpr uint32_t kBvolR = kBase + 0x76Eu;
    constexpr uint32_t kIrqInfo = kBase + 0x7C2u;
    constexpr uint32_t kDma4Madr = 0x1F8010C0u;
    constexpr uint32_t kDma4Bcr = 0x1F8010C4u;
    constexpr uint32_t kDma4Chcr = 0x1F8010C8u;

    int failures = 0;

    void expect(bool value, const char *message)
    {
        if (!value)
        {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    }

    uint32_t voiceReg(int voice, uint32_t offset) { return kBase + static_cast<uint32_t>(voice) * 0x10u + offset; }
    uint32_t voiceAddr(int voice, uint32_t offset) { return kBase + 0x1C0u + static_cast<uint32_t>(voice) * 0xCu + offset; }

    struct Rig
    {
        IopMemory memory;
        IopSpu2 spu2{memory};
        std::vector<int16_t> output;
        std::vector<int> interrupts;
        uint64_t cycle = 0;

        Rig()
        {
            memory.setSpu2(&spu2);
            spu2.setOutputSink([this](const int16_t *frames, size_t count)
                               { output.insert(output.end(), frames, frames + count * 2u); });
            spu2.setInterruptSink([this](int irq)
                                  { interrupts.push_back(irq); });
            // Core 0 voices to the dry output, full master volume on both cores.
            memory.write16(kMmix, 0x0FF0u);
            memory.write16(kMmix + 0x400u, 0x0FFCu);
            for (uint32_t core = 0; core < 2u; ++core)
            {
                memory.write16(kMvolL + core * 0x28u, 0x3FFFu);
                memory.write16(kMvolR + core * 0x28u, 0x3FFFu);
            }
            memory.write16(kBase + 0x788u + 0x08u, 0x7FFFu); // core 1 AVOLL: core 0 output in
            memory.write16(kBase + 0x788u + 0x0Au, 0x7FFFu);
        }

        void runSamples(uint64_t samples)
        {
            cycle += samples * IopSpu2::CyclesPerSample;
            spu2.advanceTo(cycle);
            spu2.flush();
        }

        // Writes words at a word address through the STD port.
        void writeIo(uint32_t address, const std::vector<uint16_t> &words)
        {
            memory.write16(kTsaH, static_cast<uint16_t>(address >> 16u));
            memory.write16(kTsaL, static_cast<uint16_t>(address));
            for (const uint16_t word : words)
                memory.write16(kStd, word);
            memory.write16(kAttr, 0x8010u);
        }
    };

    // One ADPCM block: shift 0, filter 0, every nibble 'nibble' (value nibble << 12).
    std::vector<uint16_t> block(uint8_t flags, uint8_t nibble)
    {
        std::vector<uint16_t> words(8u);
        words[0] = static_cast<uint16_t>(flags) << 8u;
        const uint16_t packed = static_cast<uint16_t>(nibble * 0x1111u);
        std::fill(words.begin() + 1, words.end(), packed);
        return words;
    }

    void keyOnVoice0(Rig &rig, uint32_t address, uint16_t adsr1, uint16_t adsr2)
    {
        rig.memory.write16(voiceReg(0, 0x0u), 0x3FFFu);
        rig.memory.write16(voiceReg(0, 0x2u), 0x3FFFu);
        rig.memory.write16(voiceReg(0, 0x4u), 0x1000u);
        rig.memory.write16(voiceReg(0, 0x6u), adsr1);
        rig.memory.write16(voiceReg(0, 0x8u), adsr2);
        rig.memory.write32(voiceAddr(0, 0x0u), (address >> 16u) | ((address & 0xFFFFu) << 16u));
        rig.memory.write16(kVmixL, 1u);
        rig.memory.write16(kVmixR, 1u);
        rig.memory.write16(kKon, 1u);
    }

    void testVoicePlaysAndLoops()
    {
        Rig rig;
        constexpr uint32_t address = 0x5000u;
        // Two blocks: loop start, then loop end + repeat.
        std::vector<uint16_t> data = block(0x04u, 0x4u);
        const std::vector<uint16_t> second = block(0x03u, 0x4u);
        data.insert(data.end(), second.begin(), second.end());
        rig.writeIo(address, data);
        expect(rig.spu2.ramWord(address) == 0x0400u && rig.spu2.ramWord(address + 9u) == 0x4444u,
               "STD writes land at TSA in sound RAM");

        // Fastest attack, slowest decay to sustain level 0xF, constant sustain, fastest release.
        keyOnVoice0(rig, address, 0x00FFu, 0x1FC0u);
        rig.runSamples(200u);
        expect(rig.output.size() == 2u * 200u, "one stereo frame per 768 IOP cycles");
        const int16_t envelope = static_cast<int16_t>(rig.memory.read16(voiceReg(0, 0xAu)));
        expect(envelope == 0x7FFF, "fastest attack then sustain level 0xF keeps the envelope at its maximum");
        expect((rig.memory.read16(kEndx) & 1u) != 0u, "ENDX is set once the loop-end block has played");
        const int16_t last = rig.output[rig.output.size() - 2u];
        expect(last > 0x0800, "a constant positive ADPCM sample comes out positive and loud");
        expect(rig.output[rig.output.size() - 1u] == last, "equal left and right volumes give equal channels");

        rig.memory.write16(kEndx, 0u);
        expect((rig.memory.read16(kEndx) & 1u) == 0u, "writing ENDX clears it");
        rig.runSamples(100u);
        expect((rig.memory.read16(kEndx) & 1u) != 0u, "a repeating voice keeps looping past the loop end");

        // Release rate 0 (fastest), linear: the voice goes silent and stops.
        rig.memory.write16(kKoff, 1u);
        rig.runSamples(64u);
        expect(rig.memory.read16(voiceReg(0, 0xAu)) == 0u, "key off releases the envelope to zero");
        expect(rig.output[rig.output.size() - 2u] == 0, "a released voice is silent");
    }

    void testEndWithoutRepeatStopsVoice()
    {
        Rig rig;
        constexpr uint32_t address = 0x6000u;
        rig.writeIo(address, block(0x01u, 0x3u));
        keyOnVoice0(rig, address, 0x000Fu, 0x1FC0u);
        rig.runSamples(64u);
        expect(rig.memory.read16(voiceReg(0, 0xAu)) == 0u, "loop end without repeat silences the voice");
        expect((rig.memory.read16(kEndx) & 1u) != 0u, "and sets ENDX");
    }

    void testDmaWriteAndCompletion()
    {
        Rig rig;
        std::vector<uint16_t> data(256u);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<uint16_t>(0xA000u + i);
        expect(rig.memory.writeRam(0x10000u, data.data(), data.size() * 2u), "IOP RAM source");
        rig.memory.write16(kTsaH, 0x0001u);
        rig.memory.write16(kTsaL, 0x2340u);
        rig.memory.write16(kAttr, 0x8020u);
        rig.memory.write32(kDma4Madr, 0x10000u);
        rig.memory.write32(kDma4Bcr, (8u << 16u) | 16u); // 8 blocks of 16 words = 512 bytes
        rig.memory.write32(kDma4Chcr, 0x01000201u);
        expect(rig.spu2.ramWord(0x12340u) == 0xA000u && rig.spu2.ramWord(0x12340u + 255u) == 0xA0FFu,
               "DMA channel 4 copies IOP RAM to sound RAM at TSA");
        expect((rig.memory.read32(kDma4Chcr) & 0x01000000u) != 0u, "the channel stays busy until it completes");
        rig.runSamples(4u);
        expect((rig.memory.read32(kDma4Chcr) & 0x01000000u) == 0u, "the start bit clears on completion");
        expect(rig.interrupts.size() == 1u && rig.interrupts[0] == 0x24, "completion raises the channel 4 interrupt");
        expect((rig.memory.read16(kBase + 0x344u) & 0x80u) != 0u, "STATX reports the DMA done");

        // Read back through the same channel (SPU2 -> IOP).
        rig.memory.write16(kTsaH, 0x0001u);
        rig.memory.write16(kTsaL, 0x2340u);
        rig.memory.write32(kDma4Madr, 0x20000u);
        expect((rig.memory.read32(kDma4Bcr) >> 16u) == 0u, "a completed transfer leaves no block to count");
        rig.memory.write32(kDma4Bcr, (8u << 16u) | 16u);
        rig.memory.write32(kDma4Chcr, 0x01000200u);
        uint16_t copy[2]{};
        expect(rig.memory.readRam(0x20000u + 254u * 2u, copy, sizeof(copy)) && copy[0] == 0xA0FEu && copy[1] == 0xA0FFu,
               "DMA reads sound RAM back to the IOP");
    }

    void testAutoDmaInput()
    {
        Rig rig;
        // Two 1 KB chunks: 256 L samples then 256 R samples each.
        std::vector<int16_t> data(1024u);
        for (size_t i = 0; i < 256u; ++i)
        {
            data[i] = 1000;
            data[256u + i] = -1000;
            data[512u + i] = 2000;
            data[768u + i] = -2000;
        }
        expect(rig.memory.writeRam(0x30000u, data.data(), data.size() * 2u), "IOP RAM stream");
        rig.memory.write16(kBvolL, 0x7FFFu);
        rig.memory.write16(kBvolR, 0x7FFFu);
        rig.memory.write16(kAdmas, 1u);
        rig.memory.write32(kDma4Madr, 0x30000u);
        rig.memory.write32(kDma4Bcr, (32u << 16u) | 16u); // 2048 bytes
        rig.memory.write32(kDma4Chcr, 0x01000201u);
        expect(rig.interrupts.empty(), "an AutoDMA transfer completes as the input consumes it");
        expect((rig.memory.read32(kDma4Chcr) & 0x01000000u) != 0u, "AutoDMA keeps the channel busy");

        rig.runSamples(256u + 16u);
        expect(rig.interrupts.size() == 1u && rig.interrupts[0] == 0x24, "AutoDMA completion raises the channel interrupt");
        expect((rig.memory.read32(kDma4Chcr) & 0x01000000u) == 0u, "and frees the channel");
        rig.runSamples(512u);
        // Volumes are all close to full scale: the input comes out about as it went in.
        const auto at = [&](size_t frame, size_t channel)
        { return static_cast<int>(rig.output[frame * 2u + channel]); };
        expect(at(100u, 0u) == 0 && at(100u, 1u) == 0, "the first half plays silence while the stream fills the second");
        const int first = at(300u, 0u);
        const int second = at(600u, 0u);
        expect(first > 990 && first <= 1000 && std::abs(at(300u, 1u) + first) <= 4, "the first chunk plays from the second half, L then R");
        expect(second > 1990 && std::abs(at(600u, 1u) + second) <= 4, "the second chunk follows");
        expect(rig.spu2.stats().admaUnderruns > 0u, "once the stream ends the input underruns");
    }

    void testIrqAddress()
    {
        Rig rig;
        constexpr uint32_t address = 0x7000u;
        std::vector<uint16_t> data = block(0x00u, 0x1u);
        const std::vector<uint16_t> second = block(0x03u, 0x1u);
        data.insert(data.end(), second.begin(), second.end());
        rig.writeIo(address, data);
        rig.memory.write16(kIrqaH, 0u);
        rig.memory.write16(kIrqaL, static_cast<uint16_t>(address + 8u + 3u));
        rig.memory.write16(kAttr, 0x8040u);
        keyOnVoice0(rig, address, 0x000Fu, 0x1FC0u);
        expect(rig.interrupts.empty(), "the first block does not hold IRQA");
        rig.runSamples(40u);
        expect(std::count(rig.interrupts.begin(), rig.interrupts.end(), IopSpu2::Irq) == 1,
               "a voice reading the block that holds IRQA raises interrupt 9");
        expect((rig.memory.read16(kIrqInfo) & 0x4u) != 0u, "IRQINFO flags core 0");
        rig.memory.write16(kAttr, 0x8000u);
        expect((rig.memory.read16(kIrqInfo) & 0x4u) == 0u, "clearing the IRQ enable acknowledges it");
        rig.runSamples(100u);
        expect(std::count(rig.interrupts.begin(), rig.interrupts.end(), IopSpu2::Irq) == 1, "disabled IRQs stay quiet");
    }

    void testRegisterAccess()
    {
        Rig rig;
        rig.memory.write32(voiceAddr(3, 0x4u), 0x23400001u);
        expect(rig.memory.read16(voiceAddr(3, 0x4u)) == 0x0001u && rig.memory.read16(voiceAddr(3, 0x6u)) == 0x2340u,
               "32-bit writes split into two 16-bit registers (LSAX high, low)");
        expect(rig.memory.read32(voiceAddr(3, 0x4u)) == 0x23400001u, "32-bit reads join them");
        rig.memory.write8(kBase + 0x1B0u + 0x400u, 0x02u);
        expect(rig.memory.read16(kAdmas + 0x400u) == 0x0002u, "core 1 registers sit 0x400 above core 0");
        rig.memory.write16(kBase + 0x760u + 0x14u, 0x1234u);
        expect(rig.memory.read16(kBase + 0x774u) == 0x1234u, "reverb coefficients read back");
    }
}

int main()
{
    testVoicePlaysAndLoops();
    testEndWithoutRepeatStopsVoice();
    testDmaWriteAndCompletion();
    testAutoDmaInput();
    testIrqAddress();
    testRegisterAccess();
    if (failures != 0)
    {
        std::cerr << failures << " SPU2 check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "SPU2 tests passed\n";
    return EXIT_SUCCESS;
}
