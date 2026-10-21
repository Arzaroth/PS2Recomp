#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ps2x::iop::detail
{
    class IopMemory;

    // SPU2 (IOP 0x1F900000-0x1F9007FF): two cores of 24 ADPCM voices with ADSR envelopes, the
    // sound data input fed by AutoDMA, reverb, and 2 MB of sound RAM loaded through DMA channels
    // 4 (core 0) and 7 (core 1) or the STD port. Produces 48 kHz stereo, one sample per 768 IOP
    // cycles; IRQA hits raise interrupt 9, DMA completions interrupts 0x24 / 0x28.
    class IopSpu2
    {
    public:
        static constexpr uint32_t Base = 0x1F900000u;
        static constexpr uint32_t End = 0x1F900800u;
        static constexpr uint32_t RamWords = 1u << 20u;
        static constexpr uint32_t SampleRate = 48000u;
        static constexpr uint64_t CyclesPerSample = 768u;
        static constexpr int Irq = 9;
        static constexpr int DmaIrq[2] = {0x24, 0x28};
        static constexpr uint32_t DmaBase[2] = {0x1F8010C0u, 0x1F801500u};

        // Interleaved stereo frames, called every few hundred samples.
        using OutputSink = std::function<void(const int16_t *frames, size_t count)>;
        using InterruptSink = std::function<void(int irq)>;

        struct Stats
        {
            uint64_t samples = 0;
            uint64_t keyOns = 0;
            uint64_t keyOffs = 0;
            uint64_t dmaWrites = 0;
            uint64_t dmaWriteBytes = 0;
            uint64_t dmaReads = 0;
            uint64_t admaTransfers = 0;
            uint64_t admaBytes = 0;
            uint64_t admaUnderruns = 0;
            uint64_t ioWrites = 0;
            uint64_t irqs = 0;
            uint64_t registerWrites = 0;
            int32_t peak = 0;
        };

        explicit IopSpu2(IopMemory &memory) noexcept;

        void reset();
        void setOutputSink(OutputSink sink) { m_output = std::move(sink); }
        void setInterruptSink(InterruptSink sink) { m_interrupt = std::move(sink); }
        // IOP cycle counter: register accesses first bring the SPU2 up to it, so that key-ons and
        // transfers land on the right sample even though mixing runs in batches.
        void setClock(const uint64_t *clock) noexcept { m_clock = clock; }

        [[nodiscard]] static bool handles(uint32_t address) noexcept { return address >= Base && address < End; }
        [[nodiscard]] static bool handlesDma(uint32_t address) noexcept
        {
            return (address >= DmaBase[0] && address < DmaBase[0] + 12u) ||
                   (address >= DmaBase[1] && address < DmaBase[1] + 12u);
        }

        [[nodiscard]] uint16_t read16(uint32_t address);
        void write16(uint32_t address, uint16_t value);
        // DMA channel registers (MADR, BCR, CHCR) of channels 4 and 7.
        [[nodiscard]] uint32_t readDma32(uint32_t address);
        void writeDma32(uint32_t address, uint32_t value);

        // Runs the SPU2 up to an IOP cycle: mixes the samples due, completes DMA transfers.
        void advanceTo(uint64_t cycle);
        // Next cycle the SPU2 must run by: a DMA completion, an AutoDMA buffer refill, every
        // sample while an IRQ address is armed, otherwise the end of a batch of samples.
        // Kept up to date after every register access and advance.
        [[nodiscard]] uint64_t nextEventCycle() const noexcept { return m_nextEvent; }
        // Hands the frames mixed so far to the output sink (otherwise sent 256 at a time).
        void flush();

        [[nodiscard]] const Stats &stats() const noexcept { return m_stats; }
        [[nodiscard]] std::vector<std::string> describe() const;
        // Sound RAM, 16-bit words.
        [[nodiscard]] uint16_t ramWord(uint32_t address) const noexcept { return m_ram[address & (RamWords - 1u)]; }
        void setRamWord(uint32_t address, uint16_t value) noexcept { m_ram[address & (RamWords - 1u)] = value; }

    private:
        struct Envelope
        {
            enum class Phase : uint8_t
            {
                Off,
                Attack,
                Decay,
                Sustain,
                Release,
            };

            Phase phase = Phase::Off;
            int32_t level = 0;
            uint32_t counter = 0;
        };

        // Fixed volume or a sweep (VOLL/VOLR, MVOLL/MVOLR).
        struct Volume
        {
            uint16_t reg = 0;
            int32_t level = 0;
            uint32_t counter = 0;

            void set(uint16_t value);
            void tick();
        };

        struct Voice
        {
            Volume volume[2];
            uint16_t pitch = 0;
            uint16_t adsr1 = 0;
            uint16_t adsr2 = 0;
            uint32_t startAddress = 0;
            uint32_t loopAddress = 0;
            uint32_t nextAddress = 0;
            bool loopAddressWritten = false;
            Envelope envelope;
            uint32_t counter = 0;
            uint8_t blockFlags = 0;
            // [0..2] last samples of the previous block, [3..30] the current block.
            std::array<int16_t, 31> samples{};
            int32_t history[2]{};
            int32_t output = 0;
        };

        struct Core
        {
            std::array<Voice, 24> voices;
            uint32_t pitchMod = 0;
            uint32_t noiseOn = 0;
            uint32_t dryMix[2]{};
            uint32_t wetMix[2]{};
            uint16_t mmix = 0;
            uint16_t attr = 0;
            uint32_t irqAddress = 0;
            uint32_t transferAddress = 0;
            uint32_t endx = 0;
            uint16_t admas = 0;
            uint16_t statx = 0;
            Volume master[2];
            int16_t effectVolume[2]{};
            int16_t externalVolume[2]{};
            int16_t inputVolume[2]{};
            // Reverb: work area and the 22 address registers, coefficients.
            uint32_t effectStart = 0;
            uint32_t effectEnd = 0;
            std::array<uint32_t, 22> reverbAddress{};
            std::array<int16_t, 10> reverbCoef{};
            uint32_t reverbPosition = 0;
            int32_t reverbOut[2]{};
            std::array<int32_t, 8> reverbHistory[2]{};
            // Sound data input (AutoDMA): read position in the 2 x 256-sample buffer.
            uint32_t inputPosition = 0;
            // DMA channel registers.
            uint32_t madr = 0;
            uint32_t bcr = 0;
            uint32_t chcr = 0;
            bool dmaBusy = false;
            bool admaRunning = false;
            uint64_t dmaCompletionCycle = 0;
            uint32_t admaAddress = 0;
            uint32_t admaRemaining = 0;
            bool inputHalfFilled[2]{};
        };

        struct StereoSample
        {
            int32_t left = 0;
            int32_t right = 0;
        };

        void catchUp();
        void updateNextEvent() noexcept;
        void writeCoreRegister(int core, uint32_t offset, uint16_t value);
        [[nodiscard]] uint16_t readCoreRegister(int core, uint32_t offset) const;
        void writeVoiceRegister(Core &core, int voice, uint32_t offset, uint16_t value);
        void writeVoiceAddress(Core &core, int voice, uint32_t offset, uint16_t value);
        void writeReverbRegister(Core &core, uint32_t offset, uint16_t value);
        void keyOn(int core, uint32_t mask);
        void keyOff(int core, uint32_t mask);
        void startDma(int core);
        void finishDma(int core);
        void fillInputHalf(int core, int half);
        void checkIrq(uint32_t address, uint32_t words);
        void decodeBlock(int core, Voice &voice);
        void advanceEnvelope(Voice &voice);
        [[nodiscard]] int32_t voiceSample(int core, int index, Voice &voice);
        [[nodiscard]] StereoSample mixCore(int core, const StereoSample &external);
        [[nodiscard]] StereoSample reverb(Core &core, const StereoSample &input);
        void renderSample();

        IopMemory &m_memory;
        std::vector<uint16_t> m_ram;
        std::array<uint16_t, 0x400> m_regs{};
        std::array<Core, 2> m_cores;
        uint16_t m_irqInfo = 0;
        uint32_t m_noiseLevel = 1;
        uint32_t m_noiseCounter = 0;
        uint64_t m_cycle = 0;
        uint64_t m_nextSampleCycle = CyclesPerSample;
        uint64_t m_nextEvent = CyclesPerSample;
        bool m_reverbPhase = false;
        std::vector<int16_t> m_pending;
        OutputSink m_output;
        InterruptSink m_interrupt;
        const uint64_t *m_clock = nullptr;
        Stats m_stats;
    };
}
