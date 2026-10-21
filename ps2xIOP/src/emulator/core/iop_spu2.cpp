#include "iop_spu2.h"

#include "iop_memory.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kRamMask = IopSpu2::RamWords - 1u;
        constexpr uint32_t kDmaStart = 1u << 24u;
        constexpr uint32_t kOutputChunk = 256u;
        // Samples mixed in one go when nothing needs a tighter schedule (about 5 ms).
        constexpr uint64_t kBatchSamples = 256u;
        // Sound data input buffers (words): core n L at 0x2000 + n * 0x400, R 0x200 words further.
        constexpr uint32_t kInputBase = 0x2000u;

        // ATTR bits.
        constexpr uint16_t kAttrIrqEnable = 0x0040u;
        constexpr uint16_t kAttrFxEnable = 0x0080u;

        // MMIX gates: Ext = core 0 output (core 1 only), Inp = sound data input, Snd = voices.
        constexpr uint16_t kWetExtR = 0x001u, kWetExtL = 0x002u, kDryExtR = 0x004u, kDryExtL = 0x008u;
        constexpr uint16_t kWetInpR = 0x010u, kWetInpL = 0x020u, kDryInpR = 0x040u, kDryInpL = 0x080u;
        constexpr uint16_t kWetSndR = 0x100u, kWetSndL = 0x200u, kDrySndR = 0x400u, kDrySndL = 0x800u;

        constexpr int32_t kFilterPositive[5] = {0, 60, 115, 98, 122};
        constexpr int32_t kFilterNegative[5] = {0, 0, -52, -55, -60};

        int32_t clamp16(int32_t value)
        {
            return std::clamp(value, -0x8000, 0x7FFF);
        }

        int32_t mul15(int32_t a, int32_t b)
        {
            return (a * b) >> 15;
        }

        // One step of an envelope or a volume sweep. rate is (shift << 2) | step, 0..127.
        void envelopeStep(int32_t &level, uint32_t &counter, uint32_t rate, bool decreasing, bool exponential)
        {
            const int32_t shift = static_cast<int32_t>(rate >> 2u);
            int32_t step = 7 - static_cast<int32_t>(rate & 3u);
            if (decreasing)
                step = ~step;
            uint32_t increment = 0x8000u;
            if (shift < 11)
                step *= 1 << (11 - shift);
            else if (shift > 11)
                increment = shift - 11 < 16 ? increment >> (shift - 11) : 0u;

            if (exponential)
            {
                if (decreasing)
                {
                    step = (step * level) >> 15;
                }
                else if (level >= 0x6000)
                {
                    if (shift < 10)
                        step >>= 2;
                    else if (shift >= 11)
                        increment >>= 2;
                    else
                    {
                        step >>= 1;
                        increment >>= 1;
                    }
                }
            }

            counter += increment;
            if ((counter & 0x8000u) == 0u)
                return;
            counter = 0u;
            level = std::clamp(level + step, 0, 0x7FFF);
        }

        // 4-point Catmull-Rom between y1 and y2; t is a 12-bit fraction.
        int32_t interpolate(int32_t y0, int32_t y1, int32_t y2, int32_t y3, uint32_t t)
        {
            const float f = static_cast<float>(t) * (1.0f / 4096.0f);
            const float a = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
            const float b = static_cast<float>(y0) - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
            const float c = -0.5f * y0 + 0.5f * y2;
            const float value = ((a * f + b) * f + c) * f + static_cast<float>(y1);
            return clamp16(static_cast<int32_t>(value));
        }

        uint32_t setHigh(uint32_t current, uint16_t value)
        {
            return ((static_cast<uint32_t>(value) & 0xFu) << 16u) | (current & 0xFFFFu);
        }

        uint32_t setLow(uint32_t current, uint16_t value)
        {
            return (current & 0xF0000u) | value;
        }
    }

    void IopSpu2::Volume::set(uint16_t value)
    {
        reg = value;
        counter = 0u;
        if ((value & 0x8000u) == 0u)
            level = static_cast<int16_t>(static_cast<uint16_t>(value << 1u));
    }

    void IopSpu2::Volume::tick()
    {
        if ((reg & 0x8000u) == 0u)
            return;
        const bool negative = (reg & 0x1000u) != 0u;
        int32_t magnitude = std::abs(level);
        envelopeStep(magnitude, counter, reg & 0x7Fu, (reg & 0x2000u) != 0u, (reg & 0x4000u) != 0u);
        level = negative ? -magnitude : magnitude;
    }

    IopSpu2::IopSpu2(IopMemory &memory) noexcept
        : m_memory(memory), m_ram(RamWords)
    {
        reset();
    }

    void IopSpu2::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        m_regs.fill(0u);
        for (Core &core : m_cores)
            core = Core{};
        m_irqInfo = 0u;
        m_noiseLevel = 1u;
        m_noiseCounter = 0u;
        m_cycle = 0u;
        m_nextSampleCycle = CyclesPerSample;
        m_reverbPhase = false;
        m_pending.clear();
        m_stats = Stats{};
        updateNextEvent();
    }

    void IopSpu2::catchUp()
    {
        if (m_clock && *m_clock > m_cycle)
            advanceTo(*m_clock);
    }

    uint16_t IopSpu2::read16(uint32_t address)
    {
        catchUp();
        const uint32_t offset = (address - Base) & 0x7FEu;
        if (offset >= 0x760u)
        {
            if (offset == 0x7C2u)
                return m_irqInfo;
            if (offset < 0x7B0u)
            {
                const int core = offset < 0x788u ? 0 : 1;
                const uint32_t reg = offset - (core == 0 ? 0x760u : 0x788u);
                if (reg == 0x10u || reg == 0x12u)
                    return static_cast<uint16_t>(m_cores[core].master[(reg - 0x10u) >> 1u].level);
            }
            return m_regs[offset >> 1u];
        }
        return readCoreRegister(static_cast<int>(offset >> 10u), offset & 0x3FEu);
    }

    uint16_t IopSpu2::readCoreRegister(int coreIndex, uint32_t offset) const
    {
        const Core &core = m_cores[coreIndex];
        const uint16_t stored = m_regs[((static_cast<uint32_t>(coreIndex) << 10u) | offset) >> 1u];
        if (offset < 0x180u)
        {
            const Voice &voice = core.voices[offset >> 4u];
            switch (offset & 0xFu)
            {
            case 0x0u:
                return voice.volume[0].reg;
            case 0x2u:
                return voice.volume[1].reg;
            case 0xAu:
                return static_cast<uint16_t>(voice.envelope.level);
            case 0xCu:
                return static_cast<uint16_t>(voice.volume[0].level);
            case 0xEu:
                return static_cast<uint16_t>(voice.volume[1].level);
            default:
                return stored;
            }
        }
        if (offset >= 0x1C0u && offset < 0x2E0u)
        {
            const Voice &voice = core.voices[(offset - 0x1C0u) / 12u];
            uint32_t address = 0u;
            switch ((offset - 0x1C0u) % 12u)
            {
            case 0u:
            case 2u:
                address = voice.startAddress;
                break;
            case 4u:
            case 6u:
                address = voice.loopAddress;
                break;
            default:
                // The word being decoded: header, then 4 samples per word.
                address = voice.envelope.phase == Envelope::Phase::Off
                              ? voice.nextAddress
                              : ((voice.nextAddress & ~7u) + 1u + ((voice.counter >> 12u) >> 2u)) & kRamMask;
                break;
            }
            return ((offset - 0x1C0u) % 12u) % 4u == 0u ? static_cast<uint16_t>(address >> 16u)
                                                         : static_cast<uint16_t>(address);
        }
        switch (offset)
        {
        case 0x19Au:
            return core.attr;
        case 0x1A8u:
            return static_cast<uint16_t>(core.transferAddress >> 16u);
        case 0x1AAu:
            return static_cast<uint16_t>(core.transferAddress);
        case 0x1B0u:
            return core.admas;
        case 0x340u:
            return static_cast<uint16_t>(core.endx);
        case 0x342u:
            return static_cast<uint16_t>(core.endx >> 16u);
        case 0x344u:
            return core.statx;
        default:
            return stored;
        }
    }

    void IopSpu2::write16(uint32_t address, uint16_t value)
    {
        catchUp();
        const struct ScheduleGuard
        {
            IopSpu2 &spu2;
            ~ScheduleGuard() { spu2.updateNextEvent(); }
        } scheduleGuard{*this};
        const uint32_t offset = (address - Base) & 0x7FEu;
        ++m_stats.registerWrites;
        m_regs[offset >> 1u] = value;
        if (offset >= 0x760u)
        {
            if (offset >= 0x7B0u)
                return;
            const int coreIndex = offset < 0x788u ? 0 : 1;
            Core &core = m_cores[coreIndex];
            const uint32_t reg = offset - (coreIndex == 0 ? 0x760u : 0x788u);
            switch (reg)
            {
            case 0x00u:
            case 0x02u:
                core.master[reg >> 1u].set(value);
                break;
            case 0x04u:
            case 0x06u:
                core.effectVolume[(reg - 0x04u) >> 1u] = static_cast<int16_t>(value);
                break;
            case 0x08u:
            case 0x0Au:
                core.externalVolume[(reg - 0x08u) >> 1u] = static_cast<int16_t>(value);
                break;
            case 0x0Cu:
            case 0x0Eu:
                core.inputVolume[(reg - 0x0Cu) >> 1u] = static_cast<int16_t>(value);
                break;
            default:
                if (reg >= 0x14u && reg < 0x28u)
                    core.reverbCoef[(reg - 0x14u) >> 1u] = static_cast<int16_t>(value);
                break;
            }
            return;
        }
        writeCoreRegister(static_cast<int>(offset >> 10u), offset & 0x3FEu, value);
    }

    void IopSpu2::writeCoreRegister(int coreIndex, uint32_t offset, uint16_t value)
    {
        Core &core = m_cores[coreIndex];
        if (offset < 0x180u)
        {
            writeVoiceRegister(core, static_cast<int>(offset >> 4u), offset & 0xFu, value);
            return;
        }
        if (offset >= 0x1C0u && offset < 0x2E0u)
        {
            writeVoiceAddress(core, static_cast<int>((offset - 0x1C0u) / 12u), (offset - 0x1C0u) % 12u, value);
            return;
        }
        if (offset >= 0x2E0u && offset < 0x340u)
        {
            writeReverbRegister(core, offset, value);
            return;
        }

        const auto setMaskLow = [&](uint32_t &mask)
        { mask = (mask & 0xFF0000u) | value; };
        const auto setMaskHigh = [&](uint32_t &mask)
        { mask = (mask & 0xFFFFu) | ((static_cast<uint32_t>(value) & 0xFFu) << 16u); };
        switch (offset)
        {
        case 0x180u:
            setMaskLow(core.pitchMod);
            break;
        case 0x182u:
            setMaskHigh(core.pitchMod);
            break;
        case 0x184u:
            setMaskLow(core.noiseOn);
            break;
        case 0x186u:
            setMaskHigh(core.noiseOn);
            break;
        case 0x188u:
            setMaskLow(core.dryMix[0]);
            break;
        case 0x18Au:
            setMaskHigh(core.dryMix[0]);
            break;
        case 0x18Cu:
            setMaskLow(core.wetMix[0]);
            break;
        case 0x18Eu:
            setMaskHigh(core.wetMix[0]);
            break;
        case 0x190u:
            setMaskLow(core.dryMix[1]);
            break;
        case 0x192u:
            setMaskHigh(core.dryMix[1]);
            break;
        case 0x194u:
            setMaskLow(core.wetMix[1]);
            break;
        case 0x196u:
            setMaskHigh(core.wetMix[1]);
            break;
        case 0x198u:
            core.mmix = value;
            break;
        case 0x19Au:
            core.attr = value;
            // libsd acknowledges an IRQ by clearing the enable bit.
            if ((value & kAttrIrqEnable) == 0u)
                m_irqInfo &= static_cast<uint16_t>(~(4u << coreIndex));
            break;
        case 0x19Cu:
            core.irqAddress = setHigh(core.irqAddress, value);
            break;
        case 0x19Eu:
            core.irqAddress = setLow(core.irqAddress, value);
            break;
        case 0x1A0u:
            keyOn(coreIndex, value);
            break;
        case 0x1A2u:
            keyOn(coreIndex, (static_cast<uint32_t>(value) & 0xFFu) << 16u);
            break;
        case 0x1A4u:
            keyOff(coreIndex, value);
            break;
        case 0x1A6u:
            keyOff(coreIndex, (static_cast<uint32_t>(value) & 0xFFu) << 16u);
            break;
        case 0x1A8u:
            core.transferAddress = setHigh(core.transferAddress, value);
            break;
        case 0x1AAu:
            core.transferAddress = setLow(core.transferAddress, value);
            break;
        case 0x1ACu:
            // Manual transfer port: each word lands at TSA.
            checkIrq(core.transferAddress, 1u);
            m_ram[core.transferAddress] = value;
            core.transferAddress = (core.transferAddress + 1u) & kRamMask;
            ++m_stats.ioWrites;
            break;
        case 0x1B0u:
            core.admas = value;
            if ((value & (1u << coreIndex)) == 0u && core.admaRunning)
            {
                core.admaRunning = false;
                core.dmaBusy = false;
                core.chcr &= ~kDmaStart;
            }
            break;
        case 0x340u:
            core.endx &= 0xFF0000u;
            break;
        case 0x342u:
            core.endx &= 0xFFFFu;
            break;
        default:
            break;
        }
    }

    void IopSpu2::writeVoiceRegister(Core &core, int index, uint32_t offset, uint16_t value)
    {
        Voice &voice = core.voices[index];
        switch (offset)
        {
        case 0x0u:
        case 0x2u:
            voice.volume[offset >> 1u].set(value);
            break;
        case 0x4u:
            voice.pitch = value;
            break;
        case 0x6u:
            voice.adsr1 = value;
            break;
        case 0x8u:
            voice.adsr2 = value;
            break;
        case 0xAu:
            voice.envelope.level = std::clamp<int32_t>(static_cast<int16_t>(value), 0, 0x7FFF);
            break;
        default:
            break;
        }
    }

    void IopSpu2::writeVoiceAddress(Core &core, int index, uint32_t offset, uint16_t value)
    {
        Voice &voice = core.voices[index];
        switch (offset)
        {
        case 0u:
            voice.startAddress = setHigh(voice.startAddress, value);
            break;
        case 2u:
            voice.startAddress = setLow(voice.startAddress, value);
            break;
        case 4u:
        case 6u:
            voice.loopAddress = offset == 4u ? setHigh(voice.loopAddress, value) : setLow(voice.loopAddress, value);
            // Written while playing: the loop-start flags of the data no longer move it.
            if (voice.envelope.phase != Envelope::Phase::Off)
                voice.loopAddressWritten = true;
            break;
        case 8u:
            voice.nextAddress = setHigh(voice.nextAddress, value);
            break;
        case 10u:
            voice.nextAddress = setLow(voice.nextAddress, value);
            break;
        default:
            break;
        }
    }

    void IopSpu2::writeReverbRegister(Core &core, uint32_t offset, uint16_t value)
    {
        if (offset == 0x2E0u || offset == 0x2E2u)
        {
            core.effectStart = offset == 0x2E0u ? setHigh(core.effectStart, value) : setLow(core.effectStart, value);
            core.reverbPosition = 0u;
            return;
        }
        if (offset == 0x33Cu)
        {
            core.effectEnd = ((static_cast<uint32_t>(value) & 0xFu) << 16u) | 0xFFFFu;
            core.reverbPosition = 0u;
            return;
        }
        if (offset >= 0x33Cu)
            return;
        const uint32_t index = (offset - 0x2E4u) >> 2u;
        uint32_t &address = core.reverbAddress[index];
        address = (offset & 2u) == 0u ? setHigh(address, value) : setLow(address, value);
    }

    void IopSpu2::keyOn(int coreIndex, uint32_t mask)
    {
        Core &core = m_cores[coreIndex];
        for (int index = 0; index < 24; ++index)
        {
            if ((mask & (1u << index)) == 0u)
                continue;
            Voice &voice = core.voices[index];
            voice.envelope = Envelope{Envelope::Phase::Attack, 0, 0u};
            voice.counter = 0u;
            voice.samples.fill(0);
            voice.history[0] = voice.history[1] = 0;
            voice.output = 0;
            voice.nextAddress = voice.startAddress;
            voice.loopAddressWritten = false;
            core.endx &= ~(1u << index);
            decodeBlock(coreIndex, voice);
            ++m_stats.keyOns;
        }
    }

    void IopSpu2::keyOff(int coreIndex, uint32_t mask)
    {
        Core &core = m_cores[coreIndex];
        for (int index = 0; index < 24; ++index)
        {
            Voice &voice = core.voices[index];
            if ((mask & (1u << index)) == 0u || voice.envelope.phase == Envelope::Phase::Off)
                continue;
            voice.envelope.phase = Envelope::Phase::Release;
            voice.envelope.counter = 0u;
            ++m_stats.keyOffs;
        }
    }

    uint32_t IopSpu2::readDma32(uint32_t address)
    {
        catchUp();
        const int coreIndex = address >= DmaBase[1] ? 1 : 0;
        const Core &core = m_cores[coreIndex];
        switch (address - DmaBase[coreIndex])
        {
        case 0u:
            return core.madr;
        case 4u:
            return core.bcr;
        default:
            return core.chcr;
        }
    }

    void IopSpu2::writeDma32(uint32_t address, uint32_t value)
    {
        catchUp();
        const struct ScheduleGuard
        {
            IopSpu2 &spu2;
            ~ScheduleGuard() { spu2.updateNextEvent(); }
        } scheduleGuard{*this};
        const int coreIndex = address >= DmaBase[1] ? 1 : 0;
        Core &core = m_cores[coreIndex];
        switch (address - DmaBase[coreIndex])
        {
        case 0u:
            core.madr = value & 0x00FFFFFFu;
            return;
        case 4u:
            core.bcr = value;
            return;
        default:
            break;
        }
        core.chcr = value;
        if ((value & kDmaStart) != 0u)
        {
            startDma(coreIndex);
        }
        else if (core.dmaBusy)
        {
            // Stopped by software (sceSdStopTrans / BlockTrans stop): no completion interrupt.
            core.dmaBusy = false;
            core.admaRunning = false;
        }
    }

    void IopSpu2::startDma(int coreIndex)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t wordsPerBlock = std::max<uint32_t>(core.bcr & 0xFFFFu, 1u);
        const uint32_t blocks = std::max<uint32_t>(core.bcr >> 16u, 1u);
        const uint32_t bytes = std::min<uint32_t>(wordsPerBlock * blocks * 4u, IopMemory::RamSize);
        const bool toSpu = (core.chcr & 1u) != 0u;
        core.dmaBusy = true;
        core.statx |= 0x80u;

        if (toSpu && (core.admas & (1u << coreIndex)) != 0u)
        {
            // AutoDMA: the sound data input consumes 512 bytes of L then 512 of R per 256 samples.
            core.admaRunning = true;
            core.admaAddress = core.madr;
            core.admaRemaining = bytes;
            ++m_stats.admaTransfers;
            const int playing = core.inputPosition >= 0x100u ? 1 : 0;
            if (!core.inputHalfFilled[1 - playing])
                fillInputHalf(coreIndex, 1 - playing);
            return;
        }

        // Sound RAM transfer at TSA: SPU2 words are 16-bit.
        const uint32_t words = bytes / 2u;
        std::vector<uint16_t> buffer(words);
        if (toSpu)
        {
            (void)m_memory.readRam(core.madr, buffer.data(), bytes);
            checkIrq(core.transferAddress, words);
            for (uint32_t i = 0; i < words; ++i)
                m_ram[(core.transferAddress + i) & kRamMask] = buffer[i];
            ++m_stats.dmaWrites;
            m_stats.dmaWriteBytes += bytes;
        }
        else
        {
            for (uint32_t i = 0; i < words; ++i)
                buffer[i] = m_ram[(core.transferAddress + i) & kRamMask];
            (void)m_memory.writeRam(core.madr, buffer.data(), bytes);
            ++m_stats.dmaReads;
        }
        core.transferAddress = (core.transferAddress + words) & kRamMask;
        core.dmaCompletionCycle = m_cycle + std::max<uint64_t>(static_cast<uint64_t>(bytes) / 2u, 64u);
    }

    void IopSpu2::finishDma(int coreIndex)
    {
        Core &core = m_cores[coreIndex];
        core.dmaBusy = false;
        core.admaRunning = false;
        core.chcr &= ~kDmaStart;
        core.bcr &= 0xFFFFu;
        if (m_interrupt)
            m_interrupt(DmaIrq[coreIndex]);
    }

    void IopSpu2::fillInputHalf(int coreIndex, int half)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t left = kInputBase + static_cast<uint32_t>(coreIndex) * 0x400u + static_cast<uint32_t>(half) * 0x100u;
        const uint32_t right = left + 0x200u;
        if (!core.admaRunning || core.admaRemaining == 0u)
        {
            std::fill_n(m_ram.begin() + left, 0x100u, uint16_t{0});
            std::fill_n(m_ram.begin() + right, 0x100u, uint16_t{0});
            core.inputHalfFilled[half] = false;
            if ((core.admas & (1u << coreIndex)) != 0u)
                ++m_stats.admaUnderruns;
            return;
        }

        std::array<uint16_t, 0x200u> chunk{};
        const uint32_t bytes = std::min<uint32_t>(core.admaRemaining, 0x400u);
        (void)m_memory.readRam(core.admaAddress, chunk.data(), bytes);
        std::copy_n(chunk.begin(), 0x100u, m_ram.begin() + left);
        std::copy_n(chunk.begin() + 0x100u, 0x100u, m_ram.begin() + right);
        core.inputHalfFilled[half] = true;
        core.admaAddress += bytes;
        core.admaRemaining -= bytes;
        core.madr = core.admaAddress;
        m_stats.admaBytes += bytes;
        checkIrq(left, 0x100u);
        checkIrq(right, 0x100u);
        if (core.admaRemaining == 0u)
            finishDma(coreIndex);
    }

    void IopSpu2::checkIrq(uint32_t address, uint32_t words)
    {
        for (int coreIndex = 0; coreIndex < 2; ++coreIndex)
        {
            const Core &core = m_cores[coreIndex];
            if ((core.attr & kAttrIrqEnable) == 0u)
                continue;
            if (((core.irqAddress - address) & kRamMask) >= words)
                continue;
            const uint16_t bit = static_cast<uint16_t>(4u << coreIndex);
            if ((m_irqInfo & bit) != 0u)
                continue;
            m_irqInfo |= bit;
            ++m_stats.irqs;
            if (m_interrupt)
                m_interrupt(Irq);
        }
    }

    void IopSpu2::decodeBlock(int coreIndex, Voice &voice)
    {
        (void)coreIndex;
        const uint32_t address = voice.nextAddress & ~7u & kRamMask;
        checkIrq(address, 8u);
        const uint16_t header = m_ram[address];
        uint32_t shift = header & 0xFu;
        uint32_t filter = (header >> 4u) & 7u;
        if (shift > 12u)
            shift = 9u;
        if (filter > 4u)
            filter = 4u;
        voice.blockFlags = static_cast<uint8_t>(header >> 8u);
        if ((voice.blockFlags & 4u) != 0u && !voice.loopAddressWritten)
            voice.loopAddress = address;

        voice.samples[0] = voice.samples[28];
        voice.samples[1] = voice.samples[29];
        voice.samples[2] = voice.samples[30];
        int32_t h0 = voice.history[0];
        int32_t h1 = voice.history[1];
        for (uint32_t i = 0; i < 28u; ++i)
        {
            const uint16_t word = m_ram[(address + 1u + (i >> 2u)) & kRamMask];
            const uint32_t nibble = (word >> ((i & 3u) * 4u)) & 0xFu;
            int32_t sample = static_cast<int16_t>(static_cast<uint16_t>(nibble << 12u)) >> shift;
            sample += (h0 * kFilterPositive[filter] + h1 * kFilterNegative[filter] + 32) >> 6;
            sample = clamp16(sample);
            h1 = h0;
            h0 = sample;
            voice.samples[3u + i] = static_cast<int16_t>(sample);
        }
        voice.history[0] = h0;
        voice.history[1] = h1;
    }

    void IopSpu2::advanceEnvelope(Voice &voice)
    {
        Envelope &env = voice.envelope;
        switch (env.phase)
        {
        case Envelope::Phase::Attack:
            envelopeStep(env.level, env.counter, (voice.adsr1 >> 8u) & 0x7Fu, false, (voice.adsr1 & 0x8000u) != 0u);
            if (env.level >= 0x7FFF)
            {
                env.phase = Envelope::Phase::Decay;
                env.counter = 0u;
            }
            break;
        case Envelope::Phase::Decay:
        {
            envelopeStep(env.level, env.counter, ((voice.adsr1 >> 4u) & 0xFu) << 2u, true, true);
            const int32_t sustain = std::min<int32_t>(((voice.adsr1 & 0xFu) + 1) * 0x800, 0x7FFF);
            if (env.level <= sustain)
            {
                env.phase = Envelope::Phase::Sustain;
                env.counter = 0u;
            }
            break;
        }
        case Envelope::Phase::Sustain:
            envelopeStep(env.level, env.counter, (voice.adsr2 >> 6u) & 0x7Fu, (voice.adsr2 & 0x4000u) != 0u,
                         (voice.adsr2 & 0x8000u) != 0u);
            break;
        case Envelope::Phase::Release:
            envelopeStep(env.level, env.counter, (voice.adsr2 & 0x1Fu) << 2u, true, (voice.adsr2 & 0x20u) != 0u);
            if (env.level <= 0)
            {
                env.level = 0;
                env.phase = Envelope::Phase::Off;
            }
            break;
        case Envelope::Phase::Off:
            break;
        }
    }

    int32_t IopSpu2::voiceSample(int coreIndex, int index, Voice &voice)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t position = voice.counter >> 12u;
        int32_t sample = (core.noiseOn & (1u << index)) != 0u
                             ? static_cast<int16_t>(static_cast<uint16_t>(m_noiseLevel))
                             : interpolate(voice.samples[position], voice.samples[position + 1u],
                                           voice.samples[position + 2u], voice.samples[position + 3u],
                                           voice.counter & 0xFFFu);
        advanceEnvelope(voice);
        sample = mul15(sample, voice.envelope.level);
        voice.output = sample;

        uint32_t step = std::min<uint32_t>(voice.pitch, 0x3FFFu);
        if (index > 0 && (core.pitchMod & (1u << index)) != 0u)
        {
            const int32_t factor = core.voices[index - 1].output + 0x8000;
            step = std::min<uint32_t>(static_cast<uint32_t>((static_cast<int32_t>(step) * factor) >> 15), 0x3FFFu);
        }
        voice.counter += step;
        while ((voice.counter >> 12u) >= 28u)
        {
            voice.counter -= 28u << 12u;
            if ((voice.blockFlags & 1u) != 0u)
            {
                core.endx |= 1u << index;
                voice.nextAddress = voice.loopAddress & ~7u;
                if ((voice.blockFlags & 2u) == 0u && (core.noiseOn & (1u << index)) == 0u)
                {
                    voice.envelope.phase = Envelope::Phase::Off;
                    voice.envelope.level = 0;
                }
            }
            else
            {
                voice.nextAddress = ((voice.nextAddress & ~7u) + 8u) & kRamMask;
            }
            decodeBlock(coreIndex, voice);
        }
        return sample;
    }

    IopSpu2::StereoSample IopSpu2::reverb(Core &core, const StereoSample &input)
    {
        if ((core.attr & kAttrFxEnable) == 0u || core.effectEnd <= core.effectStart)
            return {};
        // The reverb runs at 24 kHz; its output holds for two samples.
        if (!m_reverbPhase)
            return {core.reverbOut[0], core.reverbOut[1]};

        const uint32_t size = core.effectEnd - core.effectStart + 1u;
        const auto at = [&](uint32_t offset, int32_t delta = 0)
        {
            // Offsets are below the work area size in practice: wrap by subtraction.
            uint32_t relative = core.reverbPosition + offset + static_cast<uint32_t>(delta + static_cast<int32_t>(size));
            while (relative >= size)
                relative = relative - size >= size ? relative % size : relative - size;
            return (core.effectStart + relative) & kRamMask;
        };
        const auto load = [&](uint32_t offset, int32_t delta = 0)
        { return static_cast<int32_t>(static_cast<int16_t>(m_ram[at(offset, delta)])); };
        const auto store = [&](uint32_t offset, int32_t value)
        { m_ram[at(offset)] = static_cast<uint16_t>(clamp16(value)); };

        const auto &a = core.reverbAddress;
        const auto &c = core.reverbCoef;
        const int32_t vIIR = c[0], vCOMB1 = c[1], vCOMB2 = c[2], vCOMB3 = c[3], vCOMB4 = c[4];
        const int32_t vWALL = c[5], vAPF1 = c[6], vAPF2 = c[7], vLIN = c[8], vRIN = c[9];
        const uint32_t dAPF1 = a[0], dAPF2 = a[1];
        const uint32_t mLSAME = a[2], mRSAME = a[3], mLCOMB1 = a[4], mRCOMB1 = a[5], mLCOMB2 = a[6], mRCOMB2 = a[7];
        const uint32_t dLSAME = a[8], dRSAME = a[9], mLDIFF = a[10], mRDIFF = a[11];
        const uint32_t mLCOMB3 = a[12], mRCOMB3 = a[13], mLCOMB4 = a[14], mRCOMB4 = a[15];
        const uint32_t dLDIFF = a[16], dRDIFF = a[17], mLAPF1 = a[18], mRAPF1 = a[19], mLAPF2 = a[20], mRAPF2 = a[21];

        const int32_t lin = mul15(clamp16(input.left), vLIN);
        const int32_t rin = mul15(clamp16(input.right), vRIN);

        store(mLSAME, mul15(lin + mul15(load(dLSAME), vWALL) - load(mLSAME, -1), vIIR) + load(mLSAME, -1));
        store(mRSAME, mul15(rin + mul15(load(dRSAME), vWALL) - load(mRSAME, -1), vIIR) + load(mRSAME, -1));
        store(mLDIFF, mul15(lin + mul15(load(dRDIFF), vWALL) - load(mLDIFF, -1), vIIR) + load(mLDIFF, -1));
        store(mRDIFF, mul15(rin + mul15(load(dLDIFF), vWALL) - load(mRDIFF, -1), vIIR) + load(mRDIFF, -1));

        int32_t lout = mul15(vCOMB1, load(mLCOMB1)) + mul15(vCOMB2, load(mLCOMB2)) +
                       mul15(vCOMB3, load(mLCOMB3)) + mul15(vCOMB4, load(mLCOMB4));
        int32_t rout = mul15(vCOMB1, load(mRCOMB1)) + mul15(vCOMB2, load(mRCOMB2)) +
                       mul15(vCOMB3, load(mRCOMB3)) + mul15(vCOMB4, load(mRCOMB4));

        const auto allPass = [&](int32_t value, uint32_t destination, uint32_t delay, int32_t coefficient)
        {
            const int32_t delayed = load(destination, -static_cast<int32_t>(delay));
            value = clamp16(value - mul15(coefficient, delayed));
            store(destination, value);
            return mul15(value, coefficient) + delayed;
        };
        lout = allPass(lout, mLAPF1, dAPF1, vAPF1);
        rout = allPass(rout, mRAPF1, dAPF1, vAPF1);
        lout = allPass(lout, mLAPF2, dAPF2, vAPF2);
        rout = allPass(rout, mRAPF2, dAPF2, vAPF2);

        core.reverbPosition = core.reverbPosition + 1u >= size ? 0u : core.reverbPosition + 1u;
        core.reverbOut[0] = clamp16(lout);
        core.reverbOut[1] = clamp16(rout);
        return {core.reverbOut[0], core.reverbOut[1]};
    }

    IopSpu2::StereoSample IopSpu2::mixCore(int coreIndex, const StereoSample &external)
    {
        Core &core = m_cores[coreIndex];
        StereoSample dry;
        StereoSample wet;
        for (int index = 0; index < 24; ++index)
        {
            Voice &voice = core.voices[index];
            voice.volume[0].tick();
            voice.volume[1].tick();
            if (voice.envelope.phase == Envelope::Phase::Off)
            {
                voice.output = 0;
                continue;
            }
            const int32_t sample = voiceSample(coreIndex, index, voice);
            const int32_t left = mul15(sample, voice.volume[0].level);
            const int32_t right = mul15(sample, voice.volume[1].level);
            const uint32_t bit = 1u << index;
            if ((core.dryMix[0] & bit) != 0u)
                dry.left += left;
            if ((core.dryMix[1] & bit) != 0u)
                dry.right += right;
            if ((core.wetMix[0] & bit) != 0u)
                wet.left += left;
            if ((core.wetMix[1] & bit) != 0u)
                wet.right += right;
        }
        dry = {clamp16(dry.left), clamp16(dry.right)};
        wet = {clamp16(wet.left), clamp16(wet.right)};

        // Sound data input (AutoDMA), read whether or not a transfer feeds it.
        StereoSample input;
        if ((core.admas & (1u << coreIndex)) != 0u)
        {
            const uint32_t left = kInputBase + static_cast<uint32_t>(coreIndex) * 0x400u + core.inputPosition;
            checkIrq(left, 1u);
            checkIrq(left + 0x200u, 1u);
            input.left = mul15(static_cast<int16_t>(m_ram[left]), core.inputVolume[0]);
            input.right = mul15(static_cast<int16_t>(m_ram[left + 0x200u]), core.inputVolume[1]);
            core.inputPosition = (core.inputPosition + 1u) & 0x1FFu;
            if ((core.inputPosition & 0xFFu) == 0u)
                fillInputHalf(coreIndex, core.inputPosition == 0u ? 1 : 0);
        }

        const uint16_t g = core.mmix;
        StereoSample out;
        out.left = ((g & kDryInpL) ? input.left : 0) + ((g & kDrySndL) ? dry.left : 0) + ((g & kDryExtL) ? external.left : 0);
        out.right = ((g & kDryInpR) ? input.right : 0) + ((g & kDrySndR) ? dry.right : 0) + ((g & kDryExtR) ? external.right : 0);
        StereoSample send;
        send.left = ((g & kWetInpL) ? input.left : 0) + ((g & kWetSndL) ? wet.left : 0) + ((g & kWetExtL) ? external.left : 0);
        send.right = ((g & kWetInpR) ? input.right : 0) + ((g & kWetSndR) ? wet.right : 0) + ((g & kWetExtR) ? external.right : 0);
        const StereoSample effect = reverb(core, send);
        out.left += mul15(effect.left, core.effectVolume[0]);
        out.right += mul15(effect.right, core.effectVolume[1]);

        core.master[0].tick();
        core.master[1].tick();
        return {clamp16(mul15(clamp16(out.left), core.master[0].level)),
                clamp16(mul15(clamp16(out.right), core.master[1].level))};
    }

    void IopSpu2::renderSample()
    {
        // Noise generator (clock from core 0's ATTR bits 8-13).
        {
            const uint32_t clock = (m_cores[0].attr >> 8u) & 0x3Fu;
            const uint32_t step = 4u + (clock & 3u);
            const uint32_t shift = clock >> 2u;
            m_noiseCounter += step;
            const uint32_t period = 0x20000u >> shift;
            while (m_noiseCounter >= period)
            {
                m_noiseCounter -= period;
                const uint32_t parity = ((m_noiseLevel >> 15u) ^ (m_noiseLevel >> 12u) ^ (m_noiseLevel >> 11u) ^
                                         (m_noiseLevel >> 10u) ^ 1u) & 1u;
                m_noiseLevel = ((m_noiseLevel << 1u) | parity) & 0xFFFFu;
            }
        }

        m_reverbPhase = !m_reverbPhase;
        const StereoSample core0 = mixCore(0, {});
        const StereoSample external{mul15(core0.left, m_cores[1].externalVolume[0]),
                                    mul15(core0.right, m_cores[1].externalVolume[1])};
        const StereoSample out = mixCore(1, external);
        m_pending.push_back(static_cast<int16_t>(out.left));
        m_pending.push_back(static_cast<int16_t>(out.right));
        m_stats.peak = std::max({m_stats.peak, std::abs(out.left), std::abs(out.right)});
        ++m_stats.samples;
        if (m_pending.size() >= kOutputChunk * 2u)
            flush();
    }

    void IopSpu2::flush()
    {
        if (m_output && !m_pending.empty())
            m_output(m_pending.data(), m_pending.size() / 2u);
        m_pending.clear();
    }

    void IopSpu2::advanceTo(uint64_t cycle)
    {
        while (m_cycle < cycle)
        {
            uint64_t next = std::min(cycle, m_nextSampleCycle);
            for (const Core &core : m_cores)
                if (core.dmaBusy && !core.admaRunning)
                    next = std::min(next, std::max(core.dmaCompletionCycle, m_cycle + 1u));
            m_cycle = next;
            for (int coreIndex = 0; coreIndex < 2; ++coreIndex)
            {
                const Core &core = m_cores[coreIndex];
                if (core.dmaBusy && !core.admaRunning && core.dmaCompletionCycle <= m_cycle)
                    finishDma(coreIndex);
            }
            if (m_cycle >= m_nextSampleCycle)
            {
                renderSample();
                m_nextSampleCycle += CyclesPerSample;
            }
        }
        updateNextEvent();
    }

    void IopSpu2::updateNextEvent() noexcept
    {
        uint64_t next = m_nextSampleCycle + (kBatchSamples - 1u) * CyclesPerSample;
        for (const Core &core : m_cores)
        {
            if ((core.attr & kAttrIrqEnable) != 0u)
            {
                m_nextEvent = m_nextSampleCycle;
                return;
            }
            if (core.dmaBusy && !core.admaRunning)
                next = std::min(next, core.dmaCompletionCycle);
            if (core.admaRunning)
            {
                // The half-buffer boundary where the next chunk is fetched (and the DMA may end).
                const uint64_t samples = 0xFFu - (core.inputPosition & 0xFFu);
                next = std::min(next, m_nextSampleCycle + samples * CyclesPerSample);
            }
        }
        m_nextEvent = next;
    }

    std::vector<std::string> IopSpu2::describe() const
    {
        std::vector<std::string> lines;
        // Nothing to report until a driver touches the SPU2.
        if (m_stats.registerWrites == 0u)
            return lines;
        std::ostringstream out;
        out << "[SPU2] samples=" << m_stats.samples << " keyOn=" << m_stats.keyOns << " keyOff=" << m_stats.keyOffs
            << " dmaW=" << m_stats.dmaWrites << " (" << m_stats.dmaWriteBytes << " B) dmaR=" << m_stats.dmaReads
            << " adma=" << m_stats.admaTransfers << " (" << m_stats.admaBytes << " B, underruns "
            << m_stats.admaUnderruns << ") io=" << m_stats.ioWrites << " irq=" << m_stats.irqs
            << " regW=" << m_stats.registerWrites << " peak=" << m_stats.peak;
        lines.push_back(out.str());
        for (int coreIndex = 0; coreIndex < 2; ++coreIndex)
        {
            const Core &core = m_cores[coreIndex];
            uint32_t on = 0u;
            for (const Voice &voice : core.voices)
                on += voice.envelope.phase != Envelope::Phase::Off ? 1u : 0u;
            std::ostringstream line;
            line << std::hex << "[SPU2] core" << coreIndex << " attr=0x" << core.attr << " mmix=0x" << core.mmix
                 << " admas=0x" << core.admas << " mvol=0x" << core.master[0].reg << "/0x" << core.master[1].reg
                 << " evol=0x" << static_cast<uint16_t>(core.effectVolume[0]) << " bvol=0x"
                 << static_cast<uint16_t>(core.inputVolume[0]) << " avol=0x"
                 << static_cast<uint16_t>(core.externalVolume[0]) << " irqa=0x" << core.irqAddress << " esa=0x"
                 << core.effectStart << " eea=0x" << core.effectEnd << " vmix=0x" << core.dryMix[0] << "/0x"
                 << core.dryMix[1] << std::dec << " voicesOn=" << on;
            lines.push_back(line.str());
        }
        return lines;
    }
}
