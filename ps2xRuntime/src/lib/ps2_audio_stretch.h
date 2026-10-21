#pragma once

// Time stretcher between an emulated audio producer and the host's real-time playback.
// The producer pushes 48 kHz stereo frames at the emulation's pace, which may run slower or
// faster than real time; the host pulls frames at the device rate. The stretcher keeps a buffer
// of about kTargetFrames: it reads its input faster or slower than real time depending on the
// fill level, by WSOLA (overlap-added 1024-frame grains, each aligned on the waveform that
// continues the previous one), which changes the duration without changing the pitch. Only an
// empty buffer silences the output, faded out by the last grain's window.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

class AudioTimeStretcher
{
public:
    static constexpr size_t kCapacityFrames = 32768u;
    static constexpr size_t kTargetFrames = 4800u;
    static constexpr size_t kGrainFrames = 1024u;
    static constexpr size_t kHopFrames = kGrainFrames / 2u;
    static constexpr size_t kSearchFrames = 256u;
    static constexpr size_t kCompareFrames = 256u;
    static constexpr double kMinRate = 0.5;
    static constexpr double kMaxRate = 2.0;

    AudioTimeStretcher()
        : m_ring(kCapacityFrames * 2u), m_tail(kHopFrames * 2u, 0.0f), m_window(kGrainFrames)
    {
        // Periodic Hann: w[i] + w[i + hop] == 1, so contiguous grains add back to the input.
        for (size_t i = 0; i < kGrainFrames; ++i)
            m_window[i] = 0.5f - 0.5f * static_cast<float>(std::cos(2.0 * 3.14159265358979323846 * i / kGrainFrames));
    }

    // Producer side. When the buffer overflows, the oldest frames are dropped.
    void push(const int16_t *frames, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (m_fill == kCapacityFrames)
                discard(1u);
            const size_t slot = (m_readFrame + m_fill) % kCapacityFrames;
            m_ring[slot * 2u] = frames[i * 2u];
            m_ring[slot * 2u + 1u] = frames[i * 2u + 1u];
            ++m_fill;
        }
    }

    // Consumer side: always fills `count` frames.
    void pull(int16_t *out, size_t count)
    {
        size_t written = 0;
        while (written < count)
        {
            if (m_outPos == m_out.size())
            {
                m_out.clear();
                m_outPos = 0;
                makeGrain();
            }
            const size_t take = std::min(count - written, (m_out.size() - m_outPos) / 2u);
            std::copy_n(m_out.begin() + static_cast<std::ptrdiff_t>(m_outPos), take * 2u, out + written * 2u);
            m_outPos += take * 2u;
            written += take;
        }
    }

    [[nodiscard]] size_t fill() const noexcept { return m_fill; }
    [[nodiscard]] double rate() const noexcept { return m_rate; }
    [[nodiscard]] uint64_t underruns() const noexcept { return m_underruns; }

private:
    [[nodiscard]] float sample(size_t frame, size_t channel) const noexcept
    {
        return m_ring[((m_readFrame + frame) % kCapacityFrames) * 2u + channel];
    }

    [[nodiscard]] float mono(size_t frame) const noexcept { return sample(frame, 0u) + sample(frame, 1u); }

    void discard(size_t frames) noexcept
    {
        frames = std::min(frames, m_fill);
        m_readFrame = (m_readFrame + frames) % kCapacityFrames;
        m_fill -= frames;
        m_position -= static_cast<double>(frames);
        m_previous -= static_cast<std::ptrdiff_t>(frames);
    }

    // Start of the grain whose beginning best continues the previous grain's second half.
    [[nodiscard]] size_t bestStart(size_t nominal) const noexcept
    {
        if (m_previous < 0)
            return nominal;
        const size_t natural = static_cast<size_t>(m_previous) + kHopFrames;
        if (natural + kCompareFrames > m_fill)
            return nominal;
        const size_t first = nominal > kSearchFrames ? nominal - kSearchFrames : 0u;
        const size_t last = std::min(nominal + kSearchFrames, m_fill - kGrainFrames);
        size_t best = nominal;
        double bestScore = -1e300;
        for (size_t start = first; start <= last; start += 2u)
        {
            double dot = 0.0;
            double energy = 1.0;
            for (size_t k = 0; k < kCompareFrames; k += 4u)
            {
                const double candidate = mono(start + k);
                dot += candidate * mono(natural + k);
                energy += candidate * candidate;
            }
            const double score = dot / std::sqrt(energy);
            if (score > bestScore)
            {
                bestScore = score;
                best = start;
            }
        }
        return best;
    }

    void makeGrain()
    {
        if (!m_primed)
        {
            if (m_fill < kTargetFrames)
            {
                emitTail();
                return;
            }
            m_primed = true;
            m_position = 0.0;
            m_previous = -1;
            m_rate = 1.0;
        }

        // Read faster when the buffer is above its target, slower below.
        const double wanted = std::clamp(static_cast<double>(m_fill) / kTargetFrames, kMinRate, kMaxRate);
        m_rate += (wanted - m_rate) * 0.05;

        const size_t nominal = static_cast<size_t>(std::max(0.0, m_position));
        if (nominal + kGrainFrames > m_fill)
        {
            ++m_underruns;
            m_primed = false;
            emitTail();
            return;
        }
        const size_t start = bestStart(nominal);

        m_out.resize(kHopFrames * 2u);
        for (size_t i = 0; i < kHopFrames; ++i)
        {
            for (size_t channel = 0; channel < 2u; ++channel)
            {
                const float value = m_tail[i * 2u + channel] + m_window[i] * sample(start + i, channel);
                m_out[i * 2u + channel] = static_cast<int16_t>(std::clamp(value, -32768.0f, 32767.0f));
                m_tail[i * 2u + channel] = m_window[kHopFrames + i] * sample(start + kHopFrames + i, channel);
            }
        }
        m_previous = static_cast<std::ptrdiff_t>(start);
        m_position += kHopFrames * m_rate;

        // Keep a search margin behind the frames still needed.
        const double keepFrom = std::min(static_cast<double>(start), m_position) - static_cast<double>(kSearchFrames);
        if (keepFrom > 0.0)
            discard(static_cast<size_t>(keepFrom));
    }

    // The previous grain's windowed second half, then silence.
    void emitTail()
    {
        m_out.resize(kHopFrames * 2u);
        for (size_t i = 0; i < kHopFrames * 2u; ++i)
        {
            m_out[i] = static_cast<int16_t>(std::clamp(m_tail[i], -32768.0f, 32767.0f));
            m_tail[i] = 0.0f;
        }
    }

    std::vector<int16_t> m_ring;
    size_t m_readFrame = 0;
    size_t m_fill = 0;
    std::vector<float> m_tail;
    std::vector<float> m_window;
    std::vector<int16_t> m_out;
    size_t m_outPos = 0;
    double m_position = 0.0;
    std::ptrdiff_t m_previous = -1;
    double m_rate = 1.0;
    bool m_primed = false;
    uint64_t m_underruns = 0;
};
