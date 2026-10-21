#include "MiniTest.h"
#include "ps2_audio_stretch.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace
{
    // Feeds a 440 Hz tone at `pace` times real time while the host pulls 1024 frames at a time.
    // Returns the pulled frames (left channel).
    std::vector<int16_t> run(AudioTimeStretcher &stretcher, double pace, size_t pulls, size_t &phase)
    {
        std::vector<int16_t> pulled;
        std::vector<int16_t> chunk;
        std::vector<int16_t> out(1024u * 2u);
        double owed = 0.0;
        for (size_t p = 0; p < pulls; ++p)
        {
            owed += 1024.0 * pace;
            const size_t frames = static_cast<size_t>(owed);
            owed -= static_cast<double>(frames);
            chunk.resize(frames * 2u);
            for (size_t i = 0; i < frames; ++i, ++phase)
            {
                const int16_t value = static_cast<int16_t>(8000.0 * std::sin(2.0 * 3.14159265358979 * 440.0 * phase / 48000.0));
                chunk[i * 2u] = value;
                chunk[i * 2u + 1u] = value;
            }
            stretcher.push(chunk.data(), frames);
            stretcher.pull(out.data(), 1024u);
            for (size_t i = 0; i < 1024u; ++i)
                pulled.push_back(out[i * 2u]);
        }
        return pulled;
    }

    // Zero crossings per second over the last second: twice the frequency of a tone.
    size_t crossingsInLastSecond(const std::vector<int16_t> &samples)
    {
        size_t crossings = 0;
        for (size_t i = samples.size() - 48000u + 1u; i < samples.size(); ++i)
            crossings += (samples[i - 1u] < 0) != (samples[i] < 0) ? 1u : 0u;
        return crossings;
    }

    // Longest run of near-zero samples: a gap in the sound.
    size_t longestSilence(const std::vector<int16_t> &samples, size_t from)
    {
        size_t longest = 0;
        size_t current = 0;
        for (size_t i = from; i < samples.size(); ++i)
        {
            current = std::abs(samples[i]) < 50 ? current + 1u : 0u;
            longest = std::max(longest, current);
        }
        return longest;
    }
}

void register_audio_stretch_tests()
{
    MiniTest::Case("AudioTimeStretcher", [](TestCase &tc)
                   {
        tc.Run("real-time input plays without gaps", [](TestCase &t)
               {
            AudioTimeStretcher stretcher;
            size_t phase = 0;
            const std::vector<int16_t> out = run(stretcher, 1.0, 400u, phase);
            t.Equals(stretcher.underruns(), static_cast<uint64_t>(0), "no underrun at real-time pace");
            t.IsTrue(longestSilence(out, 48000u) < 20u, "no silent gap once primed");
            t.IsTrue(std::abs(stretcher.rate() - 1.0) < 0.1, "reads at about real time");
        });

        tc.Run("slow emulation is stretched instead of cut", [](TestCase &t)
               {
            AudioTimeStretcher stretcher;
            size_t phase = 0;
            (void)run(stretcher, 1.0, 20u, phase);
            const std::vector<int16_t> out = run(stretcher, 0.7, 600u, phase);
            t.Equals(stretcher.underruns(), static_cast<uint64_t>(0), "no underrun at 70% pace");
            t.IsTrue(longestSilence(out, 48000u) < 20u, "no silent gap at 70% pace");
            t.IsTrue(std::abs(stretcher.rate() - 0.7) < 0.1, "reads at the emulation's pace");
            const size_t crossings = crossingsInLastSecond(out);
            t.IsTrue(crossings > 850u && crossings < 910u, "a 440 Hz tone keeps its pitch");
        });

        tc.Run("an empty buffer fades to silence", [](TestCase &t)
               {
            AudioTimeStretcher stretcher;
            size_t phase = 0;
            (void)run(stretcher, 1.0, 50u, phase);
            const std::vector<int16_t> out = run(stretcher, 0.0, 50u, phase);
            t.IsTrue(stretcher.underruns() > 0u, "the starved stream underruns");
            t.IsTrue(longestSilence(out, 0u) > 30000u, "then stays silent");
        });
    });
}
