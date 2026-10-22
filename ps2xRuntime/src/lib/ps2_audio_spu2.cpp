// SPU2 output: the IOP emulator mixes 48 kHz stereo in emulated time; a raylib stream plays it in
// real time. A time stretcher sits between the two, so that an emulation running slower or faster
// than real time slows down or speeds up the sound (same pitch) instead of cutting it.
// PS2X_SPU2_WAV=<path> also records the output to a WAV file.

#include "runtime/ps2_audio.h"
#include "ps2_audio_stretch.h"
#include "ps2_host_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace
{
    constexpr uint32_t kSampleRate = 48000u;
    constexpr unsigned int kStreamBufferFrames = 1024u;

    void writeWavHeader(std::ofstream &file, uint32_t dataBytes)
    {
        const auto u32 = [&](uint32_t value)
        { file.write(reinterpret_cast<const char *>(&value), 4); };
        const auto u16 = [&](uint16_t value)
        { file.write(reinterpret_cast<const char *>(&value), 2); };
        file.write("RIFF", 4);
        u32(36u + dataBytes);
        file.write("WAVEfmt ", 8);
        u32(16u);
        u16(1u);
        u16(2u);
        u32(kSampleRate);
        u32(kSampleRate * 4u);
        u16(4u);
        u16(16u);
        file.write("data", 4);
        u32(dataBytes);
    }
}

struct PS2AudioBackend::Spu2Stream
{
#if !defined(PLATFORM_VITA)
    AudioStream stream{};
#endif
    bool playing = false;
    std::mutex mutex;
    AudioTimeStretcher stretcher;
    std::ofstream wav;
    uint32_t wavBytes = 0;

    void push(const int16_t *frames, size_t count)
    {
        std::lock_guard<std::mutex> lock(mutex);
        stretcher.push(frames, count);
        if (wav.is_open())
        {
            wav.write(reinterpret_cast<const char *>(frames), static_cast<std::streamsize>(count * 4u));
            wavBytes += static_cast<uint32_t>(count * 4u);
        }
    }

    void pull(int16_t *out, unsigned int frames)
    {
        std::lock_guard<std::mutex> lock(mutex);
        stretcher.pull(out, frames);
    }

    void closeWav()
    {
        if (!wav.is_open())
            return;
        wav.seekp(0);
        writeWavHeader(wav, wavBytes);
        wav.close();
    }
};

namespace
{
    std::atomic<PS2AudioBackend::Spu2Stream *> g_spu2Stream{nullptr};

#if !defined(PLATFORM_VITA)
    void spu2StreamCallback(void *buffer, unsigned int frames)
    {
        if (PS2AudioBackend::Spu2Stream *stream = g_spu2Stream.load())
            stream->pull(static_cast<int16_t *>(buffer), frames);
        else
            std::memset(buffer, 0, static_cast<size_t>(frames) * 4u);
    }
#endif
}

void PS2AudioBackend::Spu2StreamDeleter::operator()(Spu2Stream *stream) const
{
    delete stream;
}

void PS2AudioBackend::pushSpu2Frames(const int16_t *frames, size_t count)
{
    if (!frames || count == 0u)
        return;
    if (!m_spu2)
    {
        m_spu2.reset(new Spu2Stream());
        if (const char *path = std::getenv("PS2X_SPU2_WAV"); path && *path)
        {
            m_spu2->wav.open(path, std::ios::binary | std::ios::trunc);
            if (m_spu2->wav.is_open())
                writeWavHeader(m_spu2->wav, 0u);
        }
#if !defined(PLATFORM_VITA)
        if (m_audioReady && IsAudioDeviceReady())
        {
            SetAudioStreamBufferSizeDefault(static_cast<int>(kStreamBufferFrames));
            m_spu2->stream = LoadAudioStream(kSampleRate, 16u, 2u);
            SetAudioStreamBufferSizeDefault(0);
            if (IsAudioStreamValid(m_spu2->stream))
            {
                g_spu2Stream.store(m_spu2.get());
                SetAudioStreamCallback(m_spu2->stream, spu2StreamCallback);
                PlayAudioStream(m_spu2->stream);
                m_spu2->playing = true;
            }
        }
#endif
    }
    {
        std::lock_guard<std::mutex> lock(m_spu2MixerMutex);
        if (m_spu2Mixer)
        {
            m_spu2MixBuffer.assign(frames, frames + count * 2u);
            m_spu2Mixer(m_spu2MixBuffer.data(), count);
            frames = m_spu2MixBuffer.data();
        }
    }
    m_spu2->push(frames, count);
}

void PS2AudioBackend::setSpu2Mixer(Spu2Mixer mixer)
{
    std::lock_guard<std::mutex> lock(m_spu2MixerMutex);
    m_spu2Mixer = std::move(mixer);
}

void PS2AudioBackend::closeSpu2Stream()
{
    if (!m_spu2)
        return;
#if !defined(PLATFORM_VITA)
    if (m_spu2->playing && IsAudioDeviceReady())
    {
        StopAudioStream(m_spu2->stream);
        g_spu2Stream.store(nullptr);
        UnloadAudioStream(m_spu2->stream);
    }
#endif
    g_spu2Stream.store(nullptr);
    m_spu2->playing = false;
    m_spu2->closeWav();
    m_spu2.reset();
}
