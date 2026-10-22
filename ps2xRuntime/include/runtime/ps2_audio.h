#ifndef PS2_AUDIO_H
#define PS2_AUDIO_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

class PS2AudioBackend
{
public:
    PS2AudioBackend();
    ~PS2AudioBackend();

    void onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes);
    void onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr);
    void onSoundCommand(uint32_t sid, uint32_t rpcNum,
                        const uint8_t *sendBuf, uint32_t sendSize,
                        uint8_t *recvBuf, uint32_t recvSize);

    void play(uint32_t sampleAddr, float pitch = 1.0f, float volume = 1.0f,
              uint32_t voiceIndex = 0xFFFFFFFFu);
    void stop(uint32_t voiceId);
    void stopAll();
    void setAudioReady(bool ready) { m_audioReady = ready; }

    // SPU2 output of the IOP emulator: interleaved stereo frames at 48 kHz of emulated time,
    // played through one raylib stream that follows the emulation speed.
    void pushSpu2Frames(const int16_t *frames, size_t count);
    // Adds host audio (e.g. a movie soundtrack) into the SPU2 output, in place, so it follows
    // emulated time. Called on the thread that pushes SPU2 frames.
    using Spu2Mixer = std::function<void(int16_t *frames, size_t count)>;
    void setSpu2Mixer(Spu2Mixer mixer);
    // Before CloseAudioDevice.
    void closeSpu2Stream();
    struct Spu2Stream;

private:
    struct DecodedSample
    {
        std::vector<int16_t> pcm;
        uint32_t sampleRate = 44100;
    };

    struct Impl;
    std::unique_ptr<Impl> m_impl;
    struct Spu2StreamDeleter
    {
        void operator()(Spu2Stream *stream) const;
    };
    std::unique_ptr<Spu2Stream, Spu2StreamDeleter> m_spu2;
    bool m_audioReady = false;
    std::mutex m_spu2MixerMutex;
    Spu2Mixer m_spu2Mixer;
    std::vector<int16_t> m_spu2MixBuffer;
    uint32_t m_mostRecentSampleKey = 0;
    std::vector<DecodedSample> m_loadOrderSamples;
    std::vector<uint32_t> m_loadOrderSampleKeys;
    std::unordered_map<uint32_t, DecodedSample> m_sampleBank;
    std::mutex m_mutex;

    void playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                          bool isBgm = false);
    void pruneFinishedSounds();
};

#endif
