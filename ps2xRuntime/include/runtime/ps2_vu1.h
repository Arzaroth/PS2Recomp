#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <array>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

class GS;
class PS2Memory;

struct VU1State
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t r;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    uint64_t cycles;
    bool ebit;
    bool haltAfterDelaySlot;
    bool dBitEnabled;
    bool tBitEnabled;
    bool stoppedByD;
    bool stoppedByT;
    uint32_t top;  // VIF TOP visible to XTOP
    uint32_t itop; // VIF ITOP visible to XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;
};

struct Vu1CompiledAccess;

class VU1Interpreter
{
    friend struct Vu1CompiledAccess;

public:
    enum class Unit : uint8_t
    {
        VU0,
        VU1
    };

    explicit VU1Interpreter(Unit unit = Unit::VU1);

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

    // Called with every PATH1 packet an XGKICK completes (recording and replay tooling).
    using PacketObserver = std::function<void(const uint8_t *data, uint32_t size)>;
    void setPacketObserver(PacketObserver observer) { m_packetObserver = std::move(observer); }
    // For callers running code that PS2Memory does not track: drop decoded instructions after changing it.
    void invalidateDecodedCode() { ++m_externalCodeGeneration; }

private:
    enum Pipeline : uint8_t
    {
        PipelineNone = 0,
        PipelineFmac,
        PipelineLsu,
        PipelineFdiv,
        PipelineEfu,
        PipelineIalu,
        PipelineBranch,
        PipelineXgkick
    };

    struct VfAccess
    {
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };

    struct InstructionUsage
    {
        std::array<VfAccess, 2> vfRead{};
        VfAccess vfWrite{};
        uint8_t vfReadCount = 0;
        uint16_t viRead = 0;
        uint16_t viWrite = 0;
        uint8_t accRead = 0;
        uint8_t accWrite = 0;
        uint8_t latency = 0;
        uint8_t vfLatency = 0;
        uint8_t viLatency = 0;
        Pipeline pipeline = PipelineNone;
        bool waitQ = false;
        bool waitP = false;
        bool readsClip = false;
        bool writesClip = false;
        bool delaysNextBranchRead = false;
        bool reserved = false;
    };

    using UpperHandler = void (VU1Interpreter::*)(uint32_t);
    enum class StepResult : uint8_t
    {
        Next,
        Stop,
        Ended
    };

    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        UpperHandler upperHandler = nullptr;
        InstructionUsage lowerUsage{};
        InstructionUsage upperUsage{};
        bool iBit = false;
        bool eBit = false;
        bool mBit = false;
        bool dBit = false;
        bool tBit = false;
        uint8_t upperVfShadowReg = 0;
        uint8_t suppressedLowerVf = 0;
        uint8_t writtenVi = 0;
    };

    struct FlagPipelineEntry
    {
        uint64_t readyCycle = 0;
        uint64_t issueCycle = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool valid = false;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;
    };

    struct ScalarPipelineEntry
    {
        uint64_t readyCycle = 0;
        float value = 0.0f;
        uint32_t statusDi = 0;
        bool valid = false;
    };

    struct XgkickPipeline
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        std::array<uint8_t, kBufferSize> packet{};
        uint32_t sourceAddress = 0;
        uint32_t totalBytes = 0;
        uint32_t copiedBytes = 0;
        uint32_t currentTagEnd = 0;
        uint32_t cycleCredit = 0;
        uint64_t issueCycle = 0;
        bool active = false;
        bool currentTagEop = false;
    };

    static constexpr uint32_t kFmacLatency = 4u;
    static constexpr uint32_t kAccForwardLatency = 1u;
    static constexpr uint32_t kMaxFlagEntries = 8u;
    static constexpr uint32_t kMaxDecodedPairs = 0x4000u / 8u;

    Unit m_unit;
    VU1State m_state;
    std::array<DecodedInstructionPair, kMaxDecodedPairs> m_decodedCodeCache{};
    std::array<uint64_t, kMaxDecodedPairs> m_decodedPairStamp{};
    DecodedInstructionPair m_uncachedDecodedPair{};
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;

    // VU1 only: per pair, whether a flag reader can observe its upper instruction's MAC and status result.
    // Computed once per code image (by content hash) and valid while the decode stamp is m_flagsLiveStamp.
    using FlagLiveness = std::array<bool, kMaxDecodedPairs>;
    // A microprogram compiled ahead of time (ps2_vu1_compile) for one VU0 or VU1 code image: runs pairs from
    // the current pc and returns Next where it has no code, for the interpreter to run that pair.
    using CompiledRun = StepResult (*)(VU1Interpreter &vu, uint64_t budgetEnd, uint8_t *vuData, uint32_t dataSize,
                                       GS &gs, PS2Memory *memory);
    struct CompiledProgram
    {
        uint64_t hash = 0;
        uint32_t codeSize = 0;
        CompiledRun run = nullptr;
    };
    static std::vector<CompiledProgram> &compiledPrograms();
    static uint64_t hashMicrocode(const uint8_t *code, uint32_t size);
    static void computeMicrocodeFlagLiveness(const uint8_t *code, uint32_t size, uint32_t pcMask, bool *live);
    struct CodeImageInfo
    {
        uint64_t hash = 0;
        FlagLiveness liveness{};
        CompiledRun compiled = nullptr;
    };
    std::vector<CodeImageInfo> m_flagLivenessCache;
    const FlagLiveness *m_flagsLive = nullptr;
    CompiledRun m_compiledRun = nullptr;
    uint64_t m_flagsLiveStamp = 0;
    bool m_elideDeadFlags = false;
    bool m_currentFlagsLive = true;
    void refreshFlagLiveness(const uint8_t *vuCode, uint32_t codeSize, uint64_t stamp);

    // Earliest readyCycle pending in the flag and Q/P queues; commitReadyPipelines() skips the rest.
    uint64_t m_nextFlagCommit = 0;
    uint64_t m_nextScalarCommit = 0;
    static void noteCommitCycle(uint64_t &next, uint64_t readyCycle)
    {
        if (readyCycle < next)
            next = readyCycle;
    }
    std::array<FlagPipelineEntry, kMaxFlagEntries> m_flagPipeline{}; // ring buffer in issue order
    uint32_t m_flagHead = 0;
    uint32_t m_flagCount = 0;
    FlagPipelineEntry *pushFlagEntry(bool upper, uint32_t overflowCode);
    template <typename Fn>
    void forEachFlagEntryIssuedThisCycle(Fn &&fn)
    {
        for (uint32_t index = m_flagCount; index-- > 0u;)
        {
            FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + index) % kMaxFlagEntries];
            if (entry.issueCycle != m_cycle)
                break;
            fn(entry);
        }
    }
    ScalarPipelineEntry m_fdiv{};
    std::array<ScalarPipelineEntry, 2> m_efu{};
    XgkickPipeline m_xgkick{};
    std::array<std::array<uint64_t, 4>, 32> m_vfReady{};
    std::array<uint64_t, 32> m_vfReadyMax{};
    std::array<uint64_t, 16> m_viReady{};
    std::array<uint64_t, 4> m_accReady{};

    // Register writes of the last few cycles with their previous values: a run that stops on its cycle
    // budget hides those still in flight, and the next run reveals them again.
    struct InFlightWrite
    {
        enum class Kind : uint8_t
        {
            Vf,
            Vi,
            Acc
        };
        uint64_t readyCycle = 0;
        float oldValue[4]{};
        float newValue[4]{};
        Kind kind = Kind::Vf;
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };
    static constexpr uint32_t kInFlightWrites = 16u;
    std::array<InFlightWrite, kInFlightWrites> m_inFlight{};
    uint32_t m_inFlightNext = 0;
    uint32_t m_inFlightCount = 0;
    bool m_inFlightHidden = false;
    bool m_logInFlight = false;
    void logInFlightWrite(InFlightWrite::Kind kind, uint8_t reg, uint8_t lanes, uint64_t readyCycle,
                          const float oldValue[4], const float newValue[4]);
    void applyInFlightWrite(const InFlightWrite &entry, bool newValue);
    void hideInFlightWrites();
    void revealInFlightWrites();

    uint64_t m_cycle = 0;
    uint64_t m_lastWritebackCycle = 0;
    uint64_t m_efuResourceReady = 0;
    uint32_t m_workingClip = 0;
    uint32_t m_currentUpperInstruction = 0;
    const float *m_fmacVs = nullptr;
    const float *m_fmacVt = nullptr;
    const float *m_fmacAcc = nullptr;
    float m_fmacQ = 0.0f;
    float m_fmacI = 0.0f;
    int32_t m_viBranchBackupValue = 0;
    uint8_t m_viBranchBackupReg = 0;
    bool m_viBranchBackupValid = false;
    uint8_t *m_activeVuData = nullptr;
    uint32_t m_activeVuDataSize = 0;
    GS *m_activeGs = nullptr;
    PS2Memory *m_activeMemory = nullptr;
    bool m_stopRequested = false;
    bool m_pendingHaltD = false;
    bool m_pendingHaltT = false;
    PacketObserver m_packetObserver;
    uint64_t m_externalCodeGeneration = 0;

    // upper() and lower() execute the pair's instructions: through decoded's handlers when interpreting,
    // with the instruction words as constants in compiled microprograms.
    template <typename Upper, typename Lower>
    StepResult stepPair(const DecodedInstructionPair &decoded, uint64_t budgetEnd, uint32_t codeSize, Upper &&upper,
                        Lower &&lower);
    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);

    InstructionUsage decodeUpperUsage(uint32_t upper) const;
    InstructionUsage decodeLowerUsage(uint32_t lower) const;
    static void addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static void addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static uint8_t vfReadLanes(const InstructionUsage &usage, uint8_t reg);
    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    const DecodedInstructionPair &getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory,
                                                                 uint32_t pc);

    void execUpper(uint32_t instr);
    // execUpper specialized by decodeInstructionPair for the common FMAC arithmetic and NOP.
    enum class FmacKind : uint8_t
    {
        Add,
        Sub,
        Madd,
        Msub,
        Mul
    };
    enum class FmacOperand : uint8_t
    {
        Broadcast,
        Q,
        I,
        Vector
    };
    template <FmacKind Kind, FmacOperand Operand, bool ToAcc>
    void execFmac(uint32_t instr);
    void execUpperNop(uint32_t instr);
    static UpperHandler selectUpperHandler(uint32_t upper);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    bool exactZeroLanes(int zeroComponents) const;
    void applyFmacFlags(float *result, uint8_t dest);
    void applyFmacDest(float *dst, float *result, uint8_t dest);
    void applyFmacDestAcc(float *result, uint8_t dest);
    void normalizeFmacResult(float *result, uint8_t dest, uint8_t laneFlags[4]);
    bool calculateFmacExactResult(uint32_t component, double &result) const;
    uint8_t normalizeFmacExactResult(float &value, double exactResult) const;
    uint32_t calculateFmacProductSticky(uint8_t dest) const;
    void updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky);
    void queueFsset(uint16_t immediate);
    void queueClip(uint32_t clip);
    void queueFcset(uint32_t clip);
    void queueQ(float value, uint32_t latency, uint32_t statusDi);
    void queueP(float value, uint32_t latency);
    void queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask);
    void startXgkick(uint32_t qwordAddress);

    void resetScheduler();
    void commitReadyPipelines();
    void advanceOneCycle();
    void advanceTo(uint64_t targetCycle);
    void flushPipelines();
    void progressXgkick(uint64_t cycles);
    // Cycles of PATH1 progress not applied yet: the transfer only has to catch up before VU memory is
    // written, before an XGKICK checks whether PATH1 is busy and when the run stops.
    uint64_t m_xgkickBacklog = 0;
    void catchUpXgkick()
    {
        if (m_xgkickBacklog != 0u)
            progressXgkick(std::exchange(m_xgkickBacklog, uint64_t{0}));
    }
    void resetXgkick();
    void finishXgkick();
    uint64_t calculatePairReadyCycle(const DecodedInstructionPair &decoded) const;
    void markPairWrites(const DecodedInstructionPair &decoded);
    bool pipelinesPending() const;

    float normalizeOperand(float value) const;
    float normalizeResult(float value, uint32_t &laneFlags) const;
    uint32_t microAddressMask() const;
    int32_t readBranchVi(uint8_t reg) const;
    void recordViWriteForBranch(uint8_t reg, int32_t oldValue);
    void reportReservedInstruction(bool upper, uint32_t instruction);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
