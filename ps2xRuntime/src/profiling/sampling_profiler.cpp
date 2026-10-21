// Opt-in in-process sampling profiler (Windows): PS2X_PROFILE=<seconds> samples every thread of the
// process roughly every millisecond for that long (after an optional PS2X_PROFILE_DELAY=<seconds>), then
// writes "<thread id> <leaf rva hex> <first in-exe caller rva hex> <count>" lines to PS2X_PROFILE_OUT
// (default ps2x_profile.txt) for symbolization against the linker map file. For threads blocked in
// system calls the leaf rva is 0 and the caller column says where in the runtime they are waiting.
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#include <tlhelp32.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <tuple>

namespace
{
    void runSamplingProfiler(double delaySeconds, double durationSeconds, std::string outPath)
    {
        std::this_thread::sleep_for(std::chrono::duration<double>(delaySeconds));
        const DWORD self = GetCurrentThreadId();
        const DWORD pid = GetCurrentProcessId();
        const auto moduleBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
        const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(moduleBase);
        const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(moduleBase + dos->e_lfanew);
        const uintptr_t moduleEnd = moduleBase + nt->OptionalHeader.SizeOfImage;
        auto inExe = [&](uintptr_t address) { return address >= moduleBase && address < moduleEnd; };

        std::map<std::tuple<DWORD, uintptr_t, uintptr_t>, uint64_t> histogram;
        std::map<DWORD, HANDLE> threads;
        uint64_t samples = 0u;

        timeBeginPeriod(1);
        const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(durationSeconds);
        auto nextRefresh = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() < end)
        {
            if (std::chrono::steady_clock::now() >= nextRefresh)
            {
                nextRefresh += std::chrono::milliseconds(500);
                HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                THREADENTRY32 entry{};
                entry.dwSize = sizeof(entry);
                if (snapshot != INVALID_HANDLE_VALUE && Thread32First(snapshot, &entry))
                {
                    do
                    {
                        if (entry.th32OwnerProcessID == pid && entry.th32ThreadID != self &&
                            threads.find(entry.th32ThreadID) == threads.end())
                        {
                            HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                                                       FALSE, entry.th32ThreadID);
                            if (handle)
                                threads[entry.th32ThreadID] = handle;
                        }
                    } while (Thread32Next(snapshot, &entry));
                }
                if (snapshot != INVALID_HANDLE_VALUE)
                    CloseHandle(snapshot);
            }

            for (const auto &[tid, handle] : threads)
            {
                if (SuspendThread(handle) == static_cast<DWORD>(-1))
                    continue;
                CONTEXT context{};
                context.ContextFlags = CONTEXT_FULL;
                if (GetThreadContext(handle, &context))
                {
                    const uintptr_t leaf = static_cast<uintptr_t>(context.Rip);
                    uintptr_t caller = inExe(leaf) ? leaf : 0u;
                    // Unwind until the first frame inside the exe (what a blocked thread is waiting in).
                    for (int depth = 0; depth < 48 && caller == 0u; ++depth)
                    {
                        DWORD64 imageBase = 0;
                        PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr);
                        if (function)
                        {
                            PVOID handlerData = nullptr;
                            DWORD64 establisherFrame = 0;
                            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, function, &context,
                                             &handlerData, &establisherFrame, nullptr);
                        }
                        else
                        {
                            context.Rip = *reinterpret_cast<DWORD64 *>(context.Rsp);
                            context.Rsp += 8;
                        }
                        if (context.Rip == 0)
                            break;
                        if (inExe(static_cast<uintptr_t>(context.Rip)))
                            caller = static_cast<uintptr_t>(context.Rip);
                    }
                    ++histogram[{tid, inExe(leaf) ? leaf - moduleBase : 0u, caller ? caller - moduleBase : 0u}];
                    ++samples;
                }
                ResumeThread(handle);
            }
            Sleep(1);
        }
        timeEndPeriod(1);

        if (FILE *file = std::fopen(outPath.c_str(), "w"))
        {
            for (const auto &[key, count] : histogram)
                std::fprintf(file, "%lu %llx %llx %llu\n", std::get<0>(key),
                             static_cast<unsigned long long>(std::get<1>(key)),
                             static_cast<unsigned long long>(std::get<2>(key)),
                             static_cast<unsigned long long>(count));
            std::fclose(file);
        }
        for (const auto &[tid, handle] : threads)
            CloseHandle(handle);
        std::fprintf(stderr, "[profiler] wrote %llu samples to %s\n", static_cast<unsigned long long>(samples), outPath.c_str());
    }

    struct SamplingProfilerStarter
    {
        SamplingProfilerStarter()
        {
            const char *duration = std::getenv("PS2X_PROFILE");
            if (!duration || std::atof(duration) <= 0.0)
                return;
            const char *delay = std::getenv("PS2X_PROFILE_DELAY");
            const char *out = std::getenv("PS2X_PROFILE_OUT");
            std::thread(runSamplingProfiler, delay ? std::atof(delay) : 0.0, std::atof(duration),
                        std::string(out && *out ? out : "ps2x_profile.txt"))
                .detach();
        }
    };

    const SamplingProfilerStarter g_samplingProfilerStarter;
}
#endif
