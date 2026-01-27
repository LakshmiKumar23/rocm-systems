/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RegistrationMPITests.cpp
 * @brief Tests for buffer registration functionality in RCCL
 *
 * This file contains tests for:
 * 1. User Buffer Registration (UBR) - explicit buffer registration via ncclCommRegister
 * 2. Graph Capture Registration - automatic buffer registration during HIP graph capture
 *
 * REQUIRED Environment Variables:
 *   NCCL_DEBUG=INFO              Enable debug logging
 *   NCCL_DEBUG_SUBSYS=REG        Enable REG subsystem logging
 *   NCCL_LOCAL_REGISTER=1        Enable local buffer registration (UBR tests)
 *   NCCL_GRAPH_REGISTER=1        Enable graph buffer registration (Graph tests)
 *   RCCL_MPI_LOG_ALL_RANKS=1     Enable per-rank logging for log verification
 *
 * Run examples:
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=UBR_*
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=GraphCapture_*
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include <cstdlib>
#include <sstream>
#include <fstream>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

// Test Configuration
namespace RegTestConfig {
    constexpr size_t SMALL_COUNT  = 1024;           // 4KB for float
    constexpr size_t MEDIUM_COUNT = 256 * 1024;     // 1MB for float
    constexpr size_t LARGE_COUNT  = 1024 * 1024;    // 4MB for float

    using DefaultType = hip_bfloat16;

    constexpr int MIN_RANKS_DEFAULT    = 2;
    constexpr int MIN_RANKS_ALLTOALL   = 4;
    constexpr int MIN_NODES_MULTINODE  = 2;
}

// REG Log Checker - Pattern checking for registration debug output
class REGLogChecker
{
public:
    explicit REGLogChecker(const std::string& logContent)
        : m_content(logContent) {}

    bool hasIPCRegistration() const
    {
        return hasPattern("IPC register buffer") ||
               hasPattern("IPC registering buffer") ||
               (hasPattern("Proxy rank") && hasPattern("register success"));
    }

    bool hasIPCReuse() const
    {
        return hasPattern("IPC reuse buffer");
    }

    bool hasNETRegistration() const
    {
        return hasPattern("NET register userbuff") ||
               hasPattern("NET reuse buffer");
    }

    bool hasAnyRegistrationSuccess() const
    {
        return hasIPCRegistration() || hasIPCReuse() || hasNETRegistration();
    }

    bool hasIPCFailure() const
    {
        return hasPattern("failed to IPC register") ||
               hasPattern("legacy IPC blocked");
    }

    std::string getSummary() const
    {
        std::ostringstream ss;
        ss << "REG Log: ";
        if (hasIPCReuse()) ss << "[IPC-REUSE] ";
        if (hasIPCRegistration()) ss << "[IPC-REG] ";
        if (hasNETRegistration()) ss << "[NET-REG] ";
        if (hasIPCFailure()) ss << "[IPC-FAIL] ";
        if (!hasAnyRegistrationSuccess() && !hasIPCFailure()) ss << "[NO-REG]";
        return ss.str();
    }

    size_t getContentLength() const { return m_content.size(); }

private:
    bool hasPattern(const std::string& pattern) const
    {
        return m_content.find(pattern) != std::string::npos;
    }

    std::string m_content;
};

// Registration Test Base Class
class RegistrationTestBase : public MPITestBase
{
protected:
    struct RegInfo {
        void* buffer = nullptr;
        void* handle = nullptr;
        size_t size = 0;
        bool registered = false;
    };

    // Buffer Management
    RegInfo allocateAndRegister(size_t size)
    {
        RegInfo info;
        info.size = size;

        if (hipMalloc(&info.buffer, size) != hipSuccess) {
            return info;
        }

        ncclResult_t result = ncclCommRegister(getActiveCommunicator(),
                                                info.buffer, size, &info.handle);
        info.registered = (result == ncclSuccess && info.handle != nullptr);

        return info;
    }

    void cleanupRegInfo(RegInfo& info)
    {
        if (info.handle) {
            ncclCommDeregister(getActiveCommunicator(), info.handle);
            info.handle = nullptr;
        }
        if (info.buffer) {
            hipFree(info.buffer);
            info.buffer = nullptr;
        }
        info.registered = false;
    }

    // Test Setup
    bool setupMultiNode(int minRanks = 2, int minNodes = 2)
    {
        int nodeCount = MPITestConstants::detectNodeCount();
        if (nodeCount < minNodes) {
            return false;
        }
        if (!validateTestPrerequisites(minRanks, kNoProcessLimit,
                                        kNoPowerOfTwoRequired, minNodes, kNoNodeLimit)) {
            return false;
        }
        return (createTestCommunicator() == ncclSuccess);
    }

    // Environment Checks
    bool isUBREnabled()
    {
        const char* localReg = getenv("NCCL_LOCAL_REGISTER");
        return (localReg && std::string(localReg) == "1");
    }

    bool isGraphRegisterEnabled()
    {
        const char* graphReg = getenv("NCCL_GRAPH_REGISTER");
        return (graphReg && std::string(graphReg) == "1");
    }

    bool isPerRankLoggingEnabled()
    {
        const char* env = getenv("RCCL_MPI_LOG_ALL_RANKS");
        return (env && std::string(env) == "1");
    }

    // Log File Access
    std::string readRankLogFile()
    {
        std::string logPath = MPIHelpers::getRankLogFilePath(getTestMpiRank());
        std::ifstream file(logPath);
        if (!file.is_open()) {
            TEST_WARN("Could not open rank log file: %s", logPath.c_str());
            return "";
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    REGLogChecker getLogChecker()
    {
        return REGLogChecker(readRankLogFile());
    }

    // Data Initialization and Verification
    template<typename T>
    void initSendBuffer(void* buffer, size_t count, int rank)
    {
        initializeBufferWithPattern<T>(buffer, count,
            [rank](size_t) { return static_cast<T>(static_cast<float>(rank + 1)); });
    }

    template<typename T>
    bool verifyAllReduceResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyReduceScatterResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyAllGatherResult(void* buffer, size_t countPerRank, int nRanks)
    {
        return verifyBufferData<T>(buffer, countPerRank * nRanks,
            [countPerRank](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank + 1));
            });
    }

    template<typename T>
    bool verifyBroadcastResult(void* buffer, size_t count, T rootValue)
    {
        return verifyBufferData<T>(buffer, count,
            [rootValue](size_t) { return rootValue; });
    }
};

// ============================================================================
// User Buffer Registration (UBR) Tests
// ============================================================================

class UBR_AllReduce : public RegistrationTestBase {};

TEST_F(UBR_AllReduce, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled())
        << "NCCL_LOCAL_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr) << "Failed to allocate send buffer";
    ASSERT_NE(recvInfo.buffer, nullptr) << "Failed to allocate recv buffer";
    ASSERT_NE(sendInfo.handle, nullptr) << "Failed to register send buffer";
    ASSERT_NE(recvInfo.handle, nullptr) << "Failed to register recv buffer";

    initSendBuffer<T>(sendInfo.buffer, count, rank);

    ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                         getNcclDataType<T>(), ncclSum,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result) << "ncclAllReduce failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream())) << "Stream sync failed";

    ASSERT_TRUE(verifyAllReduceResult<T>(recvInfo.buffer, count, nRanks))
        << "AllReduce result verification failed";
}

class UBR_AllGather : public RegistrationTestBase {};

TEST_F(UBR_AllGather, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled())
        << "NCCL_LOCAL_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr) << "Failed to allocate send buffer";
    ASSERT_NE(recvInfo.buffer, nullptr) << "Failed to allocate recv buffer";
    ASSERT_NE(sendInfo.handle, nullptr) << "Failed to register send buffer";
    ASSERT_NE(recvInfo.handle, nullptr) << "Failed to register recv buffer";

    initSendBuffer<T>(sendInfo.buffer, countPerRank, rank);

    ncclResult_t result = ncclAllGather(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                         getNcclDataType<T>(),
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result) << "ncclAllGather failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream())) << "Stream sync failed";

    ASSERT_TRUE(verifyAllGatherResult<T>(recvInfo.buffer, countPerRank, nRanks))
        << "AllGather result verification failed";
}

class UBR_ReduceScatter : public RegistrationTestBase {};

TEST_F(UBR_ReduceScatter, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled())
        << "NCCL_LOCAL_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr) << "Failed to allocate send buffer";
    ASSERT_NE(recvInfo.buffer, nullptr) << "Failed to allocate recv buffer";
    ASSERT_NE(sendInfo.handle, nullptr) << "Failed to register send buffer";
    ASSERT_NE(recvInfo.handle, nullptr) << "Failed to register recv buffer";

    initSendBuffer<T>(sendInfo.buffer, countPerRank * nRanks, rank);

    ncclResult_t result = ncclReduceScatter(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result) << "ncclReduceScatter failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream())) << "Stream sync failed";

    ASSERT_TRUE(verifyReduceScatterResult<T>(recvInfo.buffer, countPerRank, nRanks))
        << "ReduceScatter result verification failed";
}

class UBR_Broadcast : public RegistrationTestBase {};

TEST_F(UBR_Broadcast, NonZeroRoot_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes with 4+ ranks";
    }

    ASSERT_TRUE(isUBREnabled())
        << "NCCL_LOCAL_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const int root = nRanks - 1;

    RegInfo bufInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(bufInfo);
    });

    ASSERT_NE(bufInfo.buffer, nullptr) << "Failed to allocate buffer";
    ASSERT_NE(bufInfo.handle, nullptr) << "Failed to register buffer";

    const T rootValue = static_cast<T>(99.0f);
    if (rank == root) {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [rootValue](size_t) { return rootValue; });
    } else {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [](size_t) { return static_cast<T>(0.0f); });
    }

    ncclResult_t result = ncclBroadcast(bufInfo.buffer, bufInfo.buffer, count,
                                         getNcclDataType<T>(), root,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result) << "ncclBroadcast failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream())) << "Stream sync failed";

    ASSERT_TRUE(verifyBroadcastResult<T>(bufInfo.buffer, count, rootValue))
        << "Broadcast result verification failed";
}

class UBR_AllToAll : public RegistrationTestBase {};

TEST_F(UBR_AllToAll, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes with 4+ ranks";
    }

    ASSERT_TRUE(isUBREnabled())
        << "NCCL_LOCAL_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t totalCount = countPerRank * nRanks;

    RegInfo sendInfo = allocateAndRegister(totalCount * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(totalCount * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr) << "Failed to allocate send buffer";
    ASSERT_NE(recvInfo.buffer, nullptr) << "Failed to allocate recv buffer";
    ASSERT_NE(sendInfo.handle, nullptr) << "Failed to register send buffer";
    ASSERT_NE(recvInfo.handle, nullptr) << "Failed to register recv buffer";

    initializeBufferWithPattern<T>(sendInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        });

    ncclResult_t result = ncclAllToAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                        getNcclDataType<T>(),
                                        getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result) << "ncclAllToAll failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream())) << "Stream sync failed";

    bool verified = verifyBufferData<T>(recvInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int srcRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
        });
    ASSERT_TRUE(verified) << "AllToAll result verification failed";
}

class UBR_SendRecv : public RegistrationTestBase {};

TEST_F(UBR_SendRecv, RingPattern_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes for SendRecv UBR";
    }

    ASSERT_TRUE(isUBREnabled())
        << "NCCL_LOCAL_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr) << "Failed to allocate send buffer";
    ASSERT_NE(recvInfo.buffer, nullptr) << "Failed to allocate recv buffer";
    ASSERT_NE(sendInfo.handle, nullptr) << "Failed to register send buffer";
    ASSERT_NE(recvInfo.handle, nullptr) << "Failed to register recv buffer";

    int sendPeer = (rank + 1) % nRanks;
    int recvPeer = (rank - 1 + nRanks) % nRanks;

    initSendBuffer<T>(sendInfo.buffer, count, rank);

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ncclResult_t sendResult = ncclSend(sendInfo.buffer, count, getNcclDataType<T>(),
                                        sendPeer, getActiveCommunicator(), getActiveStream());
    ncclResult_t recvResult = ncclRecv(recvInfo.buffer, count, getNcclDataType<T>(),
                                        recvPeer, getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, sendResult) << "ncclSend failed";
    ASSERT_EQ(ncclSuccess, recvResult) << "ncclRecv failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream())) << "Stream sync failed";

    T expected = static_cast<T>(static_cast<float>(recvPeer + 1));
    bool verified = verifyBufferData<T>(recvInfo.buffer, count,
        [expected](size_t) { return expected; });
    ASSERT_TRUE(verified) << "SendRecv result verification failed";
}

// ============================================================================
// Graph Capture Registration Tests
// ============================================================================

class GraphCapture_AllToAll : public RegistrationTestBase {};

TEST_F(GraphCapture_AllToAll, MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isPerRankLoggingEnabled())
        << "RCCL_MPI_LOG_ALL_RANKS must be set to 1 for log verification";

    ASSERT_TRUE(isGraphRegisterEnabled())
        << "NCCL_GRAPH_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t totalCount = countPerRank * nRanks;
    const size_t bufSize = totalCount * sizeof(T);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bufSize)) << "Failed to allocate send buffer";
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bufSize)) << "Failed to allocate recv buffer";

    auto bufCleanup = makeScopeGuard([&]() {
        if (sendBuf) hipFree(sendBuf);
        if (recvBuf) hipFree(recvBuf);
    });

    initializeBufferWithPattern<T>(sendBuf, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        });

    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;

    // Graph capture
    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal))
        << "hipStreamBeginCapture failed";

    ncclResult_t ncclErr = ncclAllToAll(sendBuf, recvBuf, countPerRank,
                                         getNcclDataType<T>(),
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, ncclErr) << "ncclAllToAll failed during graph capture";

    ASSERT_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph))
        << "hipStreamEndCapture failed";
    ASSERT_NE(nullptr, graph) << "Graph capture returned null graph";

    size_t numNodes = 0;
    hipGraphGetNodes(graph, nullptr, &numNodes);
    ASSERT_GT(numNodes, 0u) << "Graph has no nodes";
    TEST_INFO("Graph captured with %zu nodes", numNodes);

    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))
        << "hipGraphInstantiate failed";

    auto graphCleanup = makeScopeGuard([&]() {
        if (graphExec) hipGraphExecDestroy(graphExec);
        if (graph) hipGraphDestroy(graph);
    });

    // Graph execution
    ASSERT_EQ(hipSuccess, hipMemset(recvBuf, 0, bufSize));
    ASSERT_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()))
        << "hipGraphLaunch failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()))
        << "hipStreamSynchronize failed";

    // Verify registration
    REGLogChecker checker = getLogChecker();
    bool registrationDetected = checker.hasAnyRegistrationSuccess();
    TEST_INFO("AllToAll_MultiNode: %s (log size: %zu bytes)",
              checker.getSummary().c_str(), checker.getContentLength());

    ASSERT_TRUE(registrationDetected)
        << "Graph registration was not detected in logs. " << checker.getSummary();

    // Verify results
    bool resultValid = verifyBufferData<T>(recvBuf, totalCount,
        [rank, countPerRank](size_t i) {
            int srcRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
        });
    ASSERT_MPI_TRUE(resultValid);
    TEST_INFO("AllToAll graph test completed successfully");
}

class GraphCapture_AllReduce : public RegistrationTestBase {};

TEST_F(GraphCapture_AllReduce, MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isPerRankLoggingEnabled())
        << "RCCL_MPI_LOG_ALL_RANKS must be set to 1 for log verification";

    ASSERT_TRUE(isGraphRegisterEnabled())
        << "NCCL_GRAPH_REGISTER must be set to 1 for this test";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::LARGE_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t bufSize = count * sizeof(T);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bufSize)) << "Failed to allocate send buffer";
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bufSize)) << "Failed to allocate recv buffer";

    auto bufCleanup = makeScopeGuard([&]() {
        if (sendBuf) hipFree(sendBuf);
        if (recvBuf) hipFree(recvBuf);
    });

    initSendBuffer<T>(sendBuf, count, rank);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;

    // Graph capture
    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal))
        << "hipStreamBeginCapture failed";

    ncclResult_t ncclErr = ncclAllReduce(sendBuf, recvBuf, count,
                                          getNcclDataType<T>(), ncclSum,
                                          getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, ncclErr) << "ncclAllReduce failed during graph capture";

    ASSERT_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph))
        << "hipStreamEndCapture failed";
    ASSERT_NE(nullptr, graph) << "Graph capture returned null graph";

    size_t numNodes = 0;
    hipGraphGetNodes(graph, nullptr, &numNodes);
    ASSERT_GT(numNodes, 0u) << "Graph has no nodes";
    TEST_INFO("AllReduce graph captured with %zu nodes", numNodes);

    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))
        << "hipGraphInstantiate failed";

    auto graphCleanup = makeScopeGuard([&]() {
        if (graphExec) hipGraphExecDestroy(graphExec);
        if (graph) hipGraphDestroy(graph);
    });

    // Graph execution
    ASSERT_EQ(hipSuccess, hipMemset(recvBuf, 0, bufSize));
    ASSERT_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()))
        << "hipGraphLaunch failed";
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()))
        << "hipStreamSynchronize failed";

    // Verify registration
    REGLogChecker checker = getLogChecker();
    bool registrationDetected = checker.hasAnyRegistrationSuccess();
    TEST_INFO("AllReduce_MultiNode: %s (log size: %zu bytes)",
              checker.getSummary().c_str(), checker.getContentLength());

    ASSERT_TRUE(registrationDetected)
        << "Graph registration was not detected in logs. " << checker.getSummary();

    // Verify results
    bool resultValid = verifyAllReduceResult<T>(recvBuf, count, nRanks);
    ASSERT_MPI_TRUE(resultValid);
    TEST_INFO("AllReduce graph test completed successfully");
}

#endif // MPI_TESTS_ENABLED
