// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <deque>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "common/assert.h"
#include "common/common_types.h"
#include "video_core/query_cache/bank_base.h"
#include "video_core/query_cache/query_base.h"

namespace VideoCommon {

/// EDEN_NX2_QUERY_CUTS=0 turns off the cheaper report bookkeeping (zortos293/Lichii#1515): free
/// ids taken by the batch, queries freed a streamer at a time, and small report lookups that probe
/// the offsets a query can start at (or skip pages no report was ever on).
[[nodiscard]] inline bool QueryCutsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_QUERY_CUTS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

class StreamerInterface {
public:
    explicit StreamerInterface(size_t id_) : id{id_}, dependence_mask{}, dependent_mask{} {}
    virtual ~StreamerInterface() = default;

    virtual QueryBase* GetQuery(size_t id) = 0;

    virtual void StartCounter() {
        /* Do Nothing */
    }

    virtual void PauseCounter() {
        /* Do Nothing */
    }

    virtual void ResetCounter() {
        /* Do Nothing */
    }

    virtual void CloseCounter() {
        /* Do Nothing */
    }

    virtual bool HasPendingSync() const {
        return false;
    }

    virtual void PresyncWrites() {
        /* Do Nothing */
    }

    virtual void SyncWrites() {
        /* Do Nothing */
    }

    // EDEN_NX2_QUERY_LAZY_SYNC (payload releases, GuestStreamer): the newest pending sync's
    // value reaches guest memory during the report (MarkInFlight .. FinishInFlight); a wait for
    // idle applies written ones as guest CPU writes (ResolveLazy) and copies only the rest.
    virtual void MarkInFlight([[maybe_unused]] size_t query_id) {}

    /// Returns true when the sync must be copied at once (a wait for idle left it pending).
    virtual bool FinishInFlight([[maybe_unused]] size_t query_id, [[maybe_unused]] bool written) {
        return false;
    }

    virtual void ResolveLazy() {}

    virtual bool HasCopyPending() const {
        return HasPendingSync();
    }

    virtual size_t WriteCounter(VAddr address, bool has_timestamp, u32 value,
                                std::optional<u32> subreport = std::nullopt) = 0;

    virtual bool HasUnsyncedQueries() const {
        return false;
    }

    virtual void PushUnsyncedQueries() {
        /* Do Nothing */
    }

    virtual void PopUnsyncedQueries() {
        /* Do Nothing */
    }

    virtual void Free(size_t query_id) = 0;

    /// Free for each id, as one batch.
    virtual void FreeMany(std::span<const size_t> query_ids) {
        for (const size_t query_id : query_ids) {
            Free(query_id);
        }
    }

    size_t GetId() const {
        return id;
    }

    u64 GetDependenceMask() const {
        return dependence_mask;
    }

    u64 GetDependentMask() const {
        return dependent_mask;
    }

    u64 GetAmendValue() const {
        return amend_value;
    }

    void SetAccumulationValue(u64 new_value) {
        accumulation_value = new_value;
    }

protected:
    void MakeDependent(StreamerInterface* depend_on) {
        dependence_mask |= 1ULL << depend_on->id;
        depend_on->dependent_mask |= 1ULL << id;
    }

    const size_t id;
    u64 dependence_mask;
    u64 dependent_mask;
    u64 amend_value{};
    u64 accumulation_value{};
};

template <typename QueryType>
class SimpleStreamer : public StreamerInterface {
public:
    explicit SimpleStreamer(size_t id_) : StreamerInterface{id_} {}
    virtual ~SimpleStreamer() = default;

protected:
    virtual QueryType* GetQuery(size_t query_id) override {
        if (query_id < slot_queries.size()) {
            return &slot_queries[query_id];
        }
        return nullptr;
    }

    virtual void Free(size_t query_id) override {
        std::scoped_lock lk(guard);
        ReleaseQuery(query_id);
    }

    virtual void FreeMany(std::span<const size_t> query_ids) override {
        std::scoped_lock lk(guard);
        for (const size_t query_id : query_ids) {
            ReleaseQuery(query_id);
        }
    }

    template <typename... Args, typename = decltype(QueryType(std::declval<Args>()...))>
    size_t BuildQuery(Args&&... args) {
        if (QueryCutsEnabled()) {
            // Called by the one thread that records reports (the GPU thread, or the split's back
            // while it runs). Free ids come from `reusable`, which only this thread touches,
            // moved there from old_queries a batch at a time: one lock per batch instead of one
            // per query (a report per draw builds two).
            if (reusable_head == reusable.size()) {
                reusable.clear();
                reusable_head = 0;
                std::scoped_lock lk(guard);
                if (old_queries_head == old_queries.size()) {
                    const size_t new_id = slot_queries.size();
                    slot_queries.emplace_back(std::forward<Args>(args)...);
                    return new_id;
                }
                // Oldest first, as they were freed.
                if (old_queries_head != 0) {
                    old_queries.erase(old_queries.begin(),
                                      old_queries.begin() +
                                          static_cast<std::ptrdiff_t>(old_queries_head));
                    old_queries_head = 0;
                }
                reusable.swap(old_queries);
            }
            const size_t new_id = reusable[reusable_head++];
            new (&slot_queries[new_id]) QueryType(std::forward<Args>(args)...);
            return new_id;
        }
        std::scoped_lock lk(guard);
        if (old_queries_head != old_queries.size()) {
            size_t new_id = old_queries[old_queries_head++];
            if (old_queries_head == old_queries.size()) {
                old_queries.clear();
                old_queries_head = 0;
            }
            new (&slot_queries[new_id]) QueryType(std::forward<Args>(args)...);
            return new_id;
        }
        size_t new_id = slot_queries.size();
        slot_queries.emplace_back(std::forward<Args>(args)...);
        return new_id;
    }

    void ReleaseQuery(size_t query_id) {

        if (query_id < slot_queries.size()) {
            if (old_queries_head >= 1024 && old_queries_head * 2 >= old_queries.size()) {
                old_queries.erase(old_queries.begin(),
                                  old_queries.begin() + static_cast<std::ptrdiff_t>(old_queries_head));
                old_queries_head = 0;
            }
            old_queries.push_back(query_id);
            return;
        }
        UNREACHABLE();
    }

    std::mutex guard;
    std::deque<QueryType> slot_queries;
    std::vector<size_t> old_queries;
    size_t old_queries_head{};
    /// EDEN_NX2_QUERY_CUTS: free ids BuildQuery took from old_queries, used from reusable_head
    /// (its thread only).
    std::vector<size_t> reusable;
    size_t reusable_head{};
};

} // namespace VideoCommon
