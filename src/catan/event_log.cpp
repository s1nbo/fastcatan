// Per-env event buffers for the opt-in native log. Allocation, draining and
// the pending/dropped counters; the append hot path is inline in the header.
#include "../../include/event_log.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

namespace catan {

namespace {

// Round the per-env slab up to a whole cache line so two threads appending to
// neighbouring envs never share one.
constexpr uint32_t STRIDE_ALIGN = 16;  // int32 per 64-byte line

}  // namespace

bool event_log_enable(EventLog& log, uint32_t n, uint32_t mask,
                      uint32_t capacity) noexcept {
    event_log_disable(log);
    if (n == 0 || mask == 0 || capacity == 0) return false;

    const uint64_t raw = uint64_t(capacity) * EVENT_INTS;
    const uint64_t stride = (raw + STRIDE_ALIGN - 1) / STRIDE_ALIGN * STRIDE_ALIGN;
    const uint64_t ints = stride * uint64_t(n);
    if (stride > std::numeric_limits<uint32_t>::max()
        || ints > uint64_t(std::numeric_limits<std::size_t>::max() / sizeof(int32_t))) {
        return false;
    }

    void* rows = std::aligned_alloc(64, std::size_t(ints) * sizeof(int32_t));
    void* cursors = std::aligned_alloc(64, std::size_t(n) * sizeof(EventCursor));
    if (!rows || !cursors) {
        std::free(rows);
        std::free(cursors);
        return false;
    }

    log.mask     = mask;
    log.capacity = capacity;
    log.stride   = uint32_t(stride);
    log.rows     = static_cast<int32_t*>(rows);
    log.cursors  = static_cast<EventCursor*>(cursors);
    std::memset(log.cursors, 0, std::size_t(n) * sizeof(EventCursor));
    return true;
}

void event_log_disable(EventLog& log) noexcept {
    std::free(log.rows);
    std::free(log.cursors);
    log.mask     = 0;
    log.capacity = 0;
    log.stride   = 0;
    log.rows     = nullptr;
    log.cursors  = nullptr;
}

void event_log_reset(EventLog& log, uint32_t n) noexcept {
    if (!log.cursors) return;
    std::memset(log.cursors, 0, std::size_t(n) * sizeof(EventCursor));
}

uint64_t event_log_pending(const EventLog& log, uint32_t n) noexcept {
    if (!log.cursors) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; ++i) total += log.cursors[i].count;
    return total;
}

uint64_t event_log_dropped(const EventLog& log, uint32_t n) noexcept {
    if (!log.cursors) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; ++i) total += log.cursors[i].dropped;
    return total;
}

uint32_t event_log_drain(EventLog& log, uint32_t n, int32_t* out,
                         uint32_t out_rows) noexcept {
    if (!log.cursors || !log.rows) return 0;
    uint32_t written = 0;
    for (uint32_t i = 0; i < n; ++i) {
        EventCursor& cursor = log.cursors[i];
        if (cursor.count == 0) continue;
        if (written == out_rows) break;

        int32_t* slab = log.rows + std::size_t(i) * log.stride;
        const uint32_t room = out_rows - written;
        const uint32_t take = cursor.count < room ? cursor.count : room;
        std::memcpy(out + std::size_t(written) * EVENT_INTS, slab,
                    std::size_t(take) * EVENT_INTS * sizeof(int32_t));
        written += take;

        const uint32_t left = cursor.count - take;
        if (left != 0) {
            // Only the env that ran out of room can be partially drained, so
            // this moves at most one slab per call.
            std::memmove(slab, slab + std::size_t(take) * EVENT_INTS,
                         std::size_t(left) * EVENT_INTS * sizeof(int32_t));
        }
        cursor.count = left;
    }
    return written;
}

}  // namespace catan
