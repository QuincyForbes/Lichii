// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <type_traits>
#include <vector>

#include "common/common_types.h"

namespace Common {

template <class Traits>
class LeastRecentlyUsedCache {
    using ObjectType = typename Traits::ObjectType;
    using TickType = typename Traits::TickType;

    static constexpr u32 NO_ITEM = std::numeric_limits<u32>::max();

    struct Item {
        ObjectType obj;
        TickType tick;
        u32 next{NO_ITEM};
        u32 prev{NO_ITEM};
    };

public:
    LeastRecentlyUsedCache() = default;
    ~LeastRecentlyUsedCache() = default;

    size_t Insert(ObjectType obj, TickType tick) {
        const u32 new_id = Build();
        auto& item = item_pool[new_id];
        item.obj = obj;
        item.tick = tick;
        Attach(new_id, item);
        return new_id;
    }

    void Touch(size_t id, TickType tick) {
        auto& item = item_pool[id];
        if (item.tick >= tick) {
            return;
        }
        item.tick = tick;
        const u32 item_id = static_cast<u32>(id);
        if (item_id == last_item) {
            return;
        }
        Detach(item_id, item);
        Attach(item_id, item);
    }

    void Free(size_t id) {
        const u32 item_id = static_cast<u32>(id);
        auto& item = item_pool[item_id];
        Detach(item_id, item);
        item.prev = NO_ITEM;
        item.next = NO_ITEM;
        free_items.push_back(item_id);
    }

    template <typename Func>
    void ForEachItemBelow(TickType tick, Func&& func) {
        static constexpr bool RETURNS_BOOL =
            std::is_same_v<std::invoke_result_t<Func, ObjectType>, bool>;
        u32 iterator = first_item;
        while (iterator != NO_ITEM) {
            Item& item = item_pool[iterator];
            if (static_cast<s64>(tick) - static_cast<s64>(item.tick) < 0) {
                return;
            }
            const u32 next = item.next;
            if constexpr (RETURNS_BOOL) {
                if (func(item.obj)) {
                    return;
                }
            } else {
                func(item.obj);
            }
            iterator = next;
        }
    }

private:
    u32 Build() {
        if (free_items.empty()) {
            const u32 item_id = static_cast<u32>(item_pool.size());
            auto& item = item_pool.emplace_back();
            item.next = NO_ITEM;
            item.prev = NO_ITEM;
            return item_id;
        }
        const u32 item_id = free_items.back();
        free_items.pop_back();
        auto& item = item_pool[item_id];
        item.next = NO_ITEM;
        item.prev = NO_ITEM;
        return item_id;
    }

    void Attach(u32 item_id, Item& item) {
        if (first_item == NO_ITEM) {
            first_item = item_id;
        }
        if (last_item == NO_ITEM) {
            last_item = item_id;
        } else {
            item.prev = last_item;
            item_pool[last_item].next = item_id;
            item.next = NO_ITEM;
            last_item = item_id;
        }
    }

    void Detach(u32 item_id, Item& item) {
        if (item.prev != NO_ITEM) {
            item_pool[item.prev].next = item.next;
        }
        if (item.next != NO_ITEM) {
            item_pool[item.next].prev = item.prev;
        }
        if (item_id == first_item) {
            first_item = item.next;
            if (first_item != NO_ITEM) {
                item_pool[first_item].prev = NO_ITEM;
            }
        }
        if (item_id == last_item) {
            last_item = item.prev;
            if (last_item != NO_ITEM) {
                item_pool[last_item].next = NO_ITEM;
            }
        }
    }

    std::vector<Item> item_pool;
    std::vector<u32> free_items;
    u32 first_item{NO_ITEM};
    u32 last_item{NO_ITEM};
};

} // namespace Common
