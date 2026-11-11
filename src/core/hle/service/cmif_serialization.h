// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/div_ceil.h"

#include "core/hle/service/cmif_types.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/service.h"
#include <typeinfo>

namespace Service {

// clang-format off
template <typename T>
struct UnwrapArg {
    using Type = std::remove_cvref_t<T>;
};

template <typename T, int A>
struct UnwrapArg<InLargeData<T, A>> {
    using Type = std::remove_cv_t<typename InLargeData<T, A>::Type>;
};

template <typename T>
struct UnwrapArg<Out<T>> {
    using Type = AutoOut<typename Out<T>::Type>;
};

template <typename T>
struct UnwrapArg<OutCopyHandle<T>> {
    using Type = AutoOut<typename OutCopyHandle<T>::Type>;
};

template <typename T>
struct UnwrapArg<OutMoveHandle<T>> {
    using Type = AutoOut<typename OutMoveHandle<T>::Type>;
};

template <typename T, int A>
struct UnwrapArg<OutLargeData<T, A>> {
    using Type = AutoOut<typename OutLargeData<T, A>::Type>;
};

enum class ArgumentType {
    InProcessId,
    InData,
    InInterface,
    InCopyHandle,
    OutData,
    OutInterface,
    OutCopyHandle,
    OutMoveHandle,
    InBuffer,
    InLargeData,
    OutBuffer,
    OutLargeData,
};

template <typename T>
struct ArgumentTraits;

template <>
struct ArgumentTraits<ClientProcessId> {
    static constexpr ArgumentType Type = ArgumentType::InProcessId;
};

template <typename T>
struct ArgumentTraits<SharedPointer<T>> {
    static constexpr ArgumentType Type = ArgumentType::InInterface;
};

template <typename T>
struct ArgumentTraits<InCopyHandle<T>> {
    static constexpr ArgumentType Type = ArgumentType::InCopyHandle;
};

template <typename T>
struct ArgumentTraits<Out<SharedPointer<T>>> {
    static constexpr ArgumentType Type = ArgumentType::OutInterface;
};

template <typename T>
struct ArgumentTraits<Out<T>> {
    static constexpr ArgumentType Type = ArgumentType::OutData;
};

template <typename T>
struct ArgumentTraits<OutCopyHandle<T>> {
    static constexpr ArgumentType Type = ArgumentType::OutCopyHandle;
};

template <typename T>
struct ArgumentTraits<OutMoveHandle<T>> {
    static constexpr ArgumentType Type = ArgumentType::OutMoveHandle;
};

template <typename T, int A>
struct ArgumentTraits<Buffer<T, A>> {
    static constexpr ArgumentType Type = (A & BufferAttr_In) == 0 ? ArgumentType::OutBuffer : ArgumentType::InBuffer;
};

template <typename T, int A>
struct ArgumentTraits<InLargeData<T, A>> {
    static constexpr ArgumentType Type = ArgumentType::InLargeData;
};

template <typename T, int A>
struct ArgumentTraits<OutLargeData<T, A>> {
    static constexpr ArgumentType Type = ArgumentType::OutLargeData;
};

template <typename T>
struct ArgumentTraits {
    static constexpr ArgumentType Type = ArgumentType::InData;
};

template <typename... Ts>
consteval bool ConstIfReference() {
    return ((!std::is_reference_v<Ts> || std::is_const_v<std::remove_reference_t<Ts>>) && ... && true);
}

struct RequestLayout {
    u32 copy_handle_count;
    u32 move_handle_count;
    u32 cmif_raw_data_size;
    u32 domain_interface_count;
};

template <typename MethodArguments, size_t PrevAlign = 1, size_t DataOffset = 0, size_t ArgIndex = 0>
constexpr u32 GetInRawDataSize() {
    if constexpr (ArgIndex >= std::tuple_size_v<MethodArguments>) {
        return static_cast<u32>(DataOffset);
    } else {
        using ArgType = std::tuple_element_t<ArgIndex, MethodArguments>;

        if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::InData || ArgumentTraits<ArgType>::Type == ArgumentType::InProcessId) {
            constexpr size_t ArgAlign = alignof(ArgType);
            constexpr size_t ArgSize = sizeof(ArgType);

            static_assert(PrevAlign <= ArgAlign, "Input argument is not ordered by alignment");

            constexpr size_t ArgOffset = Common::AlignUp(DataOffset, ArgAlign);
            constexpr size_t ArgEnd = ArgOffset + ArgSize;

            return GetInRawDataSize<MethodArguments, ArgAlign, ArgEnd, ArgIndex + 1>();
        } else {
            return GetInRawDataSize<MethodArguments, PrevAlign, DataOffset, ArgIndex + 1>();
        }
    }
}

template <typename MethodArguments, size_t PrevAlign = 1, size_t DataOffset = 0, size_t ArgIndex = 0>
constexpr u32 GetOutRawDataSize() {
    if constexpr (ArgIndex >= std::tuple_size_v<MethodArguments>) {
        return static_cast<u32>(DataOffset);
    } else {
        using ArgType = std::tuple_element_t<ArgIndex, MethodArguments>;

        if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutData) {
            using RawArgType = typename ArgType::Type;
            constexpr size_t ArgAlign = alignof(RawArgType);
            constexpr size_t ArgSize = sizeof(RawArgType);

            static_assert(PrevAlign <= ArgAlign, "Output argument is not ordered by alignment");

            constexpr size_t ArgOffset = Common::AlignUp(DataOffset, ArgAlign);
            constexpr size_t ArgEnd = ArgOffset + ArgSize;

            return GetOutRawDataSize<MethodArguments, ArgAlign, ArgEnd, ArgIndex + 1>();
        } else {
            return GetOutRawDataSize<MethodArguments, PrevAlign, DataOffset, ArgIndex + 1>();
        }
    }
}

template <ArgumentType DataType, typename MethodArguments, size_t ArgCount = 0, size_t ArgIndex = 0>
constexpr u32 GetArgumentTypeCount() {
    if constexpr (ArgIndex >= std::tuple_size_v<MethodArguments>) {
        return static_cast<u32>(ArgCount);
    } else {
        using ArgType = std::tuple_element_t<ArgIndex, MethodArguments>;

        if constexpr (ArgumentTraits<ArgType>::Type == DataType) {
            return GetArgumentTypeCount<DataType, MethodArguments, ArgCount + 1, ArgIndex + 1>();
        } else {
            return GetArgumentTypeCount<DataType, MethodArguments, ArgCount, ArgIndex + 1>();
        }
    }
}

template <typename MethodArguments>
constexpr RequestLayout GetNonDomainReplyInLayout() {
    return RequestLayout{
        .copy_handle_count = GetArgumentTypeCount<ArgumentType::InCopyHandle, MethodArguments>(),
        .move_handle_count = 0,
        .cmif_raw_data_size = GetInRawDataSize<MethodArguments>(),
        .domain_interface_count = 0,
    };
}

template <typename MethodArguments>
constexpr RequestLayout GetDomainReplyInLayout() {
    return RequestLayout{
        .copy_handle_count = GetArgumentTypeCount<ArgumentType::InCopyHandle, MethodArguments>(),
        .move_handle_count = 0,
        .cmif_raw_data_size = GetInRawDataSize<MethodArguments>(),
        .domain_interface_count = GetArgumentTypeCount<ArgumentType::InInterface, MethodArguments>(),
    };
}

template <typename MethodArguments>
constexpr RequestLayout GetNonDomainReplyOutLayout() {
    return RequestLayout{
        .copy_handle_count = GetArgumentTypeCount<ArgumentType::OutCopyHandle, MethodArguments>(),
        .move_handle_count = GetArgumentTypeCount<ArgumentType::OutMoveHandle, MethodArguments>() + GetArgumentTypeCount<ArgumentType::OutInterface, MethodArguments>(),
        .cmif_raw_data_size = GetOutRawDataSize<MethodArguments>(),
        .domain_interface_count = 0,
    };
}

template <typename MethodArguments>
constexpr RequestLayout GetDomainReplyOutLayout() {
    return RequestLayout{
        .copy_handle_count = GetArgumentTypeCount<ArgumentType::OutCopyHandle, MethodArguments>(),
        .move_handle_count = GetArgumentTypeCount<ArgumentType::OutMoveHandle, MethodArguments>(),
        .cmif_raw_data_size = GetOutRawDataSize<MethodArguments>(),
        .domain_interface_count = GetArgumentTypeCount<ArgumentType::OutInterface, MethodArguments>(),
    };
}

template <typename MethodArguments>
constexpr RequestLayout GetReplyInLayout(bool is_domain) {
    return is_domain ? GetDomainReplyInLayout<MethodArguments>() : GetNonDomainReplyInLayout<MethodArguments>();
}

template <typename MethodArguments>
constexpr RequestLayout GetReplyOutLayout(bool is_domain) {
    return is_domain ? GetDomainReplyOutLayout<MethodArguments>() : GetNonDomainReplyOutLayout<MethodArguments>();
}

struct OutTemporaryBuffers {
    static constexpr std::size_t MaxBuffers = 8;
    std::array<Common::ScratchBuffer<u8>, MaxBuffers> scratch{};
    std::array<bool, MaxBuffers> direct{};

    Common::ScratchBuffer<u8>& operator[](std::size_t index) noexcept {
        return scratch[index];
    }
    const Common::ScratchBuffer<u8>& operator[](std::size_t index) const noexcept {
        return scratch[index];
    }
};

struct ScopedOutScratch {
    static constexpr std::size_t TrimThresholdBytes = 1024 * 1024;
    static constexpr u32 TrimAfterIdleCalls = 64;

    struct ThreadState {
        OutTemporaryBuffers buffers{};
        u32 idle_calls{};
        bool in_use{};
    };

    ScopedOutScratch() noexcept {
        static thread_local ThreadState t_state{};
        if (!t_state.in_use) [[likely]] {
            t_state.in_use = true;
            t_state.buffers.direct.fill(false);
            for (auto& buf : t_state.buffers.scratch) {
                buf.resize_destructive(0);
            }
            state = &t_state;
            active = &t_state.buffers;
        } else {
            fallback.emplace();
            active = &*fallback;
        }
    }

    ~ScopedOutScratch() {
        if (state != nullptr) [[likely]] {
            bool any_large_capacity = false;
            bool used_large_this_call = false;
            for (const auto& buf : state->buffers.scratch) {
                if (buf.capacity() > TrimThresholdBytes) {
                    any_large_capacity = true;
                }
                if (buf.size() > TrimThresholdBytes) {
                    used_large_this_call = true;
                }
            }
            if (used_large_this_call) {
                state->idle_calls = 0;
            } else if (any_large_capacity && ++state->idle_calls >= TrimAfterIdleCalls) {
                for (auto& buf : state->buffers.scratch) {
                    if (buf.capacity() > TrimThresholdBytes) {
                        buf.release();
                    }
                }
                state->idle_calls = 0;
            }
            state->in_use = false;
        }
    }

    ThreadState* state{};
    OutTemporaryBuffers* active{};
    std::optional<OutTemporaryBuffers> fallback{};
};

template <typename MethodArguments, typename CallArguments, size_t PrevAlign = 1, size_t DataOffset = 0, size_t HandleIndex = 0, size_t InBufferIndex = 0, size_t OutBufferIndex = 0, bool RawDataFinished = false, size_t ArgIndex = 0>
void ReadInArgument(bool is_domain, CallArguments& args, const u8* raw_data, HLERequestContext& ctx, OutTemporaryBuffers& temp) {
    if constexpr (ArgIndex >= std::tuple_size_v<CallArguments>) {
        return;
    } else {
        using ArgType = std::tuple_element_t<ArgIndex, MethodArguments>;

        if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::InData || ArgumentTraits<ArgType>::Type == ArgumentType::InProcessId) {
            constexpr size_t ArgAlign = alignof(ArgType);
            constexpr size_t ArgSize = sizeof(ArgType);

            static_assert(PrevAlign <= ArgAlign, "Input argument is not ordered by alignment");
            static_assert(!RawDataFinished, "All input interface arguments must appear after raw data");
            static_assert(!std::is_pointer_v<ArgType>, "Input raw data must not be a pointer");
            static_assert(std::is_trivially_copyable_v<ArgType>, "Input raw data must be trivially copyable");

            constexpr size_t ArgOffset = Common::AlignUp(DataOffset, ArgAlign);
            constexpr size_t ArgEnd = ArgOffset + ArgSize;

            if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::InProcessId) {
                // TODO: abort parsing if PID is not provided?
                // TODO: validate against raw data value?
                std::get<ArgIndex>(args).pid = ctx.GetPID();
            } else {
                std::memcpy(&std::get<ArgIndex>(args), raw_data + ArgOffset, ArgSize);
            }

            return ReadInArgument<MethodArguments, CallArguments, ArgAlign, ArgEnd, HandleIndex, InBufferIndex, OutBufferIndex, false, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::InInterface) {
            constexpr size_t ArgAlign = alignof(u32);
            constexpr size_t ArgSize = sizeof(u32);
            constexpr size_t ArgOffset = Common::AlignUp(DataOffset, ArgAlign);
            constexpr size_t ArgEnd = ArgOffset + ArgSize;

            ASSERT(is_domain);
            ASSERT(ctx.GetDomainMessageHeader().input_object_count > 0);

            u32 value{};
            std::memcpy(&value, raw_data + ArgOffset, ArgSize);
            std::get<ArgIndex>(args) = ctx.GetDomainHandler<typename ArgType::element_type>(value - 1);

            return ReadInArgument<MethodArguments, CallArguments, ArgAlign, ArgEnd, HandleIndex, InBufferIndex, OutBufferIndex, true, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::InCopyHandle) {
            std::get<ArgIndex>(args) = ctx.GetObjectFromHandle<typename ArgType::Type>(ctx.GetCopyHandle(HandleIndex)).GetPointerUnsafe();

            return ReadInArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, HandleIndex + 1, InBufferIndex, OutBufferIndex, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::InLargeData) {
            constexpr size_t BufferSize = sizeof(typename ArgType::Type);

            // Clear the existing data.
            std::memset(&std::get<ArgIndex>(args), 0, BufferSize);

            std::span<const u8> buffer{};

            ASSERT(ctx.CanReadBuffer(InBufferIndex));
            if constexpr (ArgType::Attr & BufferAttr_HipcAutoSelect) {
                buffer = ctx.ReadBuffer(InBufferIndex);
            } else if constexpr (ArgType::Attr & BufferAttr_HipcMapAlias) {
                buffer = ctx.ReadBufferA(InBufferIndex);
            } else /* if (ArgType::Attr & BufferAttr_HipcPointer) */ {
                buffer = ctx.ReadBufferX(InBufferIndex);
            }

            std::memcpy(&std::get<ArgIndex>(args), buffer.data(), (std::min)(BufferSize, buffer.size()));

            return ReadInArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, HandleIndex, InBufferIndex + 1, OutBufferIndex, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::InBuffer) {
            using ElementType = typename ArgType::Type;

            std::span<const u8> buffer{};

            if (ctx.CanReadBuffer(InBufferIndex)) {
                if constexpr (ArgType::Attr & BufferAttr_HipcAutoSelect) {
                    buffer = ctx.ReadBuffer(InBufferIndex);
                } else if constexpr (ArgType::Attr & BufferAttr_HipcMapAlias) {
                    buffer = ctx.ReadBufferA(InBufferIndex);
                } else /* if (ArgType::Attr & BufferAttr_HipcPointer) */ {
                    buffer = ctx.ReadBufferX(InBufferIndex);
                }
            }

            ElementType* ptr = (ElementType*) buffer.data();
            size_t size = buffer.size() / sizeof(ElementType);

            std::get<ArgIndex>(args) = std::span(ptr, size);

            return ReadInArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, HandleIndex, InBufferIndex + 1, OutBufferIndex, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutLargeData) {
            constexpr size_t BufferSize = sizeof(typename ArgType::Type);

            // Clear the existing data.
            std::memset(&std::get<ArgIndex>(args).raw, 0, BufferSize);

            return ReadInArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, HandleIndex, InBufferIndex, OutBufferIndex + 1, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutBuffer) {
            using ElementType = typename ArgType::Type;
            static_assert(OutBufferIndex < OutTemporaryBuffers::MaxBuffers);

            std::span<u8> raw_span{};
            temp.direct[OutBufferIndex] = false;
            if (ctx.CanWriteBuffer(OutBufferIndex)) {
                const size_t write_size = ctx.GetWriteBufferSize(OutBufferIndex);
                if (write_size == 0) {
                    temp.direct[OutBufferIndex] = true;
                    temp[OutBufferIndex].resize_destructive(0);
                } else {
                    if constexpr (ArgType::Attr & BufferAttr_HipcAutoSelect) {
                        raw_span = ctx.GetWriteBufferSpan(OutBufferIndex);
                    } else if constexpr (ArgType::Attr & BufferAttr_HipcMapAlias) {
                        raw_span = ctx.GetWriteBufferSpanB(OutBufferIndex);
                    } else /* if (ArgType::Attr & BufferAttr_HipcPointer) */ {
                        raw_span = ctx.GetWriteBufferSpanC(OutBufferIndex);
                    }
                    if (raw_span.data() != nullptr &&
                        (reinterpret_cast<uintptr_t>(raw_span.data()) % alignof(ElementType)) == 0) {
                        temp.direct[OutBufferIndex] = true;
                        temp[OutBufferIndex].resize_destructive(0);
                    } else {
                        temp[OutBufferIndex].resize_destructive(write_size);
                        raw_span = std::span<u8>{temp[OutBufferIndex].data(), write_size};
                    }
                }
            } else {
                temp[OutBufferIndex].resize_destructive(0);
            }

            ElementType* ptr = reinterpret_cast<ElementType*>(raw_span.data());
            size_t size = raw_span.size() / sizeof(ElementType);

            std::get<ArgIndex>(args) = std::span(ptr, size);

            return ReadInArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, HandleIndex, InBufferIndex, OutBufferIndex + 1, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else {
            return ReadInArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, HandleIndex, InBufferIndex, OutBufferIndex, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        }
    }
}

template <typename MethodArguments, typename CallArguments, size_t PrevAlign = 1, size_t DataOffset = 0, size_t OutBufferIndex = 0, bool RawDataFinished = false, size_t ArgIndex = 0>
void WriteOutArgument(bool is_domain, CallArguments& args, u8* raw_data, HLERequestContext& ctx, OutTemporaryBuffers& temp) {
    if constexpr (ArgIndex >= std::tuple_size_v<CallArguments>) {
        return;
    } else {
        using ArgType = std::tuple_element_t<ArgIndex, MethodArguments>;

        if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutData) {
            using RawArgType = decltype(std::get<ArgIndex>(args).raw);
            constexpr size_t ArgAlign = alignof(RawArgType);
            constexpr size_t ArgSize = sizeof(RawArgType);

            static_assert(PrevAlign <= ArgAlign, "Output argument is not ordered by alignment");
            static_assert(!RawDataFinished, "All output interface arguments must appear after raw data");
            static_assert(!std::is_pointer_v<ArgType>, "Output raw data must not be a pointer");
            static_assert(!std::is_pointer_v<RawArgType>, "Output raw data must not be a pointer");
            static_assert(std::is_trivially_copyable_v<RawArgType>, "Output raw data must be trivially copyable");

            constexpr size_t ArgOffset = Common::AlignUp(DataOffset, ArgAlign);
            constexpr size_t ArgEnd = ArgOffset + ArgSize;

            std::memcpy(raw_data + ArgOffset, &std::get<ArgIndex>(args).raw, ArgSize);

            return WriteOutArgument<MethodArguments, CallArguments, ArgAlign, ArgEnd, OutBufferIndex, false, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutInterface) {
            if (is_domain) {
                ctx.AddDomainObject(std::get<ArgIndex>(args).raw);
            } else {
                ctx.AddMoveInterface(std::get<ArgIndex>(args).raw);
            }

            return WriteOutArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, OutBufferIndex, true, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutCopyHandle) {
            ctx.AddCopyObject(std::get<ArgIndex>(args).raw);

            return WriteOutArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, OutBufferIndex, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutMoveHandle) {
            ctx.AddMoveObject(std::get<ArgIndex>(args).raw);

            return WriteOutArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, OutBufferIndex, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutLargeData) {
            constexpr size_t BufferSize = sizeof(typename ArgType::Type);

            ASSERT(ctx.CanWriteBuffer(OutBufferIndex));
            if constexpr (ArgType::Attr & BufferAttr_HipcAutoSelect) {
                ctx.WriteBuffer(std::get<ArgIndex>(args), OutBufferIndex);
            } else if constexpr (ArgType::Attr & BufferAttr_HipcMapAlias) {
                ctx.WriteBufferB(&std::get<ArgIndex>(args), BufferSize, OutBufferIndex);
            } else /* if (ArgType::Attr & BufferAttr_HipcPointer) */ {
                ctx.WriteBufferC(&std::get<ArgIndex>(args), BufferSize, OutBufferIndex);
            }

            return WriteOutArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, OutBufferIndex + 1, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else if constexpr (ArgumentTraits<ArgType>::Type == ArgumentType::OutBuffer) {
            static_assert(OutBufferIndex < OutTemporaryBuffers::MaxBuffers);
            if (temp.direct[OutBufferIndex]) {
                if constexpr (ArgType::Attr & BufferAttr_HipcAutoSelect) {
                    ctx.CommitWriteBufferSpan(OutBufferIndex);
                } else if constexpr (ArgType::Attr & BufferAttr_HipcMapAlias) {
                    ctx.CommitWriteBufferSpanB(OutBufferIndex);
                } else /* if (ArgType::Attr & BufferAttr_HipcPointer) */ {
                    ctx.CommitWriteBufferSpanC(OutBufferIndex);
                }
            } else {
                auto& buffer = temp[OutBufferIndex];
                const size_t size = buffer.size();

                if (size > 0 && ctx.CanWriteBuffer(OutBufferIndex)) {
                    if constexpr (ArgType::Attr & BufferAttr_HipcAutoSelect) {
                        ctx.WriteBuffer(buffer.data(), size, OutBufferIndex);
                    } else if constexpr (ArgType::Attr & BufferAttr_HipcMapAlias) {
                        ctx.WriteBufferB(buffer.data(), size, OutBufferIndex);
                    } else /* if (ArgType::Attr & BufferAttr_HipcPointer) */ {
                        ctx.WriteBufferC(buffer.data(), size, OutBufferIndex);
                    }
                }
            }

            return WriteOutArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, OutBufferIndex + 1, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        } else {
            return WriteOutArgument<MethodArguments, CallArguments, PrevAlign, DataOffset, OutBufferIndex, RawDataFinished, ArgIndex + 1>(is_domain, args, raw_data, ctx, temp);
        }
    }
}

template <bool Domain, typename T, typename... A>
void CmifReplyWrapImpl(HLERequestContext& ctx, T& t, Result (T::*f)(A...)) {
    const auto mgr_holder = ctx.GetManager();
    auto* const mgr = mgr_holder.get();
    // Verify domain state.
    if constexpr (!Domain) {
        const bool is_domain = mgr ? mgr->IsDomain() : false;
        ASSERT_MSG(!is_domain,
            "Non-domain reply used on domain session\n"
            "Service={} (TIPC={} CmdType={} Cmd={:#08x}\n"
            "HasDomainHeader={} DomainHandlers={}\nDesc={}",
            t.GetServiceName(), ctx.IsTipc(),
            u32(ctx.GetCommandType()), u32(ctx.GetCommand()),
            ctx.HasDomainMessageHeader(), mgr ? u32(mgr->DomainHandlerCount()) : 0u,
            ctx.Description());
    }
    const bool is_domain = Domain ? mgr->IsDomain() : false;

    static_assert(ConstIfReference<A...>(), "Arguments taken by reference must be const");
    using MethodArguments = std::tuple<std::remove_cvref_t<A>...>;

    ScopedOutScratch scratch_scope{};
    OutTemporaryBuffers& buffers = *scratch_scope.active;
    auto call_arguments = std::tuple<typename UnwrapArg<A>::Type...>();

    // Read inputs.
    const size_t offset_plus_command_id = ctx.GetDataPayloadOffset() + 2;
    ReadInArgument<MethodArguments>(is_domain, call_arguments, reinterpret_cast<u8*>(ctx.CommandBuffer() + offset_plus_command_id), ctx, buffers);

    // Call.
    const auto Callable = [&]<typename... CallArgs>(CallArgs&... args) {
        return (t.*f)(args...);
    };
    const Result res = std::apply(Callable, call_arguments);

    // Write result.
    const RequestLayout layout = GetReplyOutLayout<MethodArguments>(is_domain);
    IPC::ResponseBuilder rb{ctx, 2 + Common::DivCeil(layout.cmif_raw_data_size, sizeof(u32)), layout.copy_handle_count, layout.move_handle_count + layout.domain_interface_count};
    rb.Push(res);

    // Write out arguments.
    WriteOutArgument<MethodArguments>(is_domain, call_arguments, reinterpret_cast<u8*>(ctx.CommandBuffer() + rb.GetCurrentOffset()), ctx, buffers);
}
// clang-format on

template <typename Self>
template <bool Domain, auto F>
inline void ServiceFramework<Self>::CmifReplyWrap(HLERequestContext& ctx) {
    return CmifReplyWrapImpl<Domain>(ctx, *static_cast<Self*>(this), F);
}

} // namespace Service
