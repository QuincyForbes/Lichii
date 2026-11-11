// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "core/file_sys/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/fsp/fs_i_file.h"
#include "core/hle/service/filesystem/fsp/fs_read_trace.h"

namespace Service::FileSystem {

IFile::IFile(Core::System& system_, FileSys::VirtualFile file_)
    : ServiceFramework{system_, "IFile"}, backend{std::make_unique<FileSys::Fsa::IFile>(file_)} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IFile::Read>, "Read"},
        {1, D<&IFile::Write>, "Write"},
        {2, D<&IFile::Flush>, "Flush"},
        {3, D<&IFile::SetSize>, "SetSize"},
        {4, D<&IFile::GetSize>, "GetSize"},
        {5, nullptr, "OperateRange"},
        {6, nullptr, "OperateRangeWithBuffer"},
    };
    // clang-format on
    RegisterHandlers(functions);
}

Result IFile::Read(
    FileSys::ReadOption option, Out<s64> out_size, s64 offset,
    const OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_buffer,
    s64 size) {
    LOG_DEBUG(Service_FS, "called, option={}, offset={:#x}, length={}", option.value, offset,
              size);

    R_UNLESS(size >= 0, FileSys::ResultOutOfRange);
    const size_t clamped_size = std::min<size_t>(static_cast<size_t>(size), out_buffer.size());

    // Read the data from the Storage backend directly into the output span.
    const ReadTraceScope trace{ReadTraceSource::File, static_cast<s64>(clamped_size)};
    size_t bytes_read = 0;
    R_TRY(backend->Read(&bytes_read, offset, out_buffer.data(), clamped_size, option));
    *out_size = static_cast<s64>(bytes_read);
    R_SUCCEED();
}

Result IFile::Write(
    const InBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> buffer,
    FileSys::WriteOption option, s64 offset, s64 size) {
    LOG_DEBUG(Service_FS, "called, option={}, offset={:#x}, length={}", option.value, offset,
              size);

    R_UNLESS(size >= 0, FileSys::ResultOutOfRange);
    const size_t clamped_size = std::min<size_t>(static_cast<size_t>(size), buffer.size());
    R_RETURN(backend->Write(offset, buffer.data(), clamped_size, option));
}

Result IFile::Flush() {
    LOG_DEBUG(Service_FS, "called");

    R_RETURN(backend->Flush());
}

Result IFile::SetSize(s64 size) {
    LOG_DEBUG(Service_FS, "called, size={}", size);

    R_RETURN(backend->SetSize(size));
}

Result IFile::GetSize(Out<s64> out_size) {
    LOG_DEBUG(Service_FS, "called");

    R_RETURN(backend->GetSize(out_size));
}

} // namespace Service::FileSystem
