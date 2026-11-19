// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
bool HasExternalUses(const IR::Inst& inst) noexcept {
    if (!inst.HasUses()) {
        return false;
    }
    if (inst.GetOpcode() != IR::Opcode::Phi) {
        return true;
    }
    int self_uses{};
    const size_t num_args{inst.NumArgs()};
    for (size_t i = 0; i < num_args; ++i) {
        const IR::Value arg{inst.Arg(i)};
        if (!arg.IsImmediate() && arg.Inst() == &inst) {
            ++self_uses;
        }
    }
    return inst.UseCount() > self_uses;
}
} // Anonymous namespace

void DeadCodeEliminationPass(IR::Program& program) {
    // Iterate over instructions in reverse order within post-order blocks so removing an
    // instruction immediately drops the use count of its producers, repeating only if erasing a
    // loop-header Phi releases back-edge operands in blocks already visited.
    bool phi_erased{};
    do {
        phi_erased = false;
        for (IR::Block* const block : program.post_order_blocks) {
            auto it{block->end()};
            while (it != block->begin()) {
                --it;
                if (!HasExternalUses(*it) && !it->MayHaveSideEffects()) {
                    phi_erased |= it->GetOpcode() == IR::Opcode::Phi;
                    it->Invalidate();
                    it = block->Instructions().erase(it);
                }
            }
        }
    } while (phi_erased);
}

} // namespace Shader::Optimization
