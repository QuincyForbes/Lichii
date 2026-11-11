// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// This file implements the SSA rewriting algorithm proposed in
//
//      Simple and Efficient Construction of Static Single Assignment Form.
//      Braun M., Buchwald S., Hack S., Leiba R., Mallon C., Zwinkau A. (2013)
//      In: Jhala R., De Bosschere K. (eds)
//      Compiler Construction. CC 2013.
//      Lecture Notes in Computer Science, vol 7791.
//      Springer, Berlin, Heidelberg
//
//      https://link.springer.com/chapter/10.1007/978-3-642-37051-9_6
//

#include <deque>
#include <map>
#include <span>
#include "common/container/unordered_map.h"
#include <variant>
#include <vector>

#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/frontend/ir/pred.h"
#include "shader_recompiler/frontend/ir/reg.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
struct FlagTag {
    auto operator<=>(const FlagTag&) const noexcept = default;
};
struct ZeroFlagTag : FlagTag {};
struct SignFlagTag : FlagTag {};
struct CarryFlagTag : FlagTag {};
struct OverflowFlagTag : FlagTag {};

struct GotoVariable : FlagTag {
    GotoVariable() = default;
    explicit GotoVariable(u32 index_) : index{index_} {}

    auto operator<=>(const GotoVariable&) const noexcept = default;

    u32 index;
};

struct IndirectBranchVariable {
    auto operator<=>(const IndirectBranchVariable&) const noexcept = default;
};

using Variant = std::variant<IR::Reg, IR::Pred, ZeroFlagTag, SignFlagTag, CarryFlagTag, OverflowFlagTag, GotoVariable, IndirectBranchVariable>;
struct DefTable {
    static constexpr size_t REG_BANK_SIZE = 256;
    static constexpr size_t NUM_REG_BANKS = (IR::NUM_REGS + REG_BANK_SIZE - 1) / REG_BANK_SIZE;
    using RegBank = std::array<IR::Value, REG_BANK_SIZE>;
    enum : size_t { ZeroFlag, SignFlag, CarryFlag, OverflowFlag, NumFlags };

    /// The definitions in one tracked block, at its SSA id; the goto variables are apart.
    /// (Each predicate and flag had a node map by block: a read walk allocated a node per
    /// block it passed, for the lookup and again for the value it left there.)
    struct BlockDefs {
        std::array<RegBank*, NUM_REG_BANKS> regs{};
        std::array<IR::Value, IR::NUM_USER_PREDS> preds{};
        std::array<IR::Value, NumFlags> flags{};
        IR::Value indirect_branch_var{};
    };

    ~DefTable() {
        for (IR::Block* const block : tracked_blocks) {
            block->SetSsaId(~0U);
        }
    }

    [[nodiscard]] const BlockDefs* Find(const IR::Block* block) const {
        const u32 id = block->SsaId();
        return id != ~0U ? &block_defs[id] : nullptr;
    }
    [[nodiscard]] BlockDefs& Track(IR::Block* block) {
        u32 id = block->SsaId();
        if (id == ~0U) {
            id = static_cast<u32>(tracked_blocks.size());
            block->SetSsaId(id);
            tracked_blocks.push_back(block);
            block_defs.emplace_back();
        }
        return block_defs[id];
    }

    IR::Value Def(IR::Block* block, IR::Reg variable) const {
        const BlockDefs* const defs = Find(block);
        if (!defs) {
            return {};
        }
        const size_t reg_index = IR::RegIndex(variable);
        const RegBank* const bank = defs->regs[reg_index / REG_BANK_SIZE];
        return bank ? (*bank)[reg_index % REG_BANK_SIZE] : IR::Value{};
    }
    void SetDef(IR::Block* block, IR::Reg variable, const IR::Value& value) {
        const size_t reg_index = IR::RegIndex(variable);
        RegBank*& bank = Track(block).regs[reg_index / REG_BANK_SIZE];
        if (!bank) {
            bank = &reg_bank_pool.emplace_back();
        }
        (*bank)[reg_index % REG_BANK_SIZE] = value;
    }

    IR::Value Def(IR::Block* block, IR::Pred variable) const {
        const BlockDefs* const defs = Find(block);
        return defs ? defs->preds[IR::PredIndex(variable)] : IR::Value{};
    }
    void SetDef(IR::Block* block, IR::Pred variable, const IR::Value& value) {
        Track(block).preds[IR::PredIndex(variable)] = value;
    }

    IR::Value Def(IR::Block* block, GotoVariable variable) const {
        const u32 id = block->SsaId();
        if (id == ~0U) {
            return {};
        }
        const auto it = goto_vars.find(GotoKey(variable, id));
        return it != goto_vars.end() ? it->second : IR::Value{};
    }
    void SetDef(IR::Block* block, GotoVariable variable, const IR::Value& value) {
        static_cast<void>(Track(block));
        goto_vars.insert_or_assign(GotoKey(variable, block->SsaId()), value);
    }

    IR::Value Def(IR::Block* block, IndirectBranchVariable) const {
        const BlockDefs* const defs = Find(block);
        return defs ? defs->indirect_branch_var : IR::Value{};
    }
    void SetDef(IR::Block* block, IndirectBranchVariable, const IR::Value& value) {
        Track(block).indirect_branch_var = value;
    }

    IR::Value Def(IR::Block* block, ZeroFlagTag) const {
        return Flag(block, ZeroFlag);
    }
    void SetDef(IR::Block* block, ZeroFlagTag, const IR::Value& value) {
        Track(block).flags[ZeroFlag] = value;
    }

    IR::Value Def(IR::Block* block, SignFlagTag) const {
        return Flag(block, SignFlag);
    }
    void SetDef(IR::Block* block, SignFlagTag, const IR::Value& value) {
        Track(block).flags[SignFlag] = value;
    }

    IR::Value Def(IR::Block* block, CarryFlagTag) const {
        return Flag(block, CarryFlag);
    }
    void SetDef(IR::Block* block, CarryFlagTag, const IR::Value& value) {
        Track(block).flags[CarryFlag] = value;
    }

    IR::Value Def(IR::Block* block, OverflowFlagTag) const {
        return Flag(block, OverflowFlag);
    }
    void SetDef(IR::Block* block, OverflowFlagTag, const IR::Value& value) {
        Track(block).flags[OverflowFlag] = value;
    }

    [[nodiscard]] IR::Value Flag(const IR::Block* block, size_t flag) const {
        const BlockDefs* const defs = Find(block);
        return defs ? defs->flags[flag] : IR::Value{};
    }
    [[nodiscard]] static u64 GotoKey(GotoVariable variable, u32 block_id) {
        return u64{variable.index} << 32 | block_id;
    }

    std::vector<IR::Block*> tracked_blocks;
    std::vector<BlockDefs> block_defs;
    std::deque<RegBank> reg_bank_pool;
    /// By goto variable and SSA id of the block.
    Common::unordered_map<u64, IR::Value> goto_vars;
};

IR::Opcode UndefOpcode(IR::Reg) noexcept {
    return IR::Opcode::UndefU32;
}

IR::Opcode UndefOpcode(IR::Pred) noexcept {
    return IR::Opcode::UndefU1;
}

IR::Opcode UndefOpcode(const FlagTag&) noexcept {
    return IR::Opcode::UndefU1;
}

IR::Opcode UndefOpcode(IndirectBranchVariable) noexcept {
    return IR::Opcode::UndefU32;
}

enum class Status {
    Start,
    SetValue,
    PreparePhiArgument,
    PushPhiArgument,
};

struct ReadState {
    ReadState(IR::Block* block_) : block{block_} {}
    ReadState() = default;

    IR::Block* block{};
    IR::Value result{};
    IR::Inst* phi{};
    IR::Block* const* pred_it{};
    IR::Block* const* pred_end{};
    Status pc{Status::Start};
};

class Pass {
public:
    template <typename Type>
    void WriteVariable(Type variable, IR::Block* block, const IR::Value& value) {
        current_def.SetDef(block, variable, value);
    }

    template <typename Type>
    IR::Value ReadVariable(Type variable, IR::Block* root_block) {
        // The pass's one stack: a read does not start another before it returns.
        std::vector<ReadState>& stack{read_stack};
        stack.clear();
        stack.emplace_back(nullptr);
        stack.emplace_back(root_block);
        const auto prepare_phi_operand{[&] {
            if (stack.back().pred_it == stack.back().pred_end) {
                IR::Inst* const phi{stack.back().phi};
                IR::Block* const block{stack.back().block};
                const IR::Value result{TryRemoveTrivialPhi(*phi, block, UndefOpcode(variable))};
                stack.pop_back();
                stack.back().result = result;
                WriteVariable(variable, block, result);
            } else {
                IR::Block* const imm_pred{*stack.back().pred_it};
                stack.back().pc = Status::PushPhiArgument;
                stack.emplace_back(imm_pred);
            }
        }};
        do {
            IR::Block* const block{stack.back().block};
            switch (stack.back().pc) {
            case Status::Start: {
                if (const IR::Value def = current_def.Def(block, variable); !def.IsEmpty()) {
                    stack.back().result = def;
                } else if (!block->IsSsaSealed()) {
                    // Incomplete CFG
                    IR::Inst* phi{&*block->PrependNewInst(block->begin(), IR::Opcode::Phi)};
                    phi->SetFlags(IR::TypeOf(UndefOpcode(variable)));

                    incomplete_phis[block].insert_or_assign(variable, phi);
                    stack.back().result = IR::Value{&*phi};
                } else if (const std::span imm_preds = block->ImmPredecessors();
                           imm_preds.size() == 1) {
                    // Optimize the common case of one predecessor: no phi needed
                    stack.back().pc = Status::SetValue;
                    stack.emplace_back(imm_preds.front());
                    break;
                } else {
                    // Break potential cycles with operandless phi
                    IR::Inst* const phi{&*block->PrependNewInst(block->begin(), IR::Opcode::Phi)};
                    phi->SetFlags(IR::TypeOf(UndefOpcode(variable)));

                    WriteVariable(variable, block, IR::Value{phi});

                    stack.back().phi = phi;
                    stack.back().pred_it = imm_preds.data();
                    stack.back().pred_end = imm_preds.data() + imm_preds.size();
                    prepare_phi_operand();
                    break;
                }
            }
                [[fallthrough]];
            case Status::SetValue: {
                const IR::Value result{stack.back().result};
                WriteVariable(variable, block, result);
                stack.pop_back();
                stack.back().result = result;
                break;
            }
            case Status::PushPhiArgument: {
                IR::Inst* const phi{stack.back().phi};
                phi->AddPhiOperand(*stack.back().pred_it, stack.back().result);
                ++stack.back().pred_it;
            }
                [[fallthrough]];
            case Status::PreparePhiArgument:
                prepare_phi_operand();
                break;
            }
        } while (stack.size() > 1);
        return stack.back().result;
    }

    void SealBlock(IR::Block* block) {
        const auto it{incomplete_phis.find(block)};
        if (it != incomplete_phis.end()) {
            for (auto& pair : it->second) {
                auto& variant{pair.first};
                auto& phi{pair.second};
                std::visit([&](auto& variable) { AddPhiOperands(variable, *phi, block); }, variant);
            }
        }
        block->SsaSeal();
    }

private:
    template <typename Type>
    IR::Value AddPhiOperands(Type variable, IR::Inst& phi, IR::Block* block) {
        for (IR::Block* const imm_pred : block->ImmPredecessors()) {
            phi.AddPhiOperand(imm_pred, ReadVariable(variable, imm_pred));
        }
        return TryRemoveTrivialPhi(phi, block, UndefOpcode(variable));
    }

    IR::Value TryRemoveTrivialPhi(IR::Inst& phi, IR::Block* block, IR::Opcode undef_opcode) {
        IR::Value same;
        const size_t num_args{phi.NumArgs()};
        for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
            const IR::Value& op{phi.Arg(arg_index)};
            if (op.Resolve() == same.Resolve() || op == IR::Value{&phi}) {
                // Unique value or self-reference
                continue;
            }
            if (!same.IsEmpty()) {
                // The phi merges at least two values: not trivial
                return IR::Value{&phi};
            }
            same = op;
        }
        // Remove the phi node from the block, it will be reinserted
        IR::Block::InstructionList& list{block->Instructions()};
        list.erase(IR::Block::InstructionList::s_iterator_to(phi));

        // Find the first non-phi instruction and use it as an insertion point
        IR::Block::iterator reinsert_point{std::ranges::find_if_not(list, IR::IsPhi)};
        if (same.IsEmpty()) {
            // The phi is unreachable or in the start block
            // Insert an undefined instruction and make it the phi node replacement
            // The "phi" node reinsertion point is specified after this instruction
            reinsert_point = block->PrependNewInst(reinsert_point, undef_opcode);
            same = IR::Value{&*reinsert_point};
            ++reinsert_point;
        }
        // Reinsert the phi node and reroute all its uses to the "same" value
        list.insert(reinsert_point, phi);
        phi.ReplaceUsesWith(same);
        // TODO: Try to recursively remove all phi users, which might have become trivial
        return same;
    }

    // TODO: Windows dies with stack exhaustion?
    std::unordered_map<IR::Block*, std::map<Variant, IR::Inst*>> incomplete_phis;
    DefTable current_def;
    std::vector<ReadState> read_stack;
};

void VisitInst(Pass& pass, IR::Block* block, IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::SetRegister:
        if (const IR::Reg reg{inst.Arg(0).Reg()}; reg != IR::Reg::RZ) {
            pass.WriteVariable(reg, block, inst.Arg(1));
        }
        break;
    case IR::Opcode::SetPred:
        if (const IR::Pred pred{inst.Arg(0).Pred()}; pred != IR::Pred::PT) {
            pass.WriteVariable(pred, block, inst.Arg(1));
        }
        break;
    case IR::Opcode::SetGotoVariable:
        pass.WriteVariable(GotoVariable{inst.Arg(0).U32()}, block, inst.Arg(1));
        break;
    case IR::Opcode::SetIndirectBranchVariable:
        pass.WriteVariable(IndirectBranchVariable{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetZFlag:
        pass.WriteVariable(ZeroFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetSFlag:
        pass.WriteVariable(SignFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetCFlag:
        pass.WriteVariable(CarryFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetOFlag:
        pass.WriteVariable(OverflowFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::GetRegister:
        if (const IR::Reg reg{inst.Arg(0).Reg()}; reg != IR::Reg::RZ) {
            inst.ReplaceUsesWith(pass.ReadVariable(reg, block));
        }
        break;
    case IR::Opcode::GetPred:
        if (const IR::Pred pred{inst.Arg(0).Pred()}; pred != IR::Pred::PT) {
            inst.ReplaceUsesWith(pass.ReadVariable(pred, block));
        }
        break;
    case IR::Opcode::GetGotoVariable:
        inst.ReplaceUsesWith(pass.ReadVariable(GotoVariable{inst.Arg(0).U32()}, block));
        break;
    case IR::Opcode::GetIndirectBranchVariable:
        inst.ReplaceUsesWith(pass.ReadVariable(IndirectBranchVariable{}, block));
        break;
    case IR::Opcode::GetZFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(ZeroFlagTag{}, block));
        break;
    case IR::Opcode::GetSFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(SignFlagTag{}, block));
        break;
    case IR::Opcode::GetCFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(CarryFlagTag{}, block));
        break;
    case IR::Opcode::GetOFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(OverflowFlagTag{}, block));
        break;
    default:
        break;
    }
}

void VisitBlock(Pass& pass, IR::Block* block) {
    for (IR::Inst& inst : block->Instructions()) {
        VisitInst(pass, block, inst);
    }
    pass.SealBlock(block);
}

IR::Type GetConcreteType(IR::Inst* inst) {
    std::deque<IR::Inst*> queue;
    queue.push_back(inst);
    while (!queue.empty()) {
        IR::Inst* current = queue.front();
        queue.pop_front();
        const size_t num_args{current->NumArgs()};
        for (size_t i = 0; i < num_args; ++i) {
            const auto set_type = current->Arg(i).Type();
            if (set_type != IR::Type::Opaque) {
                return set_type;
            }
            if (!current->Arg(i).IsImmediate()) {
                queue.push_back(current->Arg(i).Inst());
            }
        }
    }
    return IR::Type::Opaque;
}
} // Anonymous namespace

void SsaRewritePass(IR::Program& program) {
    Pass pass;
    const auto end{program.post_order_blocks.rend()};
    for (auto block = program.post_order_blocks.rbegin(); block != end; ++block) {
        VisitBlock(pass, *block);
    }
    for (auto block = program.post_order_blocks.rbegin(); block != end; ++block) {
        for (IR::Inst& inst : (*block)->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::Phi) {
                if (inst.Type() == IR::Type::Opaque) {
                    inst.SetFlags(GetConcreteType(&inst));
                }
                inst.OrderPhiArgs();
            }
        }
    }
}

} // namespace Shader::Optimization
