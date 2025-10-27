static TCGReg tcg_out_tlb_read(TCGContext* s, TCGReg addr_reg, TCGMemOp opc,

                               int mem_index, bool is_ld)

{

    int s_mask = (1 << (opc & MO_SIZE)) - 1;

    int ofs, a_off;

    uint64_t tlb_mask;



    /* For aligned accesses, we check the first byte and include the alignment

       bits within the address.  For unaligned access, we check that we don't

       cross pages using the address of the last byte of the access.  */

    if ((opc & MO_AMASK) == MO_ALIGN || s_mask == 0) {

        a_off = 0;

        tlb_mask = TARGET_PAGE_MASK | s_mask;

    } else {

        a_off = s_mask;

        tlb_mask = TARGET_PAGE_MASK;

    }



    if (facilities & FACILITY_GEN_INST_EXT) {

        tcg_out_risbg(s, TCG_REG_R2, addr_reg,

                      64 - CPU_TLB_BITS - CPU_TLB_ENTRY_BITS,

                      63 - CPU_TLB_ENTRY_BITS,

                      64 + CPU_TLB_ENTRY_BITS - TARGET_PAGE_BITS, 1);

        if (a_off) {

            tcg_out_insn(s, RX, LA, TCG_REG_R3, addr_reg, TCG_REG_NONE, a_off);

            tgen_andi(s, TCG_TYPE_TL, TCG_REG_R3, tlb_mask);

        } else {

            tgen_andi_risbg(s, TCG_REG_R3, addr_reg, tlb_mask);

        }

    } else {

        tcg_out_sh64(s, RSY_SRLG, TCG_REG_R2, addr_reg, TCG_REG_NONE,

                     TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS);

        tcg_out_insn(s, RX, LA, TCG_REG_R3, addr_reg, TCG_REG_NONE, a_off);

        tgen_andi(s, TCG_TYPE_I64, TCG_REG_R2,

                  (CPU_TLB_SIZE - 1) << CPU_TLB_ENTRY_BITS);

        tgen_andi(s, TCG_TYPE_TL, TCG_REG_R3, tlb_mask);

    }



    if (is_ld) {

        ofs = offsetof(CPUArchState, tlb_table[mem_index][0].addr_read);

    } else {

        ofs = offsetof(CPUArchState, tlb_table[mem_index][0].addr_write);

    }

    if (TARGET_LONG_BITS == 32) {

        tcg_out_mem(s, RX_C, RXY_CY, TCG_REG_R3, TCG_REG_R2, TCG_AREG0, ofs);

    } else {

        tcg_out_mem(s, 0, RXY_CG, TCG_REG_R3, TCG_REG_R2, TCG_AREG0, ofs);

    }



    ofs = offsetof(CPUArchState, tlb_table[mem_index][0].addend);

    tcg_out_mem(s, 0, RXY_LG, TCG_REG_R2, TCG_REG_R2, TCG_AREG0, ofs);



    if (TARGET_LONG_BITS == 32) {

        tgen_ext32u(s, TCG_REG_R3, addr_reg);

        return TCG_REG_R3;

    }

    return addr_reg;

}
