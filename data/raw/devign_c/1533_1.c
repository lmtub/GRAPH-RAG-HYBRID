static void tcg_out_qemu_ld(TCGContext *s, const TCGArg *args, bool is_64)

{

    TCGReg datalo, datahi, addrlo, rbase;

    TCGReg addrhi __attribute__((unused));

    TCGMemOpIdx oi;

    TCGMemOp opc, s_bits;

#ifdef CONFIG_SOFTMMU

    int mem_index;

    tcg_insn_unit *label_ptr;

#endif



    datalo = *args++;

    datahi = (TCG_TARGET_REG_BITS == 32 && is_64 ? *args++ : 0);

    addrlo = *args++;

    addrhi = (TCG_TARGET_REG_BITS < TARGET_LONG_BITS ? *args++ : 0);

    oi = *args++;

    opc = get_memop(oi);

    s_bits = opc & MO_SIZE;



#ifdef CONFIG_SOFTMMU

    mem_index = get_mmuidx(oi);

    addrlo = tcg_out_tlb_read(s, s_bits, addrlo, addrhi, mem_index, true);



    /* Load a pointer into the current opcode w/conditional branch-link. */

    label_ptr = s->code_ptr;

    tcg_out_bc_noaddr(s, BC | BI(7, CR_EQ) | BO_COND_FALSE | LK);



    rbase = TCG_REG_R3;

#else  /* !CONFIG_SOFTMMU */

    rbase = GUEST_BASE ? TCG_GUEST_BASE_REG : 0;

    if (TCG_TARGET_REG_BITS > TARGET_LONG_BITS) {

        tcg_out_ext32u(s, TCG_REG_TMP1, addrlo);

        addrlo = TCG_REG_TMP1;

    }

#endif



    if (TCG_TARGET_REG_BITS == 32 && s_bits == MO_64) {

        if (opc & MO_BSWAP) {

            tcg_out32(s, ADDI | TAI(TCG_REG_R0, addrlo, 4));

            tcg_out32(s, LWBRX | TAB(datalo, rbase, addrlo));

            tcg_out32(s, LWBRX | TAB(datahi, rbase, TCG_REG_R0));

        } else if (rbase != 0) {

            tcg_out32(s, ADDI | TAI(TCG_REG_R0, addrlo, 4));

            tcg_out32(s, LWZX | TAB(datahi, rbase, addrlo));

            tcg_out32(s, LWZX | TAB(datalo, rbase, TCG_REG_R0));

        } else if (addrlo == datahi) {

            tcg_out32(s, LWZ | TAI(datalo, addrlo, 4));

            tcg_out32(s, LWZ | TAI(datahi, addrlo, 0));

        } else {

            tcg_out32(s, LWZ | TAI(datahi, addrlo, 0));

            tcg_out32(s, LWZ | TAI(datalo, addrlo, 4));

        }

    } else {

        uint32_t insn = qemu_ldx_opc[opc & (MO_BSWAP | MO_SSIZE)];

        if (!HAVE_ISA_2_06 && insn == LDBRX) {

            tcg_out32(s, ADDI | TAI(TCG_REG_R0, addrlo, 4));

            tcg_out32(s, LWBRX | TAB(datalo, rbase, addrlo));

            tcg_out32(s, LWBRX | TAB(TCG_REG_R0, rbase, TCG_REG_R0));

            tcg_out_rld(s, RLDIMI, datalo, TCG_REG_R0, 32, 0);

        } else if (insn) {

            tcg_out32(s, insn | TAB(datalo, rbase, addrlo));

        } else {

            insn = qemu_ldx_opc[opc & (MO_SIZE | MO_BSWAP)];

            tcg_out32(s, insn | TAB(datalo, rbase, addrlo));

            insn = qemu_exts_opc[s_bits];

            tcg_out32(s, insn | RA(datalo) | RS(datalo));

        }

    }



#ifdef CONFIG_SOFTMMU

    add_qemu_ldst_label(s, true, oi, datalo, datahi, addrlo, addrhi,

                        s->code_ptr, label_ptr);

#endif

}
