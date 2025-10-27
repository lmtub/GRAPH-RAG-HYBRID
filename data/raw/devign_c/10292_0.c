static inline void tcg_out_tlb_load(TCGContext *s, TCGReg addrlo, TCGReg addrhi,

                                    int mem_index, TCGMemOp s_bits,

                                    tcg_insn_unit **label_ptr, int which)

{

    const TCGReg r0 = TCG_REG_L0;

    const TCGReg r1 = TCG_REG_L1;

    TCGType ttype = TCG_TYPE_I32;

    TCGType htype = TCG_TYPE_I32;

    int trexw = 0, hrexw = 0;



    if (TCG_TARGET_REG_BITS == 64) {

        if (TARGET_LONG_BITS == 64) {

            ttype = TCG_TYPE_I64;

            trexw = P_REXW;

        }

        if (TCG_TYPE_PTR == TCG_TYPE_I64) {

            htype = TCG_TYPE_I64;

            hrexw = P_REXW;

        }

    }



    tcg_out_mov(s, htype, r0, addrlo);

    tcg_out_mov(s, ttype, r1, addrlo);



    tcg_out_shifti(s, SHIFT_SHR + hrexw, r0,

                   TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS);



    tgen_arithi(s, ARITH_AND + trexw, r1,

                TARGET_PAGE_MASK | ((1 << s_bits) - 1), 0);

    tgen_arithi(s, ARITH_AND + hrexw, r0,

                (CPU_TLB_SIZE - 1) << CPU_TLB_ENTRY_BITS, 0);



    tcg_out_modrm_sib_offset(s, OPC_LEA + hrexw, r0, TCG_AREG0, r0, 0,

                             offsetof(CPUArchState, tlb_table[mem_index][0])

                             + which);



    /* cmp 0(r0), r1 */

    tcg_out_modrm_offset(s, OPC_CMP_GvEv + trexw, r1, r0, 0);



    /* Prepare for both the fast path add of the tlb addend, and the slow

       path function argument setup.  There are two cases worth note:

       For 32-bit guest and x86_64 host, MOVL zero-extends the guest address

       before the fastpath ADDQ below.  For 64-bit guest and x32 host, MOVQ

       copies the entire guest address for the slow path, while truncation

       for the 32-bit host happens with the fastpath ADDL below.  */

    tcg_out_mov(s, ttype, r1, addrlo);



    /* jne slow_path */

    tcg_out_opc(s, OPC_JCC_long + JCC_JNE, 0, 0, 0);

    label_ptr[0] = s->code_ptr;

    s->code_ptr += 4;



    if (TARGET_LONG_BITS > TCG_TARGET_REG_BITS) {

        /* cmp 4(r0), addrhi */

        tcg_out_modrm_offset(s, OPC_CMP_GvEv, addrhi, r0, 4);



        /* jne slow_path */

        tcg_out_opc(s, OPC_JCC_long + JCC_JNE, 0, 0, 0);

        label_ptr[1] = s->code_ptr;

        s->code_ptr += 4;

    }



    /* TLB Hit.  */



    /* add addend(r0), r1 */

    tcg_out_modrm_offset(s, OPC_ADD_GvEv + hrexw, r1, r0,

                         offsetof(CPUTLBEntry, addend) - which);

}
