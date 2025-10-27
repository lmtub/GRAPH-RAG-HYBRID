static void tcg_out_qemu_st(TCGContext *s, TCGReg data, TCGReg addr,

                            TCGMemOpIdx oi)

{

    TCGMemOp memop = get_memop(oi);

#ifdef CONFIG_SOFTMMU

    unsigned memi = get_mmuidx(oi);

    TCGReg addrz, param;

    tcg_insn_unit *func;

    tcg_insn_unit *label_ptr;



    addrz = tcg_out_tlb_load(s, addr, memi, memop & MO_SIZE,

                             offsetof(CPUTLBEntry, addr_write));



    /* The fast path is exactly one insn.  Thus we can perform the entire

       TLB Hit in the (annulled) delay slot of the branch over TLB Miss.  */

    /* beq,a,pt %[xi]cc, label0 */

    label_ptr = s->code_ptr;

    tcg_out_bpcc0(s, COND_E, BPCC_A | BPCC_PT

                  | (TARGET_LONG_BITS == 64 ? BPCC_XCC : BPCC_ICC), 0);

    /* delay slot */

    tcg_out_ldst_rr(s, data, addrz, TCG_REG_O1,

                    qemu_st_opc[memop & (MO_BSWAP | MO_SIZE)]);



    /* TLB Miss.  */



    param = TCG_REG_O1;

    if (!SPARC64 && TARGET_LONG_BITS == 64) {

        /* Skip the high-part; we'll perform the extract in the trampoline.  */

        param++;

    }

    tcg_out_mov(s, TCG_TYPE_REG, param++, addr);

    if (!SPARC64 && (memop & MO_SIZE) == MO_64) {

        /* Skip the high-part; we'll perform the extract in the trampoline.  */

        param++;

    }

    tcg_out_mov(s, TCG_TYPE_REG, param++, data);



    func = qemu_st_trampoline[memop & (MO_BSWAP | MO_SIZE)];

    assert(func != NULL);

    tcg_out_call_nodelay(s, func);

    /* delay slot */

    tcg_out_movi(s, TCG_TYPE_I32, param, oi);



    *label_ptr |= INSN_OFF19(tcg_ptr_byte_diff(s->code_ptr, label_ptr));

#else

    if (SPARC64 && TARGET_LONG_BITS == 32) {

        tcg_out_arithi(s, TCG_REG_T1, addr, 0, SHIFT_SRL);

        addr = TCG_REG_T1;

    }

    tcg_out_ldst_rr(s, data, addr,

                    (guest_base ? TCG_GUEST_BASE_REG : TCG_REG_G0),

                    qemu_st_opc[memop & (MO_BSWAP | MO_SIZE)]);

#endif /* CONFIG_SOFTMMU */

}
