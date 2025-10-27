static TCGReg tcg_out_tlb_load(TCGContext *s, TCGReg addr, int mem_index,

                               TCGMemOp s_bits, int which)

{

    const TCGReg r0 = TCG_REG_O0;

    const TCGReg r1 = TCG_REG_O1;

    const TCGReg r2 = TCG_REG_O2;

    int tlb_ofs;



    /* Shift the page number down.  */

    tcg_out_arithi(s, r1, addr, TARGET_PAGE_BITS, SHIFT_SRL);



    /* Mask out the page offset, except for the required alignment.  */

    tcg_out_movi(s, TCG_TYPE_TL, TCG_REG_T1,

                 TARGET_PAGE_MASK | ((1 << s_bits) - 1));



    /* Mask the tlb index.  */

    tcg_out_arithi(s, r1, r1, CPU_TLB_SIZE - 1, ARITH_AND);

    

    /* Mask page, part 2.  */

    tcg_out_arith(s, r0, addr, TCG_REG_T1, ARITH_AND);



    /* Shift the tlb index into place.  */

    tcg_out_arithi(s, r1, r1, CPU_TLB_ENTRY_BITS, SHIFT_SLL);



    /* Relative to the current ENV.  */

    tcg_out_arith(s, r1, TCG_AREG0, r1, ARITH_ADD);



    /* Find a base address that can load both tlb comparator and addend.  */

    tlb_ofs = offsetof(CPUArchState, tlb_table[mem_index][0]);

    if (!check_fit_tl(tlb_ofs + sizeof(CPUTLBEntry), 13)) {

        tcg_out_addi(s, r1, tlb_ofs & ~0x3ff);

        tlb_ofs &= 0x3ff;

    }



    /* Load the tlb comparator and the addend.  */

    tcg_out_ld(s, TCG_TYPE_TL, r2, r1, tlb_ofs + which);

    tcg_out_ld(s, TCG_TYPE_PTR, r1, r1, tlb_ofs+offsetof(CPUTLBEntry, addend));



    /* subcc arg0, arg2, %g0 */

    tcg_out_cmp(s, r0, r2, 0);



    /* If the guest address must be zero-extended, do so now.  */

    if (SPARC64 && TARGET_LONG_BITS == 32) {

        tcg_out_arithi(s, r0, addr, 0, SHIFT_SRL);

        return r0;

    }

    return addr;

}
