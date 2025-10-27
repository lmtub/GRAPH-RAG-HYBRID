static void tcg_out_qemu_st(TCGContext *s, const TCGArg *args,
                            int opc)
{
    int addr_reg, data_reg, arg0, arg1, arg2, mem_index, s_bits;
#if defined(CONFIG_SOFTMMU)
    uint32_t *label1_ptr, *label2_ptr;
    data_reg = *args++;
    addr_reg = *args++;
    mem_index = *args;
    s_bits = opc;
    arg0 = TCG_REG_O0;
    arg1 = TCG_REG_O1;
    arg2 = TCG_REG_O2;
#if defined(CONFIG_SOFTMMU)
    /* srl addr_reg, x, arg1 */
    tcg_out_arithi(s, arg1, addr_reg, TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS,
                   SHIFT_SRL);
    tcg_out_arithi(s, arg0, addr_reg, TARGET_PAGE_MASK | ((1 << s_bits) - 1),
                   ARITH_AND);
    /* and arg1, x, arg1 */
    tcg_out_andi(s, arg1, (CPU_TLB_SIZE - 1) << CPU_TLB_ENTRY_BITS);
    /* add arg1, x, arg1 */
    tcg_out_addi(s, arg1, offsetof(CPUState,
                                   tlb_table[mem_index][0].addr_write));
    /* add env, arg1, arg1 */
    tcg_out_arith(s, arg1, TCG_AREG0, arg1, ARITH_ADD);
    /* ld [arg1], arg2 */
    tcg_out32(s, TARGET_LD_OP | INSN_RD(arg2) | INSN_RS1(arg1) |
              INSN_RS2(TCG_REG_G0));
    /* subcc arg0, arg2, %g0 */
    tcg_out_arith(s, TCG_REG_G0, arg0, arg2, ARITH_SUBCC);
    /* will become:
       be label1 */
    label1_ptr = (uint32_t *)s->code_ptr;
    tcg_out32(s, 0);
    /* mov (delay slot) */
    tcg_out_mov(s, arg0, addr_reg);
    /* mov */
    tcg_out_mov(s, arg1, data_reg);
    /* mov */
    tcg_out_movi(s, TCG_TYPE_I32, arg2, mem_index);
    /* XXX: move that code at the end of the TB */
    /* qemu_st_helper[s_bits](arg0, arg1, arg2) */
    tcg_out32(s, CALL | ((((tcg_target_ulong)qemu_st_helpers[s_bits]
                           - (tcg_target_ulong)s->code_ptr) >> 2)
                         & 0x3fffffff));
    /* Store AREG0 in stack to avoid ugly glibc bugs that mangle
       global registers */
    // delay slot
    tcg_out_ldst(s, TCG_AREG0, TCG_REG_CALL_STACK,
                 TCG_TARGET_CALL_STACK_OFFSET - sizeof(long), HOST_ST_OP);
    tcg_out_ldst(s, TCG_AREG0, TCG_REG_CALL_STACK,
                 TCG_TARGET_CALL_STACK_OFFSET - sizeof(long), HOST_LD_OP);
    /* will become:
       ba label2 */
    label2_ptr = (uint32_t *)s->code_ptr;
    tcg_out32(s, 0);
    /* nop (delay slot) */
    tcg_out_nop(s);
    /* label1: */
    *label1_ptr = (INSN_OP(0) | INSN_COND(COND_E, 0) | INSN_OP2(0x2) |
                   INSN_OFF22((unsigned long)s->code_ptr -
                              (unsigned long)label1_ptr));
    /* ld [arg1 + x], arg1 */
    tcg_out_ldst(s, arg1, arg1, offsetof(CPUTLBEntry, addend) -
                 offsetof(CPUTLBEntry, addr_write), HOST_LD_OP);
    /* add addr_reg, arg1, arg0 */
    tcg_out_arith(s, arg0, addr_reg, arg1, ARITH_ADD);
    arg0 = addr_reg;
    switch(opc) {
    case 0:
        /* stb data_reg, [arg0] */
        tcg_out_ldst(s, data_reg, arg0, 0, STB);
        break;
    case 1:
#ifdef TARGET_WORDS_BIGENDIAN
        /* sth data_reg, [arg0] */
        tcg_out_ldst(s, data_reg, arg0, 0, STH);
        /* stha data_reg, [arg0] ASI_PRIMARY_LITTLE */
        tcg_out_ldst_asi(s, data_reg, arg0, 0, STHA, ASI_PRIMARY_LITTLE);
        break;
    case 2:
#ifdef TARGET_WORDS_BIGENDIAN
        /* stw data_reg, [arg0] */
        tcg_out_ldst(s, data_reg, arg0, 0, STW);
        /* stwa data_reg, [arg0] ASI_PRIMARY_LITTLE */
        tcg_out_ldst_asi(s, data_reg, arg0, 0, STWA, ASI_PRIMARY_LITTLE);
        break;
    case 3:
#ifdef TARGET_WORDS_BIGENDIAN
        /* stx data_reg, [arg0] */
        tcg_out_ldst(s, data_reg, arg0, 0, STX);
        /* stxa data_reg, [arg0] ASI_PRIMARY_LITTLE */
        tcg_out_ldst_asi(s, data_reg, arg0, 0, STXA, ASI_PRIMARY_LITTLE);
        break;
    default:
        tcg_abort();
    }
#if defined(CONFIG_SOFTMMU)
    /* label2: */
    *label2_ptr = (INSN_OP(0) | INSN_COND(COND_A, 0) | INSN_OP2(0x2) |
                   INSN_OFF22((unsigned long)s->code_ptr -
                              (unsigned long)label2_ptr));
}