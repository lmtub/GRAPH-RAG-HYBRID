static inline void tcg_out_qemu_st(TCGContext *s, const TCGArg *args, int opc)

{

    int addr_reg, data_reg, data_reg2, bswap;

#ifdef CONFIG_SOFTMMU

    int mem_index, s_bits;

# if TARGET_LONG_BITS == 64

    int addr_reg2;

# endif

    uint32_t *label_ptr;

#endif



#ifdef TARGET_WORDS_BIGENDIAN

    bswap = 1;

#else

    bswap = 0;

#endif

    data_reg = *args++;

    if (opc == 3)

        data_reg2 = *args++;

    else

        data_reg2 = 0; /* suppress warning */

    addr_reg = *args++;

#ifdef CONFIG_SOFTMMU

# if TARGET_LONG_BITS == 64

    addr_reg2 = *args++;

# endif

    mem_index = *args;

    s_bits = opc & 3;



    /* Should generate something like the following:

     *  shr r8, addr_reg, #TARGET_PAGE_BITS

     *  and r0, r8, #(CPU_TLB_SIZE - 1)   @ Assumption: CPU_TLB_BITS <= 8

     *  add r0, env, r0 lsl #CPU_TLB_ENTRY_BITS

     */

    tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                    TCG_REG_R8, 0, addr_reg, SHIFT_IMM_LSR(TARGET_PAGE_BITS));

    tcg_out_dat_imm(s, COND_AL, ARITH_AND,

                    TCG_REG_R0, TCG_REG_R8, CPU_TLB_SIZE - 1);

    tcg_out_dat_reg(s, COND_AL, ARITH_ADD, TCG_REG_R0,

                    TCG_AREG0, TCG_REG_R0, SHIFT_IMM_LSL(CPU_TLB_ENTRY_BITS));

    /* In the

     *  ldr r1 [r0, #(offsetof(CPUState, tlb_table[mem_index][0].addr_write))]

     * below, the offset is likely to exceed 12 bits if mem_index != 0 and

     * not exceed otherwise, so use an

     *  add r0, r0, #(mem_index * sizeof *CPUState.tlb_table)

     * before.

     */

    if (mem_index)

        tcg_out_dat_imm(s, COND_AL, ARITH_ADD, TCG_REG_R0, TCG_REG_R0,

                        (mem_index << (TLB_SHIFT & 1)) |

                        ((16 - (TLB_SHIFT >> 1)) << 8));

    tcg_out_ld32_12(s, COND_AL, TCG_REG_R1, TCG_REG_R0,

                    offsetof(CPUState, tlb_table[0][0].addr_write));

    tcg_out_dat_reg(s, COND_AL, ARITH_CMP, 0, TCG_REG_R1,

                    TCG_REG_R8, SHIFT_IMM_LSL(TARGET_PAGE_BITS));

    /* Check alignment.  */

    if (s_bits)

        tcg_out_dat_imm(s, COND_EQ, ARITH_TST,

                        0, addr_reg, (1 << s_bits) - 1);

#  if TARGET_LONG_BITS == 64

    /* XXX: possibly we could use a block data load or writeback in

     * the first access.  */

    tcg_out_ld32_12(s, COND_EQ, TCG_REG_R1, TCG_REG_R0,

                    offsetof(CPUState, tlb_table[0][0].addr_write) + 4);

    tcg_out_dat_reg(s, COND_EQ, ARITH_CMP, 0,

                    TCG_REG_R1, addr_reg2, SHIFT_IMM_LSL(0));

#  endif

    tcg_out_ld32_12(s, COND_EQ, TCG_REG_R1, TCG_REG_R0,

                    offsetof(CPUState, tlb_table[0][0].addend));



    switch (opc) {

    case 0:

        tcg_out_st8_r(s, COND_EQ, data_reg, addr_reg, TCG_REG_R1);

        break;

    case 1:

        if (bswap) {

            tcg_out_bswap16(s, COND_EQ, TCG_REG_R0, data_reg);

            tcg_out_st16_r(s, COND_EQ, TCG_REG_R0, addr_reg, TCG_REG_R1);

        } else {

            tcg_out_st16_r(s, COND_EQ, data_reg, addr_reg, TCG_REG_R1);

        }

        break;

    case 2:

    default:

        if (bswap) {

            tcg_out_bswap32(s, COND_EQ, TCG_REG_R0, data_reg);

            tcg_out_st32_r(s, COND_EQ, TCG_REG_R0, addr_reg, TCG_REG_R1);

        } else {

            tcg_out_st32_r(s, COND_EQ, data_reg, addr_reg, TCG_REG_R1);

        }

        break;

    case 3:

        if (bswap) {

            tcg_out_bswap32(s, COND_EQ, TCG_REG_R0, data_reg2);

            tcg_out_st32_rwb(s, COND_EQ, TCG_REG_R0, TCG_REG_R1, addr_reg);

            tcg_out_bswap32(s, COND_EQ, TCG_REG_R0, data_reg);

            tcg_out_st32_12(s, COND_EQ, data_reg, TCG_REG_R1, 4);

        } else {

            tcg_out_st32_rwb(s, COND_EQ, data_reg, TCG_REG_R1, addr_reg);

            tcg_out_st32_12(s, COND_EQ, data_reg2, TCG_REG_R1, 4);

        }

        break;

    }



    label_ptr = (void *) s->code_ptr;

    tcg_out_b(s, COND_EQ, 8);



    /* TODO: move this code to where the constants pool will be */

    if (addr_reg != TCG_REG_R0) {

        tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                        TCG_REG_R0, 0, addr_reg, SHIFT_IMM_LSL(0));

    }

# if TARGET_LONG_BITS == 32

    switch (opc) {

    case 0:

        tcg_out_ext8u(s, COND_AL, TCG_REG_R1, data_reg);

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R2, 0, mem_index);

        break;

    case 1:

        tcg_out_ext16u(s, COND_AL, TCG_REG_R1, data_reg);

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R2, 0, mem_index);

        break;

    case 2:

        if (data_reg != TCG_REG_R1) {

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                            TCG_REG_R1, 0, data_reg, SHIFT_IMM_LSL(0));

        }

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R2, 0, mem_index);

        break;

    case 3:

        if (data_reg != TCG_REG_R1) {

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                            TCG_REG_R1, 0, data_reg, SHIFT_IMM_LSL(0));

        }

        if (data_reg2 != TCG_REG_R2) {

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                            TCG_REG_R2, 0, data_reg2, SHIFT_IMM_LSL(0));

        }

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R3, 0, mem_index);

        break;

    }

# else

    if (addr_reg2 != TCG_REG_R1) {

        tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                        TCG_REG_R1, 0, addr_reg2, SHIFT_IMM_LSL(0));

    }

    switch (opc) {

    case 0:

        tcg_out_ext8u(s, COND_AL, TCG_REG_R2, data_reg);

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R3, 0, mem_index);

        break;

    case 1:

        tcg_out_ext16u(s, COND_AL, TCG_REG_R2, data_reg);

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R3, 0, mem_index);

        break;

    case 2:

        if (data_reg != TCG_REG_R2) {

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                            TCG_REG_R2, 0, data_reg, SHIFT_IMM_LSL(0));

        }

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R3, 0, mem_index);

        break;

    case 3:

        tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R8, 0, mem_index);

        tcg_out32(s, (COND_AL << 28) | 0x052d8010); /* str r8, [sp, #-0x10]! */

        if (data_reg != TCG_REG_R2) {

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                            TCG_REG_R2, 0, data_reg, SHIFT_IMM_LSL(0));

        }

        if (data_reg2 != TCG_REG_R3) {

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                            TCG_REG_R3, 0, data_reg2, SHIFT_IMM_LSL(0));

        }

        break;

    }

# endif



    tcg_out_bl(s, COND_AL, (tcg_target_long) qemu_st_helpers[s_bits] -

                    (tcg_target_long) s->code_ptr);

# if TARGET_LONG_BITS == 64

    if (opc == 3)

        tcg_out_dat_imm(s, COND_AL, ARITH_ADD, TCG_REG_R13, TCG_REG_R13, 0x10);

# endif



    *label_ptr += ((void *) s->code_ptr - (void *) label_ptr - 8) >> 2;

#else /* !CONFIG_SOFTMMU */

    if (GUEST_BASE) {

        uint32_t offset = GUEST_BASE;

        int i;

        int rot;



        while (offset) {

            i = ctz32(offset) & ~1;

            rot = ((32 - i) << 7) & 0xf00;



            tcg_out_dat_imm(s, COND_AL, ARITH_ADD, TCG_REG_R1, addr_reg,

                            ((offset >> i) & 0xff) | rot);

            addr_reg = TCG_REG_R1;

            offset &= ~(0xff << i);

        }

    }

    switch (opc) {

    case 0:

        tcg_out_st8_12(s, COND_AL, data_reg, addr_reg, 0);

        break;

    case 1:

        if (bswap) {

            tcg_out_bswap16(s, COND_AL, TCG_REG_R0, data_reg);

            tcg_out_st16_8(s, COND_AL, TCG_REG_R0, addr_reg, 0);

        } else {

            tcg_out_st16_8(s, COND_AL, data_reg, addr_reg, 0);

        }

        break;

    case 2:

    default:

        if (bswap) {

            tcg_out_bswap32(s, COND_AL, TCG_REG_R0, data_reg);

            tcg_out_st32_12(s, COND_AL, TCG_REG_R0, addr_reg, 0);

        } else {

            tcg_out_st32_12(s, COND_AL, data_reg, addr_reg, 0);

        }

        break;

    case 3:

        /* TODO: use block store -

         * check that data_reg2 > data_reg or the other way */

        if (bswap) {

            tcg_out_bswap32(s, COND_AL, TCG_REG_R0, data_reg2);

            tcg_out_st32_12(s, COND_AL, TCG_REG_R0, addr_reg, 0);

            tcg_out_bswap32(s, COND_AL, TCG_REG_R0, data_reg);

            tcg_out_st32_12(s, COND_AL, TCG_REG_R0, addr_reg, 4);

        } else {

            tcg_out_st32_12(s, COND_AL, data_reg, addr_reg, 0);

            tcg_out_st32_12(s, COND_AL, data_reg2, addr_reg, 4);

        }

        break;

    }

#endif

}
