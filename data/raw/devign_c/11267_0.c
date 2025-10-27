static TCGReg tcg_out_tlb_read(TCGContext *s, TCGReg addrlo, TCGReg addrhi,

                               TCGMemOp opc, int mem_index, bool is_load)

{

    TCGReg base = TCG_AREG0;

    int cmp_off =

        (is_load

         ? offsetof(CPUArchState, tlb_table[mem_index][0].addr_read)

         : offsetof(CPUArchState, tlb_table[mem_index][0].addr_write));

    int add_off = offsetof(CPUArchState, tlb_table[mem_index][0].addend);

    unsigned s_bits = opc & MO_SIZE;

    unsigned a_bits = get_alignment_bits(opc);



    /* V7 generates the following:

     *   ubfx   r0, addrlo, #TARGET_PAGE_BITS, #CPU_TLB_BITS

     *   add    r2, env, #high

     *   add    r2, r2, r0, lsl #CPU_TLB_ENTRY_BITS

     *   ldr    r0, [r2, #cmp]

     *   ldr    r2, [r2, #add]

     *   movw   tmp, #page_align_mask

     *   bic    tmp, addrlo, tmp

     *   cmp    r0, tmp

     *

     * Otherwise we generate:

     *   shr    tmp, addrlo, #TARGET_PAGE_BITS

     *   add    r2, env, #high

     *   and    r0, tmp, #(CPU_TLB_SIZE - 1)

     *   add    r2, r2, r0, lsl #CPU_TLB_ENTRY_BITS

     *   ldr    r0, [r2, #cmp]

     *   ldr    r2, [r2, #add]

     *   tst    addrlo, #s_mask

     *   cmpeq  r0, tmp, lsl #TARGET_PAGE_BITS

     */

    if (use_armv7_instructions) {

        tcg_out_extract(s, COND_AL, TCG_REG_R0, addrlo,

                        TARGET_PAGE_BITS, CPU_TLB_BITS);

    } else {

        tcg_out_dat_reg(s, COND_AL, ARITH_MOV, TCG_REG_TMP,

                        0, addrlo, SHIFT_IMM_LSR(TARGET_PAGE_BITS));

    }



    /* We checked that the offset is contained within 16 bits above.  */

    if (add_off > 0xfff || (use_armv6_instructions && cmp_off > 0xff)) {

        tcg_out_dat_imm(s, COND_AL, ARITH_ADD, TCG_REG_R2, base,

                        (24 << 7) | (cmp_off >> 8));

        base = TCG_REG_R2;

        add_off -= cmp_off & 0xff00;

        cmp_off &= 0xff;

    }

    if (!use_armv7_instructions) {

        tcg_out_dat_imm(s, COND_AL, ARITH_AND,

                        TCG_REG_R0, TCG_REG_TMP, CPU_TLB_SIZE - 1);

    }

    tcg_out_dat_reg(s, COND_AL, ARITH_ADD, TCG_REG_R2, base,

                    TCG_REG_R0, SHIFT_IMM_LSL(CPU_TLB_ENTRY_BITS));



    /* Load the tlb comparator.  Use ldrd if needed and available,

       but due to how the pointer needs setting up, ldm isn't useful.

       Base arm5 doesn't have ldrd, but armv5te does.  */

    if (use_armv6_instructions && TARGET_LONG_BITS == 64) {

        tcg_out_ldrd_8(s, COND_AL, TCG_REG_R0, TCG_REG_R2, cmp_off);

    } else {

        tcg_out_ld32_12(s, COND_AL, TCG_REG_R0, TCG_REG_R2, cmp_off);

        if (TARGET_LONG_BITS == 64) {

            tcg_out_ld32_12(s, COND_AL, TCG_REG_R1, TCG_REG_R2, cmp_off + 4);

        }

    }



    /* Load the tlb addend.  */

    tcg_out_ld32_12(s, COND_AL, TCG_REG_R2, TCG_REG_R2, add_off);



    /* Check alignment.  We don't support inline unaligned acceses,

       but we can easily support overalignment checks.  */

    if (a_bits < s_bits) {

        a_bits = s_bits;

    }



    if (use_armv7_instructions) {

        tcg_target_ulong mask = ~(TARGET_PAGE_MASK | ((1 << a_bits) - 1));

        int rot = encode_imm(mask);



        if (rot >= 0) { 

            tcg_out_dat_imm(s, COND_AL, ARITH_BIC, TCG_REG_TMP, addrlo,

                            rotl(mask, rot) | (rot << 7));

        } else {

            tcg_out_movi32(s, COND_AL, TCG_REG_TMP, mask);

            tcg_out_dat_reg(s, COND_AL, ARITH_BIC, TCG_REG_TMP,

                            addrlo, TCG_REG_TMP, 0);

        }

        tcg_out_dat_reg(s, COND_AL, ARITH_CMP, 0, TCG_REG_R0, TCG_REG_TMP, 0);

    } else {

        if (a_bits) {

            tcg_out_dat_imm(s, COND_AL, ARITH_TST, 0, addrlo,

                            (1 << a_bits) - 1);

        }

        tcg_out_dat_reg(s, (a_bits ? COND_EQ : COND_AL), ARITH_CMP,

                        0, TCG_REG_R0, TCG_REG_TMP,

                        SHIFT_IMM_LSL(TARGET_PAGE_BITS));

    }



    if (TARGET_LONG_BITS == 64) {

        tcg_out_dat_reg(s, COND_EQ, ARITH_CMP, 0, TCG_REG_R1, addrhi, 0);

    }



    return TCG_REG_R2;

}
