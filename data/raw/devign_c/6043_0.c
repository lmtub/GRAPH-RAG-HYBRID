static inline void tcg_out_op(TCGContext *s, TCGOpcode opc, const TCGArg *args,

                              const int *const_args)

{

    int c;



    switch (opc) {

    case INDEX_op_exit_tb:

        if (check_fit_tl(args[0], 13)) {

            tcg_out_arithi(s, TCG_REG_G0, TCG_REG_I7, 8, RETURN);

            tcg_out_movi_imm13(s, TCG_REG_O0, args[0]);

        } else {

            tcg_out_movi(s, TCG_TYPE_PTR, TCG_REG_I0, args[0] & ~0x3ff);

            tcg_out_arithi(s, TCG_REG_G0, TCG_REG_I7, 8, RETURN);

            tcg_out_arithi(s, TCG_REG_O0, TCG_REG_O0,

                           args[0] & 0x3ff, ARITH_OR);

        }

        break;

    case INDEX_op_goto_tb:

        if (s->tb_jmp_offset) {

            /* direct jump method */

            uint32_t old_insn = *(uint32_t *)s->code_ptr;

            s->tb_jmp_offset[args[0]] = s->code_ptr - s->code_buf;

            /* Make sure to preserve links during retranslation.  */

            tcg_out32(s, CALL | (old_insn & ~INSN_OP(-1)));

        } else {

            /* indirect jump method */

            tcg_out_ld_ptr(s, TCG_REG_T1, (uintptr_t)(s->tb_next + args[0]));

            tcg_out_arithi(s, TCG_REG_G0, TCG_REG_T1, 0, JMPL);

        }

        tcg_out_nop(s);

        s->tb_next_offset[args[0]] = s->code_ptr - s->code_buf;

        break;

    case INDEX_op_call:

        if (const_args[0]) {

            tcg_out_calli(s, args[0]);

        } else {

            tcg_out_arithi(s, TCG_REG_O7, args[0], 0, JMPL);

        }

        /* delay slot */

        tcg_out_nop(s);

        break;

    case INDEX_op_br:

        tcg_out_bpcc(s, COND_A, BPCC_PT, args[0]);

        tcg_out_nop(s);

        break;

    case INDEX_op_movi_i32:

        tcg_out_movi(s, TCG_TYPE_I32, args[0], (uint32_t)args[1]);

        break;



#define OP_32_64(x)                             \

        glue(glue(case INDEX_op_, x), _i32):    \

        glue(glue(case INDEX_op_, x), _i64)



    OP_32_64(ld8u):

        tcg_out_ldst(s, args[0], args[1], args[2], LDUB);

        break;

    OP_32_64(ld8s):

        tcg_out_ldst(s, args[0], args[1], args[2], LDSB);

        break;

    OP_32_64(ld16u):

        tcg_out_ldst(s, args[0], args[1], args[2], LDUH);

        break;

    OP_32_64(ld16s):

        tcg_out_ldst(s, args[0], args[1], args[2], LDSH);

        break;

    case INDEX_op_ld_i32:

    case INDEX_op_ld32u_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], LDUW);

        break;

    OP_32_64(st8):

        tcg_out_ldst(s, args[0], args[1], args[2], STB);

        break;

    OP_32_64(st16):

        tcg_out_ldst(s, args[0], args[1], args[2], STH);

        break;

    case INDEX_op_st_i32:

    case INDEX_op_st32_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], STW);

        break;

    OP_32_64(add):

        c = ARITH_ADD;

        goto gen_arith;

    OP_32_64(sub):

        c = ARITH_SUB;

        goto gen_arith;

    OP_32_64(and):

        c = ARITH_AND;

        goto gen_arith;

    OP_32_64(andc):

        c = ARITH_ANDN;

        goto gen_arith;

    OP_32_64(or):

        c = ARITH_OR;

        goto gen_arith;

    OP_32_64(orc):

        c = ARITH_ORN;

        goto gen_arith;

    OP_32_64(xor):

        c = ARITH_XOR;

        goto gen_arith;

    case INDEX_op_shl_i32:

        c = SHIFT_SLL;

    do_shift32:

        /* Limit immediate shift count lest we create an illegal insn.  */

        tcg_out_arithc(s, args[0], args[1], args[2] & 31, const_args[2], c);

        break;

    case INDEX_op_shr_i32:

        c = SHIFT_SRL;

        goto do_shift32;

    case INDEX_op_sar_i32:

        c = SHIFT_SRA;

        goto do_shift32;

    case INDEX_op_mul_i32:

        c = ARITH_UMUL;

        goto gen_arith;



    OP_32_64(neg):

	c = ARITH_SUB;

	goto gen_arith1;

    OP_32_64(not):

	c = ARITH_ORN;

	goto gen_arith1;



    case INDEX_op_div_i32:

        tcg_out_div32(s, args[0], args[1], args[2], const_args[2], 0);

        break;

    case INDEX_op_divu_i32:

        tcg_out_div32(s, args[0], args[1], args[2], const_args[2], 1);

        break;



    case INDEX_op_brcond_i32:

        tcg_out_brcond_i32(s, args[2], args[0], args[1], const_args[1],

                           args[3]);

        break;

    case INDEX_op_setcond_i32:

        tcg_out_setcond_i32(s, args[3], args[0], args[1],

                            args[2], const_args[2]);

        break;

    case INDEX_op_movcond_i32:

        tcg_out_movcond_i32(s, args[5], args[0], args[1],

                            args[2], const_args[2], args[3], const_args[3]);

        break;



    case INDEX_op_add2_i32:

        tcg_out_addsub2(s, args[0], args[1], args[2], args[3],

                        args[4], const_args[4], args[5], const_args[5],

                        ARITH_ADDCC, ARITH_ADDX);

        break;

    case INDEX_op_sub2_i32:

        tcg_out_addsub2(s, args[0], args[1], args[2], args[3],

                        args[4], const_args[4], args[5], const_args[5],

                        ARITH_SUBCC, ARITH_SUBX);

        break;

    case INDEX_op_mulu2_i32:

        c = ARITH_UMUL;

        goto do_mul2;

    case INDEX_op_muls2_i32:

        c = ARITH_SMUL;

    do_mul2:

        /* The 32-bit multiply insns produce a full 64-bit result.  If the

           destination register can hold it, we can avoid the slower RDY.  */

        tcg_out_arithc(s, args[0], args[2], args[3], const_args[3], c);

        if (SPARC64 || args[0] <= TCG_REG_O7) {

            tcg_out_arithi(s, args[1], args[0], 32, SHIFT_SRLX);

        } else {

            tcg_out_rdy(s, args[1]);

        }

        break;



    case INDEX_op_qemu_ld_i32:

        tcg_out_qemu_ld(s, args[0], args[1], args[2], args[3], false);

        break;

    case INDEX_op_qemu_ld_i64:

        tcg_out_qemu_ld(s, args[0], args[1], args[2], args[3], true);

        break;

    case INDEX_op_qemu_st_i32:

        tcg_out_qemu_st(s, args[0], args[1], args[2], args[3]);

        break;

    case INDEX_op_qemu_st_i64:

        tcg_out_qemu_st(s, args[0], args[1], args[2], args[3]);

        break;



    case INDEX_op_movi_i64:

        tcg_out_movi(s, TCG_TYPE_I64, args[0], args[1]);

        break;

    case INDEX_op_ld32s_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], LDSW);

        break;

    case INDEX_op_ld_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], LDX);

        break;

    case INDEX_op_st_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], STX);

        break;

    case INDEX_op_shl_i64:

        c = SHIFT_SLLX;

    do_shift64:

        /* Limit immediate shift count lest we create an illegal insn.  */

        tcg_out_arithc(s, args[0], args[1], args[2] & 63, const_args[2], c);

        break;

    case INDEX_op_shr_i64:

        c = SHIFT_SRLX;

        goto do_shift64;

    case INDEX_op_sar_i64:

        c = SHIFT_SRAX;

        goto do_shift64;

    case INDEX_op_mul_i64:

        c = ARITH_MULX;

        goto gen_arith;

    case INDEX_op_div_i64:

        c = ARITH_SDIVX;

        goto gen_arith;

    case INDEX_op_divu_i64:

        c = ARITH_UDIVX;

        goto gen_arith;

    case INDEX_op_ext32s_i64:

        tcg_out_arithi(s, args[0], args[1], 0, SHIFT_SRA);

        break;

    case INDEX_op_ext32u_i64:

        tcg_out_arithi(s, args[0], args[1], 0, SHIFT_SRL);

        break;

    case INDEX_op_trunc_shr_i32:

        if (args[2] == 0) {

            tcg_out_mov(s, TCG_TYPE_I32, args[0], args[1]);

        } else {

            tcg_out_arithi(s, args[0], args[1], args[2], SHIFT_SRLX);

        }

        break;



    case INDEX_op_brcond_i64:

        tcg_out_brcond_i64(s, args[2], args[0], args[1], const_args[1],

                           args[3]);

        break;

    case INDEX_op_setcond_i64:

        tcg_out_setcond_i64(s, args[3], args[0], args[1],

                            args[2], const_args[2]);

        break;

    case INDEX_op_movcond_i64:

        tcg_out_movcond_i64(s, args[5], args[0], args[1],

                            args[2], const_args[2], args[3], const_args[3]);

        break;



    gen_arith:

        tcg_out_arithc(s, args[0], args[1], args[2], const_args[2], c);

        break;



    gen_arith1:

	tcg_out_arithc(s, args[0], TCG_REG_G0, args[1], const_args[1], c);

	break;



    default:

        fprintf(stderr, "unknown opcode 0x%x\n", opc);

        tcg_abort();

    }

}
