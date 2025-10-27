static inline void tcg_out_op(TCGContext *s, int opc, const TCGArg *args,

                              const int *const_args)

{

    int c;



    switch (opc) {

    case INDEX_op_exit_tb:

        tcg_out_movi(s, TCG_TYPE_PTR, TCG_REG_I0, args[0]);

        tcg_out32(s, JMPL | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_I7) |

                  INSN_IMM13(8));

        tcg_out32(s, RESTORE | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_G0) |

                      INSN_RS2(TCG_REG_G0));

        break;

    case INDEX_op_goto_tb:

        if (s->tb_jmp_offset) {

            /* direct jump method */

            if (ABS(args[0] - (unsigned long)s->code_ptr) ==

                (ABS(args[0] - (unsigned long)s->code_ptr) & 0x1fffff)) {

                tcg_out32(s, BA |

                          INSN_OFF22(args[0] - (unsigned long)s->code_ptr));

            } else {

                tcg_out_movi(s, TCG_TYPE_PTR, TCG_REG_I5, args[0]);

                tcg_out32(s, JMPL | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_I5) |

                          INSN_RS2(TCG_REG_G0));

            }

            s->tb_jmp_offset[args[0]] = s->code_ptr - s->code_buf;

        } else {

            /* indirect jump method */

            tcg_out_ld_ptr(s, TCG_REG_I5, (tcg_target_long)(s->tb_next + args[0]));

            tcg_out32(s, JMPL | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_I5) |

                      INSN_RS2(TCG_REG_G0));

        }

        tcg_out_nop(s);

        s->tb_next_offset[args[0]] = s->code_ptr - s->code_buf;

        break;

    case INDEX_op_call:

        if (const_args[0]) {

            tcg_out32(s, CALL | ((((tcg_target_ulong)args[0]

                                  - (tcg_target_ulong)s->code_ptr) >> 2)

                                 & 0x3fffffff));

            tcg_out_nop(s);

        } else {

            tcg_out_ld_ptr(s, TCG_REG_O7, (tcg_target_long)(s->tb_next + args[0]));

            tcg_out32(s, JMPL | INSN_RD(TCG_REG_O7) | INSN_RS1(TCG_REG_O7) |

                      INSN_RS2(TCG_REG_G0));

            tcg_out_nop(s);

        }

        break;

    case INDEX_op_jmp:

        fprintf(stderr, "unimplemented jmp\n");

        break;

    case INDEX_op_br:

        fprintf(stderr, "unimplemented br\n");

        break;

    case INDEX_op_movi_i32:

        tcg_out_movi(s, TCG_TYPE_I32, args[0], (uint32_t)args[1]);

        break;



#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

#define OP_32_64(x)                             \

        glue(glue(case INDEX_op_, x), _i32:)    \

        glue(glue(case INDEX_op_, x), _i64:)

#else

#define OP_32_64(x)                             \

        glue(glue(case INDEX_op_, x), _i32:)

#endif

        OP_32_64(ld8u);

        tcg_out_ldst(s, args[0], args[1], args[2], LDUB);

        break;

        OP_32_64(ld8s);

        tcg_out_ldst(s, args[0], args[1], args[2], LDSB);

        break;

        OP_32_64(ld16u);

        tcg_out_ldst(s, args[0], args[1], args[2], LDUH);

        break;

        OP_32_64(ld16s);

        tcg_out_ldst(s, args[0], args[1], args[2], LDSH);

        break;

    case INDEX_op_ld_i32:

#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

    case INDEX_op_ld32u_i64:

#endif

        tcg_out_ldst(s, args[0], args[1], args[2], LDUW);

        break;

        OP_32_64(st8);

        tcg_out_ldst(s, args[0], args[1], args[2], STB);

        break;

        OP_32_64(st16);

        tcg_out_ldst(s, args[0], args[1], args[2], STH);

        break;

    case INDEX_op_st_i32:

#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

    case INDEX_op_st32_i64:

#endif

        tcg_out_ldst(s, args[0], args[1], args[2], STW);

        break;

        OP_32_64(add);

        c = ARITH_ADD;

        goto gen_arith32;

        OP_32_64(sub);

        c = ARITH_SUB;

        goto gen_arith32;

        OP_32_64(and);

        c = ARITH_AND;

        goto gen_arith32;

        OP_32_64(or);

        c = ARITH_OR;

        goto gen_arith32;

        OP_32_64(xor);

        c = ARITH_XOR;

        goto gen_arith32;

    case INDEX_op_shl_i32:

        c = SHIFT_SLL;

        goto gen_arith32;

    case INDEX_op_shr_i32:

        c = SHIFT_SRL;

        goto gen_arith32;

    case INDEX_op_sar_i32:

        c = SHIFT_SRA;

        goto gen_arith32;

    case INDEX_op_mul_i32:

        c = ARITH_UMUL;

        goto gen_arith32;

    case INDEX_op_div2_i32:

#if defined(__sparc_v9__) || defined(__sparc_v8plus__)

        c = ARITH_SDIVX;

        goto gen_arith32;

#else

        tcg_out_sety(s, 0);

        c = ARITH_SDIV;

        goto gen_arith32;

#endif

    case INDEX_op_divu2_i32:

#if defined(__sparc_v9__) || defined(__sparc_v8plus__)

        c = ARITH_UDIVX;

        goto gen_arith32;

#else

        tcg_out_sety(s, 0);

        c = ARITH_UDIV;

        goto gen_arith32;

#endif



    case INDEX_op_brcond_i32:

        fprintf(stderr, "unimplemented brcond\n");

        break;



    case INDEX_op_qemu_ld8u:

        fprintf(stderr, "unimplemented qld\n");

        break;

    case INDEX_op_qemu_ld8s:

        fprintf(stderr, "unimplemented qld\n");

        break;

    case INDEX_op_qemu_ld16u:

        fprintf(stderr, "unimplemented qld\n");

        break;

    case INDEX_op_qemu_ld16s:

        fprintf(stderr, "unimplemented qld\n");

        break;

    case INDEX_op_qemu_ld32u:

        fprintf(stderr, "unimplemented qld\n");

        break;

    case INDEX_op_qemu_ld32s:

        fprintf(stderr, "unimplemented qld\n");

        break;

    case INDEX_op_qemu_st8:

        fprintf(stderr, "unimplemented qst\n");

        break;

    case INDEX_op_qemu_st16:

        fprintf(stderr, "unimplemented qst\n");

        break;

    case INDEX_op_qemu_st32:

        fprintf(stderr, "unimplemented qst\n");

        break;



#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

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

        goto gen_arith32;

    case INDEX_op_shr_i64:

        c = SHIFT_SRLX;

        goto gen_arith32;

    case INDEX_op_sar_i64:

        c = SHIFT_SRAX;

        goto gen_arith32;

    case INDEX_op_mul_i64:

        c = ARITH_MULX;

        goto gen_arith32;

    case INDEX_op_div2_i64:

        c = ARITH_SDIVX;

        goto gen_arith32;

    case INDEX_op_divu2_i64:

        c = ARITH_UDIVX;

        goto gen_arith32;



    case INDEX_op_brcond_i64:

        fprintf(stderr, "unimplemented brcond\n");

        break;

    case INDEX_op_qemu_ld64:

        fprintf(stderr, "unimplemented qld\n");

        break;

    case INDEX_op_qemu_st64:

        fprintf(stderr, "unimplemented qst\n");

        break;



#endif

    gen_arith32:

        if (const_args[2]) {

            tcg_out_arithi(s, args[0], args[1], args[2], c);

        } else {

            tcg_out_arith(s, args[0], args[1], args[2], c);

        }

        break;



    default:

        fprintf(stderr, "unknown opcode 0x%x\n", opc);

        tcg_abort();

    }

}
