static void disas_arm_insn(CPUARMState * env, DisasContext *s)

{

    unsigned int cond, insn, val, op1, i, shift, rm, rs, rn, rd, sh;

    TCGv tmp;

    TCGv tmp2;

    TCGv tmp3;

    TCGv addr;

    TCGv_i64 tmp64;



    insn = arm_ldl_code(env, s->pc, s->bswap_code);

    s->pc += 4;



    /* M variants do not implement ARM mode.  */

    if (IS_M(env))

        goto illegal_op;

    cond = insn >> 28;

    if (cond == 0xf){

        /* In ARMv3 and v4 the NV condition is UNPREDICTABLE; we

         * choose to UNDEF. In ARMv5 and above the space is used

         * for miscellaneous unconditional instructions.

         */

        ARCH(5);



        /* Unconditional instructions.  */

        if (((insn >> 25) & 7) == 1) {

            /* NEON Data processing.  */

            if (!arm_feature(env, ARM_FEATURE_NEON))

                goto illegal_op;



            if (disas_neon_data_insn(env, s, insn))

                goto illegal_op;


        }

        if ((insn & 0x0f100000) == 0x04000000) {

            /* NEON load/store.  */

            if (!arm_feature(env, ARM_FEATURE_NEON))

                goto illegal_op;



            if (disas_neon_ls_insn(env, s, insn))

                goto illegal_op;


        }

        if (((insn & 0x0f30f000) == 0x0510f000) ||

            ((insn & 0x0f30f010) == 0x0710f000)) {

            if ((insn & (1 << 22)) == 0) {

                /* PLDW; v7MP */

                if (!arm_feature(env, ARM_FEATURE_V7MP)) {

                    goto illegal_op;

                }

            }

            /* Otherwise PLD; v5TE+ */

            ARCH(5TE);


        }

        if (((insn & 0x0f70f000) == 0x0450f000) ||

            ((insn & 0x0f70f010) == 0x0650f000)) {

            ARCH(7);

            return; /* PLI; V7 */

        }

        if (((insn & 0x0f700000) == 0x04100000) ||

            ((insn & 0x0f700010) == 0x06100000)) {

            if (!arm_feature(env, ARM_FEATURE_V7MP)) {

                goto illegal_op;

            }

            return; /* v7MP: Unallocated memory hint: must NOP */

        }



        if ((insn & 0x0ffffdff) == 0x01010000) {

            ARCH(6);

            /* setend */

            if (((insn >> 9) & 1) != s->bswap_code) {

                /* Dynamic endianness switching not implemented. */

                goto illegal_op;

            }


        } else if ((insn & 0x0fffff00) == 0x057ff000) {

            switch ((insn >> 4) & 0xf) {

            case 1: /* clrex */

                ARCH(6K);

                gen_clrex(s);


            case 4: /* dsb */

            case 5: /* dmb */

            case 6: /* isb */

                ARCH(7);

                /* We don't emulate caches so these are a no-op.  */


            default:

                goto illegal_op;

            }

        } else if ((insn & 0x0e5fffe0) == 0x084d0500) {

            /* srs */

            if (IS_USER(s)) {

                goto illegal_op;

            }

            ARCH(6);

            gen_srs(s, (insn & 0x1f), (insn >> 23) & 3, insn & (1 << 21));


        } else if ((insn & 0x0e50ffe0) == 0x08100a00) {

            /* rfe */

            int32_t offset;

            if (IS_USER(s))

                goto illegal_op;

            ARCH(6);

            rn = (insn >> 16) & 0xf;

            addr = load_reg(s, rn);

            i = (insn >> 23) & 3;

            switch (i) {

            case 0: offset = -4; break; /* DA */

            case 1: offset = 0; break; /* IA */

            case 2: offset = -8; break; /* DB */

            case 3: offset = 4; break; /* IB */

            default: abort();

            }

            if (offset)

                tcg_gen_addi_i32(addr, addr, offset);

            /* Load PC into tmp and CPSR into tmp2.  */

            tmp = gen_ld32(addr, 0);

            tcg_gen_addi_i32(addr, addr, 4);

            tmp2 = gen_ld32(addr, 0);

            if (insn & (1 << 21)) {

                /* Base writeback.  */

                switch (i) {

                case 0: offset = -8; break;

                case 1: offset = 4; break;

                case 2: offset = -4; break;

                case 3: offset = 0; break;

                default: abort();

                }

                if (offset)

                    tcg_gen_addi_i32(addr, addr, offset);

                store_reg(s, rn, addr);

            } else {

                tcg_temp_free_i32(addr);

            }

            gen_rfe(s, tmp, tmp2);


        } else if ((insn & 0x0e000000) == 0x0a000000) {

            /* branch link and change to thumb (blx <offset>) */

            int32_t offset;



            val = (uint32_t)s->pc;

            tmp = tcg_temp_new_i32();

            tcg_gen_movi_i32(tmp, val);

            store_reg(s, 14, tmp);

            /* Sign-extend the 24-bit offset */

            offset = (((int32_t)insn) << 8) >> 8;

            /* offset * 4 + bit24 * 2 + (thumb bit) */

            val += (offset << 2) | ((insn >> 23) & 2) | 1;

            /* pipeline offset */

            val += 4;

            /* protected by ARCH(5); above, near the start of uncond block */

            gen_bx_im(s, val);


        } else if ((insn & 0x0e000f00) == 0x0c000100) {

            if (arm_feature(env, ARM_FEATURE_IWMMXT)) {

                /* iWMMXt register transfer.  */

                if (env->cp15.c15_cpar & (1 << 1))

                    if (!disas_iwmmxt_insn(env, s, insn))


            }

        } else if ((insn & 0x0fe00000) == 0x0c400000) {

            /* Coprocessor double register transfer.  */

            ARCH(5TE);

        } else if ((insn & 0x0f000010) == 0x0e000010) {

            /* Additional coprocessor register transfer.  */

        } else if ((insn & 0x0ff10020) == 0x01000000) {

            uint32_t mask;

            uint32_t val;

            /* cps (privileged) */

            if (IS_USER(s))


            mask = val = 0;

            if (insn & (1 << 19)) {

                if (insn & (1 << 8))

                    mask |= CPSR_A;

                if (insn & (1 << 7))

                    mask |= CPSR_I;

                if (insn & (1 << 6))

                    mask |= CPSR_F;

                if (insn & (1 << 18))

                    val |= mask;

            }

            if (insn & (1 << 17)) {

                mask |= CPSR_M;

                val |= (insn & 0x1f);

            }

            if (mask) {

                gen_set_psr_im(s, mask, 0, val);

            }


        }

        goto illegal_op;

    }

    if (cond != 0xe) {

        /* if not always execute, we generate a conditional jump to

           next instruction */

        s->condlabel = gen_new_label();

        gen_test_cc(cond ^ 1, s->condlabel);

        s->condjmp = 1;

    }

    if ((insn & 0x0f900000) == 0x03000000) {

        if ((insn & (1 << 21)) == 0) {

            ARCH(6T2);

            rd = (insn >> 12) & 0xf;

            val = ((insn >> 4) & 0xf000) | (insn & 0xfff);

            if ((insn & (1 << 22)) == 0) {

                /* MOVW */

                tmp = tcg_temp_new_i32();

                tcg_gen_movi_i32(tmp, val);

            } else {

                /* MOVT */

                tmp = load_reg(s, rd);

                tcg_gen_ext16u_i32(tmp, tmp);

                tcg_gen_ori_i32(tmp, tmp, val << 16);

            }

            store_reg(s, rd, tmp);

        } else {

            if (((insn >> 12) & 0xf) != 0xf)

                goto illegal_op;

            if (((insn >> 16) & 0xf) == 0) {

                gen_nop_hint(s, insn & 0xff);

            } else {

                /* CPSR = immediate */

                val = insn & 0xff;

                shift = ((insn >> 8) & 0xf) * 2;

                if (shift)

                    val = (val >> shift) | (val << (32 - shift));

                i = ((insn & (1 << 22)) != 0);

                if (gen_set_psr_im(s, msr_mask(env, s, (insn >> 16) & 0xf, i), i, val))

                    goto illegal_op;

            }

        }

    } else if ((insn & 0x0f900000) == 0x01000000

               && (insn & 0x00000090) != 0x00000090) {

        /* miscellaneous instructions */

        op1 = (insn >> 21) & 3;

        sh = (insn >> 4) & 0xf;

        rm = insn & 0xf;

        switch (sh) {

        case 0x0: /* move program status register */

            if (op1 & 1) {

                /* PSR = reg */

                tmp = load_reg(s, rm);

                i = ((op1 & 2) != 0);

                if (gen_set_psr(s, msr_mask(env, s, (insn >> 16) & 0xf, i), i, tmp))

                    goto illegal_op;

            } else {

                /* reg = PSR */

                rd = (insn >> 12) & 0xf;

                if (op1 & 2) {

                    if (IS_USER(s))

                        goto illegal_op;

                    tmp = load_cpu_field(spsr);

                } else {

                    tmp = tcg_temp_new_i32();

                    gen_helper_cpsr_read(tmp, cpu_env);

                }

                store_reg(s, rd, tmp);

            }

            break;

        case 0x1:

            if (op1 == 1) {

                /* branch/exchange thumb (bx).  */

                ARCH(4T);

                tmp = load_reg(s, rm);

                gen_bx(s, tmp);

            } else if (op1 == 3) {

                /* clz */

                ARCH(5);

                rd = (insn >> 12) & 0xf;

                tmp = load_reg(s, rm);

                gen_helper_clz(tmp, tmp);

                store_reg(s, rd, tmp);

            } else {

                goto illegal_op;

            }

            break;

        case 0x2:

            if (op1 == 1) {

                ARCH(5J); /* bxj */

                /* Trivial implementation equivalent to bx.  */

                tmp = load_reg(s, rm);

                gen_bx(s, tmp);

            } else {

                goto illegal_op;

            }

            break;

        case 0x3:

            if (op1 != 1)

              goto illegal_op;



            ARCH(5);

            /* branch link/exchange thumb (blx) */

            tmp = load_reg(s, rm);

            tmp2 = tcg_temp_new_i32();

            tcg_gen_movi_i32(tmp2, s->pc);

            store_reg(s, 14, tmp2);

            gen_bx(s, tmp);

            break;

        case 0x5: /* saturating add/subtract */

            ARCH(5TE);

            rd = (insn >> 12) & 0xf;

            rn = (insn >> 16) & 0xf;

            tmp = load_reg(s, rm);

            tmp2 = load_reg(s, rn);

            if (op1 & 2)

                gen_helper_double_saturate(tmp2, cpu_env, tmp2);

            if (op1 & 1)

                gen_helper_sub_saturate(tmp, cpu_env, tmp, tmp2);

            else

                gen_helper_add_saturate(tmp, cpu_env, tmp, tmp2);

            tcg_temp_free_i32(tmp2);

            store_reg(s, rd, tmp);

            break;

        case 7:

            /* SMC instruction (op1 == 3)

               and undefined instructions (op1 == 0 || op1 == 2)

               will trap */

            if (op1 != 1) {

                goto illegal_op;

            }

            /* bkpt */

            ARCH(5);

            gen_exception_insn(s, 4, EXCP_BKPT);

            break;

        case 0x8: /* signed multiply */

        case 0xa:

        case 0xc:

        case 0xe:

            ARCH(5TE);

            rs = (insn >> 8) & 0xf;

            rn = (insn >> 12) & 0xf;

            rd = (insn >> 16) & 0xf;

            if (op1 == 1) {

                /* (32 * 16) >> 16 */

                tmp = load_reg(s, rm);

                tmp2 = load_reg(s, rs);

                if (sh & 4)

                    tcg_gen_sari_i32(tmp2, tmp2, 16);

                else

                    gen_sxth(tmp2);

                tmp64 = gen_muls_i64_i32(tmp, tmp2);

                tcg_gen_shri_i64(tmp64, tmp64, 16);

                tmp = tcg_temp_new_i32();

                tcg_gen_trunc_i64_i32(tmp, tmp64);

                tcg_temp_free_i64(tmp64);

                if ((sh & 2) == 0) {

                    tmp2 = load_reg(s, rn);

                    gen_helper_add_setq(tmp, cpu_env, tmp, tmp2);

                    tcg_temp_free_i32(tmp2);

                }

                store_reg(s, rd, tmp);

            } else {

                /* 16 * 16 */

                tmp = load_reg(s, rm);

                tmp2 = load_reg(s, rs);

                gen_mulxy(tmp, tmp2, sh & 2, sh & 4);

                tcg_temp_free_i32(tmp2);

                if (op1 == 2) {

                    tmp64 = tcg_temp_new_i64();

                    tcg_gen_ext_i32_i64(tmp64, tmp);

                    tcg_temp_free_i32(tmp);

                    gen_addq(s, tmp64, rn, rd);

                    gen_storeq_reg(s, rn, rd, tmp64);

                    tcg_temp_free_i64(tmp64);

                } else {

                    if (op1 == 0) {

                        tmp2 = load_reg(s, rn);

                        gen_helper_add_setq(tmp, cpu_env, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                    }

                    store_reg(s, rd, tmp);

                }

            }

            break;

        default:

            goto illegal_op;

        }

    } else if (((insn & 0x0e000000) == 0 &&

                (insn & 0x00000090) != 0x90) ||

               ((insn & 0x0e000000) == (1 << 25))) {

        int set_cc, logic_cc, shiftop;



        op1 = (insn >> 21) & 0xf;

        set_cc = (insn >> 20) & 1;

        logic_cc = table_logic_cc[op1] & set_cc;



        /* data processing instruction */

        if (insn & (1 << 25)) {

            /* immediate operand */

            val = insn & 0xff;

            shift = ((insn >> 8) & 0xf) * 2;

            if (shift) {

                val = (val >> shift) | (val << (32 - shift));

            }

            tmp2 = tcg_temp_new_i32();

            tcg_gen_movi_i32(tmp2, val);

            if (logic_cc && shift) {

                gen_set_CF_bit31(tmp2);

            }

        } else {

            /* register */

            rm = (insn) & 0xf;

            tmp2 = load_reg(s, rm);

            shiftop = (insn >> 5) & 3;

            if (!(insn & (1 << 4))) {

                shift = (insn >> 7) & 0x1f;

                gen_arm_shift_im(tmp2, shiftop, shift, logic_cc);

            } else {

                rs = (insn >> 8) & 0xf;

                tmp = load_reg(s, rs);

                gen_arm_shift_reg(tmp2, shiftop, tmp, logic_cc);

            }

        }

        if (op1 != 0x0f && op1 != 0x0d) {

            rn = (insn >> 16) & 0xf;

            tmp = load_reg(s, rn);

        } else {

            TCGV_UNUSED(tmp);

        }

        rd = (insn >> 12) & 0xf;

        switch(op1) {

        case 0x00:

            tcg_gen_and_i32(tmp, tmp, tmp2);

            if (logic_cc) {

                gen_logic_CC(tmp);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x01:

            tcg_gen_xor_i32(tmp, tmp, tmp2);

            if (logic_cc) {

                gen_logic_CC(tmp);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x02:

            if (set_cc && rd == 15) {

                /* SUBS r15, ... is used for exception return.  */

                if (IS_USER(s)) {

                    goto illegal_op;

                }

                gen_sub_CC(tmp, tmp, tmp2);

                gen_exception_return(s, tmp);

            } else {

                if (set_cc) {

                    gen_sub_CC(tmp, tmp, tmp2);

                } else {

                    tcg_gen_sub_i32(tmp, tmp, tmp2);

                }

                store_reg_bx(env, s, rd, tmp);

            }

            break;

        case 0x03:

            if (set_cc) {

                gen_sub_CC(tmp, tmp2, tmp);

            } else {

                tcg_gen_sub_i32(tmp, tmp2, tmp);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x04:

            if (set_cc) {

                gen_add_CC(tmp, tmp, tmp2);

            } else {

                tcg_gen_add_i32(tmp, tmp, tmp2);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x05:

            if (set_cc) {

                gen_adc_CC(tmp, tmp, tmp2);

            } else {

                gen_add_carry(tmp, tmp, tmp2);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x06:

            if (set_cc) {

                gen_sbc_CC(tmp, tmp, tmp2);

            } else {

                gen_sub_carry(tmp, tmp, tmp2);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x07:

            if (set_cc) {

                gen_sbc_CC(tmp, tmp2, tmp);

            } else {

                gen_sub_carry(tmp, tmp2, tmp);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x08:

            if (set_cc) {

                tcg_gen_and_i32(tmp, tmp, tmp2);

                gen_logic_CC(tmp);

            }

            tcg_temp_free_i32(tmp);

            break;

        case 0x09:

            if (set_cc) {

                tcg_gen_xor_i32(tmp, tmp, tmp2);

                gen_logic_CC(tmp);

            }

            tcg_temp_free_i32(tmp);

            break;

        case 0x0a:

            if (set_cc) {

                gen_sub_CC(tmp, tmp, tmp2);

            }

            tcg_temp_free_i32(tmp);

            break;

        case 0x0b:

            if (set_cc) {

                gen_add_CC(tmp, tmp, tmp2);

            }

            tcg_temp_free_i32(tmp);

            break;

        case 0x0c:

            tcg_gen_or_i32(tmp, tmp, tmp2);

            if (logic_cc) {

                gen_logic_CC(tmp);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        case 0x0d:

            if (logic_cc && rd == 15) {

                /* MOVS r15, ... is used for exception return.  */

                if (IS_USER(s)) {

                    goto illegal_op;

                }

                gen_exception_return(s, tmp2);

            } else {

                if (logic_cc) {

                    gen_logic_CC(tmp2);

                }

                store_reg_bx(env, s, rd, tmp2);

            }

            break;

        case 0x0e:

            tcg_gen_andc_i32(tmp, tmp, tmp2);

            if (logic_cc) {

                gen_logic_CC(tmp);

            }

            store_reg_bx(env, s, rd, tmp);

            break;

        default:

        case 0x0f:

            tcg_gen_not_i32(tmp2, tmp2);

            if (logic_cc) {

                gen_logic_CC(tmp2);

            }

            store_reg_bx(env, s, rd, tmp2);

            break;

        }

        if (op1 != 0x0f && op1 != 0x0d) {

            tcg_temp_free_i32(tmp2);

        }

    } else {

        /* other instructions */

        op1 = (insn >> 24) & 0xf;

        switch(op1) {

        case 0x0:

        case 0x1:

            /* multiplies, extra load/stores */

            sh = (insn >> 5) & 3;

            if (sh == 0) {

                if (op1 == 0x0) {

                    rd = (insn >> 16) & 0xf;

                    rn = (insn >> 12) & 0xf;

                    rs = (insn >> 8) & 0xf;

                    rm = (insn) & 0xf;

                    op1 = (insn >> 20) & 0xf;

                    switch (op1) {

                    case 0: case 1: case 2: case 3: case 6:

                        /* 32 bit mul */

                        tmp = load_reg(s, rs);

                        tmp2 = load_reg(s, rm);

                        tcg_gen_mul_i32(tmp, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        if (insn & (1 << 22)) {

                            /* Subtract (mls) */

                            ARCH(6T2);

                            tmp2 = load_reg(s, rn);

                            tcg_gen_sub_i32(tmp, tmp2, tmp);

                            tcg_temp_free_i32(tmp2);

                        } else if (insn & (1 << 21)) {

                            /* Add */

                            tmp2 = load_reg(s, rn);

                            tcg_gen_add_i32(tmp, tmp, tmp2);

                            tcg_temp_free_i32(tmp2);

                        }

                        if (insn & (1 << 20))

                            gen_logic_CC(tmp);

                        store_reg(s, rd, tmp);

                        break;

                    case 4:

                        /* 64 bit mul double accumulate (UMAAL) */

                        ARCH(6);

                        tmp = load_reg(s, rs);

                        tmp2 = load_reg(s, rm);

                        tmp64 = gen_mulu_i64_i32(tmp, tmp2);

                        gen_addq_lo(s, tmp64, rn);

                        gen_addq_lo(s, tmp64, rd);

                        gen_storeq_reg(s, rn, rd, tmp64);

                        tcg_temp_free_i64(tmp64);

                        break;

                    case 8: case 9: case 10: case 11:

                    case 12: case 13: case 14: case 15:

                        /* 64 bit mul: UMULL, UMLAL, SMULL, SMLAL. */

                        tmp = load_reg(s, rs);

                        tmp2 = load_reg(s, rm);

                        if (insn & (1 << 22)) {

                            tcg_gen_muls2_i32(tmp, tmp2, tmp, tmp2);

                        } else {

                            tcg_gen_mulu2_i32(tmp, tmp2, tmp, tmp2);

                        }

                        if (insn & (1 << 21)) { /* mult accumulate */

                            TCGv al = load_reg(s, rn);

                            TCGv ah = load_reg(s, rd);

                            tcg_gen_add2_i32(tmp, tmp2, tmp, tmp2, al, ah);

                            tcg_temp_free(al);

                            tcg_temp_free(ah);

                        }

                        if (insn & (1 << 20)) {

                            gen_logicq_cc(tmp, tmp2);

                        }

                        store_reg(s, rn, tmp);

                        store_reg(s, rd, tmp2);

                        break;

                    default:

                        goto illegal_op;

                    }

                } else {

                    rn = (insn >> 16) & 0xf;

                    rd = (insn >> 12) & 0xf;

                    if (insn & (1 << 23)) {

                        /* load/store exclusive */

                        op1 = (insn >> 21) & 0x3;

                        if (op1)

                            ARCH(6K);

                        else

                            ARCH(6);

                        addr = tcg_temp_local_new_i32();

                        load_reg_var(s, addr, rn);

                        if (insn & (1 << 20)) {

                            switch (op1) {

                            case 0: /* ldrex */

                                gen_load_exclusive(s, rd, 15, addr, 2);

                                break;

                            case 1: /* ldrexd */

                                gen_load_exclusive(s, rd, rd + 1, addr, 3);

                                break;

                            case 2: /* ldrexb */

                                gen_load_exclusive(s, rd, 15, addr, 0);

                                break;

                            case 3: /* ldrexh */

                                gen_load_exclusive(s, rd, 15, addr, 1);

                                break;

                            default:

                                abort();

                            }

                        } else {

                            rm = insn & 0xf;

                            switch (op1) {

                            case 0:  /*  strex */

                                gen_store_exclusive(s, rd, rm, 15, addr, 2);

                                break;

                            case 1: /*  strexd */

                                gen_store_exclusive(s, rd, rm, rm + 1, addr, 3);

                                break;

                            case 2: /*  strexb */

                                gen_store_exclusive(s, rd, rm, 15, addr, 0);

                                break;

                            case 3: /* strexh */

                                gen_store_exclusive(s, rd, rm, 15, addr, 1);

                                break;

                            default:

                                abort();

                            }

                        }

                        tcg_temp_free(addr);

                    } else {

                        /* SWP instruction */

                        rm = (insn) & 0xf;



                        /* ??? This is not really atomic.  However we know

                           we never have multiple CPUs running in parallel,

                           so it is good enough.  */

                        addr = load_reg(s, rn);

                        tmp = load_reg(s, rm);

                        if (insn & (1 << 22)) {

                            tmp2 = gen_ld8u(addr, IS_USER(s));

                            gen_st8(tmp, addr, IS_USER(s));

                        } else {

                            tmp2 = gen_ld32(addr, IS_USER(s));

                            gen_st32(tmp, addr, IS_USER(s));

                        }

                        tcg_temp_free_i32(addr);

                        store_reg(s, rd, tmp2);

                    }

                }

            } else {

                int address_offset;

                int load;

                /* Misc load/store */

                rn = (insn >> 16) & 0xf;

                rd = (insn >> 12) & 0xf;

                addr = load_reg(s, rn);

                if (insn & (1 << 24))

                    gen_add_datah_offset(s, insn, 0, addr);

                address_offset = 0;

                if (insn & (1 << 20)) {

                    /* load */

                    switch(sh) {

                    case 1:

                        tmp = gen_ld16u(addr, IS_USER(s));

                        break;

                    case 2:

                        tmp = gen_ld8s(addr, IS_USER(s));

                        break;

                    default:

                    case 3:

                        tmp = gen_ld16s(addr, IS_USER(s));

                        break;

                    }

                    load = 1;

                } else if (sh & 2) {

                    ARCH(5TE);

                    /* doubleword */

                    if (sh & 1) {

                        /* store */

                        tmp = load_reg(s, rd);

                        gen_st32(tmp, addr, IS_USER(s));

                        tcg_gen_addi_i32(addr, addr, 4);

                        tmp = load_reg(s, rd + 1);

                        gen_st32(tmp, addr, IS_USER(s));

                        load = 0;

                    } else {

                        /* load */

                        tmp = gen_ld32(addr, IS_USER(s));

                        store_reg(s, rd, tmp);

                        tcg_gen_addi_i32(addr, addr, 4);

                        tmp = gen_ld32(addr, IS_USER(s));

                        rd++;

                        load = 1;

                    }

                    address_offset = -4;

                } else {

                    /* store */

                    tmp = load_reg(s, rd);

                    gen_st16(tmp, addr, IS_USER(s));

                    load = 0;

                }

                /* Perform base writeback before the loaded value to

                   ensure correct behavior with overlapping index registers.

                   ldrd with base writeback is is undefined if the

                   destination and index registers overlap.  */

                if (!(insn & (1 << 24))) {

                    gen_add_datah_offset(s, insn, address_offset, addr);

                    store_reg(s, rn, addr);

                } else if (insn & (1 << 21)) {

                    if (address_offset)

                        tcg_gen_addi_i32(addr, addr, address_offset);

                    store_reg(s, rn, addr);

                } else {

                    tcg_temp_free_i32(addr);

                }

                if (load) {

                    /* Complete the load.  */

                    store_reg(s, rd, tmp);

                }

            }

            break;

        case 0x4:

        case 0x5:

            goto do_ldst;

        case 0x6:

        case 0x7:

            if (insn & (1 << 4)) {

                ARCH(6);

                /* Armv6 Media instructions.  */

                rm = insn & 0xf;

                rn = (insn >> 16) & 0xf;

                rd = (insn >> 12) & 0xf;

                rs = (insn >> 8) & 0xf;

                switch ((insn >> 23) & 3) {

                case 0: /* Parallel add/subtract.  */

                    op1 = (insn >> 20) & 7;

                    tmp = load_reg(s, rn);

                    tmp2 = load_reg(s, rm);

                    sh = (insn >> 5) & 7;

                    if ((op1 & 3) == 0 || sh == 5 || sh == 6)

                        goto illegal_op;

                    gen_arm_parallel_addsub(op1, sh, tmp, tmp2);

                    tcg_temp_free_i32(tmp2);

                    store_reg(s, rd, tmp);

                    break;

                case 1:

                    if ((insn & 0x00700020) == 0) {

                        /* Halfword pack.  */

                        tmp = load_reg(s, rn);

                        tmp2 = load_reg(s, rm);

                        shift = (insn >> 7) & 0x1f;

                        if (insn & (1 << 6)) {

                            /* pkhtb */

                            if (shift == 0)

                                shift = 31;

                            tcg_gen_sari_i32(tmp2, tmp2, shift);

                            tcg_gen_andi_i32(tmp, tmp, 0xffff0000);

                            tcg_gen_ext16u_i32(tmp2, tmp2);

                        } else {

                            /* pkhbt */

                            if (shift)

                                tcg_gen_shli_i32(tmp2, tmp2, shift);

                            tcg_gen_ext16u_i32(tmp, tmp);

                            tcg_gen_andi_i32(tmp2, tmp2, 0xffff0000);

                        }

                        tcg_gen_or_i32(tmp, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        store_reg(s, rd, tmp);

                    } else if ((insn & 0x00200020) == 0x00200000) {

                        /* [us]sat */

                        tmp = load_reg(s, rm);

                        shift = (insn >> 7) & 0x1f;

                        if (insn & (1 << 6)) {

                            if (shift == 0)

                                shift = 31;

                            tcg_gen_sari_i32(tmp, tmp, shift);

                        } else {

                            tcg_gen_shli_i32(tmp, tmp, shift);

                        }

                        sh = (insn >> 16) & 0x1f;

                        tmp2 = tcg_const_i32(sh);

                        if (insn & (1 << 22))

                          gen_helper_usat(tmp, cpu_env, tmp, tmp2);

                        else

                          gen_helper_ssat(tmp, cpu_env, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        store_reg(s, rd, tmp);

                    } else if ((insn & 0x00300fe0) == 0x00200f20) {

                        /* [us]sat16 */

                        tmp = load_reg(s, rm);

                        sh = (insn >> 16) & 0x1f;

                        tmp2 = tcg_const_i32(sh);

                        if (insn & (1 << 22))

                          gen_helper_usat16(tmp, cpu_env, tmp, tmp2);

                        else

                          gen_helper_ssat16(tmp, cpu_env, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        store_reg(s, rd, tmp);

                    } else if ((insn & 0x00700fe0) == 0x00000fa0) {

                        /* Select bytes.  */

                        tmp = load_reg(s, rn);

                        tmp2 = load_reg(s, rm);

                        tmp3 = tcg_temp_new_i32();

                        tcg_gen_ld_i32(tmp3, cpu_env, offsetof(CPUARMState, GE));

                        gen_helper_sel_flags(tmp, tmp3, tmp, tmp2);

                        tcg_temp_free_i32(tmp3);

                        tcg_temp_free_i32(tmp2);

                        store_reg(s, rd, tmp);

                    } else if ((insn & 0x000003e0) == 0x00000060) {

                        tmp = load_reg(s, rm);

                        shift = (insn >> 10) & 3;

                        /* ??? In many cases it's not necessary to do a

                           rotate, a shift is sufficient.  */

                        if (shift != 0)

                            tcg_gen_rotri_i32(tmp, tmp, shift * 8);

                        op1 = (insn >> 20) & 7;

                        switch (op1) {

                        case 0: gen_sxtb16(tmp);  break;

                        case 2: gen_sxtb(tmp);    break;

                        case 3: gen_sxth(tmp);    break;

                        case 4: gen_uxtb16(tmp);  break;

                        case 6: gen_uxtb(tmp);    break;

                        case 7: gen_uxth(tmp);    break;

                        default: goto illegal_op;

                        }

                        if (rn != 15) {

                            tmp2 = load_reg(s, rn);

                            if ((op1 & 3) == 0) {

                                gen_add16(tmp, tmp2);

                            } else {

                                tcg_gen_add_i32(tmp, tmp, tmp2);

                                tcg_temp_free_i32(tmp2);

                            }

                        }

                        store_reg(s, rd, tmp);

                    } else if ((insn & 0x003f0f60) == 0x003f0f20) {

                        /* rev */

                        tmp = load_reg(s, rm);

                        if (insn & (1 << 22)) {

                            if (insn & (1 << 7)) {

                                gen_revsh(tmp);

                            } else {

                                ARCH(6T2);

                                gen_helper_rbit(tmp, tmp);

                            }

                        } else {

                            if (insn & (1 << 7))

                                gen_rev16(tmp);

                            else

                                tcg_gen_bswap32_i32(tmp, tmp);

                        }

                        store_reg(s, rd, tmp);

                    } else {

                        goto illegal_op;

                    }

                    break;

                case 2: /* Multiplies (Type 3).  */

                    switch ((insn >> 20) & 0x7) {

                    case 5:

                        if (((insn >> 6) ^ (insn >> 7)) & 1) {

                            /* op2 not 00x or 11x : UNDEF */

                            goto illegal_op;

                        }

                        /* Signed multiply most significant [accumulate].

                           (SMMUL, SMMLA, SMMLS) */

                        tmp = load_reg(s, rm);

                        tmp2 = load_reg(s, rs);

                        tmp64 = gen_muls_i64_i32(tmp, tmp2);



                        if (rd != 15) {

                            tmp = load_reg(s, rd);

                            if (insn & (1 << 6)) {

                                tmp64 = gen_subq_msw(tmp64, tmp);

                            } else {

                                tmp64 = gen_addq_msw(tmp64, tmp);

                            }

                        }

                        if (insn & (1 << 5)) {

                            tcg_gen_addi_i64(tmp64, tmp64, 0x80000000u);

                        }

                        tcg_gen_shri_i64(tmp64, tmp64, 32);

                        tmp = tcg_temp_new_i32();

                        tcg_gen_trunc_i64_i32(tmp, tmp64);

                        tcg_temp_free_i64(tmp64);

                        store_reg(s, rn, tmp);

                        break;

                    case 0:

                    case 4:

                        /* SMLAD, SMUAD, SMLSD, SMUSD, SMLALD, SMLSLD */

                        if (insn & (1 << 7)) {

                            goto illegal_op;

                        }

                        tmp = load_reg(s, rm);

                        tmp2 = load_reg(s, rs);

                        if (insn & (1 << 5))

                            gen_swap_half(tmp2);

                        gen_smul_dual(tmp, tmp2);

                        if (insn & (1 << 6)) {

                            /* This subtraction cannot overflow. */

                            tcg_gen_sub_i32(tmp, tmp, tmp2);

                        } else {

                            /* This addition cannot overflow 32 bits;

                             * however it may overflow considered as a signed

                             * operation, in which case we must set the Q flag.

                             */

                            gen_helper_add_setq(tmp, cpu_env, tmp, tmp2);

                        }

                        tcg_temp_free_i32(tmp2);

                        if (insn & (1 << 22)) {

                            /* smlald, smlsld */

                            tmp64 = tcg_temp_new_i64();

                            tcg_gen_ext_i32_i64(tmp64, tmp);

                            tcg_temp_free_i32(tmp);

                            gen_addq(s, tmp64, rd, rn);

                            gen_storeq_reg(s, rd, rn, tmp64);

                            tcg_temp_free_i64(tmp64);

                        } else {

                            /* smuad, smusd, smlad, smlsd */

                            if (rd != 15)

                              {

                                tmp2 = load_reg(s, rd);

                                gen_helper_add_setq(tmp, cpu_env, tmp, tmp2);

                                tcg_temp_free_i32(tmp2);

                              }

                            store_reg(s, rn, tmp);

                        }

                        break;

                    case 1:

                    case 3:

                        /* SDIV, UDIV */

                        if (!arm_feature(env, ARM_FEATURE_ARM_DIV)) {

                            goto illegal_op;

                        }

                        if (((insn >> 5) & 7) || (rd != 15)) {

                            goto illegal_op;

                        }

                        tmp = load_reg(s, rm);

                        tmp2 = load_reg(s, rs);

                        if (insn & (1 << 21)) {

                            gen_helper_udiv(tmp, tmp, tmp2);

                        } else {

                            gen_helper_sdiv(tmp, tmp, tmp2);

                        }

                        tcg_temp_free_i32(tmp2);

                        store_reg(s, rn, tmp);

                        break;

                    default:

                        goto illegal_op;

                    }

                    break;

                case 3:

                    op1 = ((insn >> 17) & 0x38) | ((insn >> 5) & 7);

                    switch (op1) {

                    case 0: /* Unsigned sum of absolute differences.  */

                        ARCH(6);

                        tmp = load_reg(s, rm);

                        tmp2 = load_reg(s, rs);

                        gen_helper_usad8(tmp, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        if (rd != 15) {

                            tmp2 = load_reg(s, rd);

                            tcg_gen_add_i32(tmp, tmp, tmp2);

                            tcg_temp_free_i32(tmp2);

                        }

                        store_reg(s, rn, tmp);

                        break;

                    case 0x20: case 0x24: case 0x28: case 0x2c:

                        /* Bitfield insert/clear.  */

                        ARCH(6T2);

                        shift = (insn >> 7) & 0x1f;

                        i = (insn >> 16) & 0x1f;

                        i = i + 1 - shift;

                        if (rm == 15) {

                            tmp = tcg_temp_new_i32();

                            tcg_gen_movi_i32(tmp, 0);

                        } else {

                            tmp = load_reg(s, rm);

                        }

                        if (i != 32) {

                            tmp2 = load_reg(s, rd);

                            tcg_gen_deposit_i32(tmp, tmp2, tmp, shift, i);

                            tcg_temp_free_i32(tmp2);

                        }

                        store_reg(s, rd, tmp);

                        break;

                    case 0x12: case 0x16: case 0x1a: case 0x1e: /* sbfx */

                    case 0x32: case 0x36: case 0x3a: case 0x3e: /* ubfx */

                        ARCH(6T2);

                        tmp = load_reg(s, rm);

                        shift = (insn >> 7) & 0x1f;

                        i = ((insn >> 16) & 0x1f) + 1;

                        if (shift + i > 32)

                            goto illegal_op;

                        if (i < 32) {

                            if (op1 & 0x20) {

                                gen_ubfx(tmp, shift, (1u << i) - 1);

                            } else {

                                gen_sbfx(tmp, shift, i);

                            }

                        }

                        store_reg(s, rd, tmp);

                        break;

                    default:

                        goto illegal_op;

                    }

                    break;

                }

                break;

            }

        do_ldst:

            /* Check for undefined extension instructions

             * per the ARM Bible IE:

             * xxxx 0111 1111 xxxx  xxxx xxxx 1111 xxxx

             */

            sh = (0xf << 20) | (0xf << 4);

            if (op1 == 0x7 && ((insn & sh) == sh))

            {

                goto illegal_op;

            }

            /* load/store byte/word */

            rn = (insn >> 16) & 0xf;

            rd = (insn >> 12) & 0xf;

            tmp2 = load_reg(s, rn);

            i = (IS_USER(s) || (insn & 0x01200000) == 0x00200000);

            if (insn & (1 << 24))

                gen_add_data_offset(s, insn, tmp2);

            if (insn & (1 << 20)) {

                /* load */

                if (insn & (1 << 22)) {

                    tmp = gen_ld8u(tmp2, i);

                } else {

                    tmp = gen_ld32(tmp2, i);

                }

            } else {

                /* store */

                tmp = load_reg(s, rd);

                if (insn & (1 << 22))

                    gen_st8(tmp, tmp2, i);

                else

                    gen_st32(tmp, tmp2, i);

            }

            if (!(insn & (1 << 24))) {

                gen_add_data_offset(s, insn, tmp2);

                store_reg(s, rn, tmp2);

            } else if (insn & (1 << 21)) {

                store_reg(s, rn, tmp2);

            } else {

                tcg_temp_free_i32(tmp2);

            }

            if (insn & (1 << 20)) {

                /* Complete the load.  */

                store_reg_from_load(env, s, rd, tmp);

            }

            break;

        case 0x08:

        case 0x09:

            {

                int j, n, user, loaded_base;

                TCGv loaded_var;

                /* load/store multiple words */

                /* XXX: store correct base if write back */

                user = 0;

                if (insn & (1 << 22)) {

                    if (IS_USER(s))

                        goto illegal_op; /* only usable in supervisor mode */



                    if ((insn & (1 << 15)) == 0)

                        user = 1;

                }

                rn = (insn >> 16) & 0xf;

                addr = load_reg(s, rn);



                /* compute total size */

                loaded_base = 0;

                TCGV_UNUSED(loaded_var);

                n = 0;

                for(i=0;i<16;i++) {

                    if (insn & (1 << i))

                        n++;

                }

                /* XXX: test invalid n == 0 case ? */

                if (insn & (1 << 23)) {

                    if (insn & (1 << 24)) {

                        /* pre increment */

                        tcg_gen_addi_i32(addr, addr, 4);

                    } else {

                        /* post increment */

                    }

                } else {

                    if (insn & (1 << 24)) {

                        /* pre decrement */

                        tcg_gen_addi_i32(addr, addr, -(n * 4));

                    } else {

                        /* post decrement */

                        if (n != 1)

                        tcg_gen_addi_i32(addr, addr, -((n - 1) * 4));

                    }

                }

                j = 0;

                for(i=0;i<16;i++) {

                    if (insn & (1 << i)) {

                        if (insn & (1 << 20)) {

                            /* load */

                            tmp = gen_ld32(addr, IS_USER(s));

                            if (user) {

                                tmp2 = tcg_const_i32(i);

                                gen_helper_set_user_reg(cpu_env, tmp2, tmp);

                                tcg_temp_free_i32(tmp2);

                                tcg_temp_free_i32(tmp);

                            } else if (i == rn) {

                                loaded_var = tmp;

                                loaded_base = 1;

                            } else {

                                store_reg_from_load(env, s, i, tmp);

                            }

                        } else {

                            /* store */

                            if (i == 15) {

                                /* special case: r15 = PC + 8 */

                                val = (long)s->pc + 4;

                                tmp = tcg_temp_new_i32();

                                tcg_gen_movi_i32(tmp, val);

                            } else if (user) {

                                tmp = tcg_temp_new_i32();

                                tmp2 = tcg_const_i32(i);

                                gen_helper_get_user_reg(tmp, cpu_env, tmp2);

                                tcg_temp_free_i32(tmp2);

                            } else {

                                tmp = load_reg(s, i);

                            }

                            gen_st32(tmp, addr, IS_USER(s));

                        }

                        j++;

                        /* no need to add after the last transfer */

                        if (j != n)

                            tcg_gen_addi_i32(addr, addr, 4);

                    }

                }

                if (insn & (1 << 21)) {

                    /* write back */

                    if (insn & (1 << 23)) {

                        if (insn & (1 << 24)) {

                            /* pre increment */

                        } else {

                            /* post increment */

                            tcg_gen_addi_i32(addr, addr, 4);

                        }

                    } else {

                        if (insn & (1 << 24)) {

                            /* pre decrement */

                            if (n != 1)

                                tcg_gen_addi_i32(addr, addr, -((n - 1) * 4));

                        } else {

                            /* post decrement */

                            tcg_gen_addi_i32(addr, addr, -(n * 4));

                        }

                    }

                    store_reg(s, rn, addr);

                } else {

                    tcg_temp_free_i32(addr);

                }

                if (loaded_base) {

                    store_reg(s, rn, loaded_var);

                }

                if ((insn & (1 << 22)) && !user) {

                    /* Restore CPSR from SPSR.  */

                    tmp = load_cpu_field(spsr);

                    gen_set_cpsr(tmp, 0xffffffff);

                    tcg_temp_free_i32(tmp);

                    s->is_jmp = DISAS_UPDATE;

                }

            }

            break;

        case 0xa:

        case 0xb:

            {

                int32_t offset;



                /* branch (and link) */

                val = (int32_t)s->pc;

                if (insn & (1 << 24)) {

                    tmp = tcg_temp_new_i32();

                    tcg_gen_movi_i32(tmp, val);

                    store_reg(s, 14, tmp);

                }

                offset = (((int32_t)insn << 8) >> 8);

                val += (offset << 2) + 4;

                gen_jmp(s, val);

            }

            break;

        case 0xc:

        case 0xd:

        case 0xe:

            /* Coprocessor.  */

            if (disas_coproc_insn(env, s, insn))

                goto illegal_op;

            break;

        case 0xf:

            /* swi */

            gen_set_pc_im(s->pc);

            s->is_jmp = DISAS_SWI;

            break;

        default:

        illegal_op:

            gen_exception_insn(s, 4, EXCP_UDEF);

            break;

        }

    }

}