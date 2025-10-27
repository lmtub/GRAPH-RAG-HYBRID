static void disas_arm_insn(CPUState * env, DisasContext *s)

{

    unsigned int cond, insn, val, op1, i, shift, rm, rs, rn, rd, sh;

    

    insn = ldl_code(s->pc);

    s->pc += 4;

    

    cond = insn >> 28;

    if (cond == 0xf){

        /* Unconditional instructions.  */

        if ((insn & 0x0d70f000) == 0x0550f000)

            return; /* PLD */

        else if ((insn & 0x0e000000) == 0x0a000000) {

            /* branch link and change to thumb (blx <offset>) */

            int32_t offset;



            val = (uint32_t)s->pc;

            gen_op_movl_T0_im(val);

            gen_movl_reg_T0(s, 14);

            /* Sign-extend the 24-bit offset */

            offset = (((int32_t)insn) << 8) >> 8;

            /* offset * 4 + bit24 * 2 + (thumb bit) */

            val += (offset << 2) | ((insn >> 23) & 2) | 1;

            /* pipeline offset */

            val += 4;

            gen_op_movl_T0_im(val);

            gen_bx(s);

            return;

        } else if ((insn & 0x0fe00000) == 0x0c400000) {

            /* Coprocessor double register transfer.  */

        } else if ((insn & 0x0f000010) == 0x0e000010) {

            /* Additional coprocessor register transfer.  */

        } else if ((insn & 0x0ff10010) == 0x01000000) {

            /* cps (privileged) */

        } else if ((insn & 0x0ffffdff) == 0x01010000) {

            /* setend */

            if (insn & (1 << 9)) {

                /* BE8 mode not implemented.  */

                goto illegal_op;

            }

            return;

        }

        goto illegal_op;

    }

    if (cond != 0xe) {

        /* if not always execute, we generate a conditional jump to

           next instruction */

        s->condlabel = gen_new_label();

        gen_test_cc[cond ^ 1](s->condlabel);

        s->condjmp = 1;

        //gen_test_cc[cond ^ 1]((long)s->tb, (long)s->pc);

        //s->is_jmp = DISAS_JUMP_NEXT;

    }

    if ((insn & 0x0f900000) == 0x03000000) {

        if ((insn & 0x0fb0f000) != 0x0320f000)

            goto illegal_op;

        /* CPSR = immediate */

        val = insn & 0xff;

        shift = ((insn >> 8) & 0xf) * 2;

        if (shift)

            val = (val >> shift) | (val << (32 - shift));

        gen_op_movl_T0_im(val);

        i = ((insn & (1 << 22)) != 0);

        if (gen_set_psr_T0(s, msr_mask(s, (insn >> 16) & 0xf, i), i))

            goto illegal_op;

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

                gen_movl_T0_reg(s, rm);

                i = ((op1 & 2) != 0);

                if (gen_set_psr_T0(s, msr_mask(s, (insn >> 16) & 0xf, i), i))

                    goto illegal_op;

            } else {

                /* reg = PSR */

                rd = (insn >> 12) & 0xf;

                if (op1 & 2) {

                    if (IS_USER(s))

                        goto illegal_op;

                    gen_op_movl_T0_spsr();

                } else {

                    gen_op_movl_T0_cpsr();

                }

                gen_movl_reg_T0(s, rd);

            }

            break;

        case 0x1:

            if (op1 == 1) {

                /* branch/exchange thumb (bx).  */

                gen_movl_T0_reg(s, rm);

                gen_bx(s);

            } else if (op1 == 3) {

                /* clz */

                rd = (insn >> 12) & 0xf;

                gen_movl_T0_reg(s, rm);

                gen_op_clz_T0();

                gen_movl_reg_T0(s, rd);

            } else {

                goto illegal_op;

            }

            break;

        case 0x2:

            if (op1 == 1) {

                ARCH(5J); /* bxj */

                /* Trivial implementation equivalent to bx.  */

                gen_movl_T0_reg(s, rm);

                gen_bx(s);

            } else {

                goto illegal_op;

            }

            break;

        case 0x3:

            if (op1 != 1)

              goto illegal_op;



            /* branch link/exchange thumb (blx) */

            val = (uint32_t)s->pc;

            gen_op_movl_T0_im(val);

            gen_movl_reg_T0(s, 14);

            gen_movl_T0_reg(s, rm);

            gen_bx(s);

            break;

        case 0x5: /* saturating add/subtract */

            rd = (insn >> 12) & 0xf;

            rn = (insn >> 16) & 0xf;

            gen_movl_T0_reg(s, rm);

            gen_movl_T1_reg(s, rn);

            if (op1 & 2)

                gen_op_double_T1_saturate();

            if (op1 & 1)

                gen_op_subl_T0_T1_saturate();

            else

                gen_op_addl_T0_T1_saturate();

            gen_movl_reg_T0(s, rd);

            break;

        case 7: /* bkpt */

            gen_op_movl_T0_im((long)s->pc - 4);

            gen_op_movl_reg_TN[0][15]();

            gen_op_bkpt();

            s->is_jmp = DISAS_JUMP;

            break;

        case 0x8: /* signed multiply */

        case 0xa:

        case 0xc:

        case 0xe:

            rs = (insn >> 8) & 0xf;

            rn = (insn >> 12) & 0xf;

            rd = (insn >> 16) & 0xf;

            if (op1 == 1) {

                /* (32 * 16) >> 16 */

                gen_movl_T0_reg(s, rm);

                gen_movl_T1_reg(s, rs);

                if (sh & 4)

                    gen_op_sarl_T1_im(16);

                else

                    gen_op_sxth_T1();

                gen_op_imulw_T0_T1();

                if ((sh & 2) == 0) {

                    gen_movl_T1_reg(s, rn);

                    gen_op_addl_T0_T1_setq();

                }

                gen_movl_reg_T0(s, rd);

            } else {

                /* 16 * 16 */

                gen_movl_T0_reg(s, rm);

                gen_movl_T1_reg(s, rs);

                gen_mulxy(sh & 2, sh & 4);

                if (op1 == 2) {

                    gen_op_signbit_T1_T0();

                    gen_op_addq_T0_T1(rn, rd);

                    gen_movl_reg_T0(s, rn);

                    gen_movl_reg_T1(s, rd);

                } else {

                    if (op1 == 0) {

                        gen_movl_T1_reg(s, rn);

                        gen_op_addl_T0_T1_setq();

                    }

                    gen_movl_reg_T0(s, rd);

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

            if (shift)

                val = (val >> shift) | (val << (32 - shift));

            gen_op_movl_T1_im(val);

            if (logic_cc && shift)

                gen_op_mov_CF_T1();

        } else {

            /* register */

            rm = (insn) & 0xf;

            gen_movl_T1_reg(s, rm);

            shiftop = (insn >> 5) & 3;

            if (!(insn & (1 << 4))) {

                shift = (insn >> 7) & 0x1f;

                if (shift != 0) {

                    if (logic_cc) {

                        gen_shift_T1_im_cc[shiftop](shift);

                    } else {

                        gen_shift_T1_im[shiftop](shift);

                    }

                } else if (shiftop != 0) {

                    if (logic_cc) {

                        gen_shift_T1_0_cc[shiftop]();

                    } else {

                        gen_shift_T1_0[shiftop]();

                    }

                }

            } else {

                rs = (insn >> 8) & 0xf;

                gen_movl_T0_reg(s, rs);

                if (logic_cc) {

                    gen_shift_T1_T0_cc[shiftop]();

                } else {

                    gen_shift_T1_T0[shiftop]();

                }

            }

        }

        if (op1 != 0x0f && op1 != 0x0d) {

            rn = (insn >> 16) & 0xf;

            gen_movl_T0_reg(s, rn);

        }

        rd = (insn >> 12) & 0xf;

        switch(op1) {

        case 0x00:

            gen_op_andl_T0_T1();

            gen_movl_reg_T0(s, rd);

            if (logic_cc)

                gen_op_logic_T0_cc();

            break;

        case 0x01:

            gen_op_xorl_T0_T1();

            gen_movl_reg_T0(s, rd);

            if (logic_cc)

                gen_op_logic_T0_cc();

            break;

        case 0x02:

            if (set_cc && rd == 15) {

                /* SUBS r15, ... is used for exception return.  */

                if (IS_USER(s))

                    goto illegal_op;

                gen_op_subl_T0_T1_cc();

                gen_exception_return(s);

            } else {

                if (set_cc)

                    gen_op_subl_T0_T1_cc();

                else

                    gen_op_subl_T0_T1();

                gen_movl_reg_T0(s, rd);

            }

            break;

        case 0x03:

            if (set_cc)

                gen_op_rsbl_T0_T1_cc();

            else

                gen_op_rsbl_T0_T1();

            gen_movl_reg_T0(s, rd);

            break;

        case 0x04:

            if (set_cc)

                gen_op_addl_T0_T1_cc();

            else

                gen_op_addl_T0_T1();

            gen_movl_reg_T0(s, rd);

            break;

        case 0x05:

            if (set_cc)

                gen_op_adcl_T0_T1_cc();

            else

                gen_op_adcl_T0_T1();

            gen_movl_reg_T0(s, rd);

            break;

        case 0x06:

            if (set_cc)

                gen_op_sbcl_T0_T1_cc();

            else

                gen_op_sbcl_T0_T1();

            gen_movl_reg_T0(s, rd);

            break;

        case 0x07:

            if (set_cc)

                gen_op_rscl_T0_T1_cc();

            else

                gen_op_rscl_T0_T1();

            gen_movl_reg_T0(s, rd);

            break;

        case 0x08:

            if (set_cc) {

                gen_op_andl_T0_T1();

                gen_op_logic_T0_cc();

            }

            break;

        case 0x09:

            if (set_cc) {

                gen_op_xorl_T0_T1();

                gen_op_logic_T0_cc();

            }

            break;

        case 0x0a:

            if (set_cc) {

                gen_op_subl_T0_T1_cc();

            }

            break;

        case 0x0b:

            if (set_cc) {

                gen_op_addl_T0_T1_cc();

            }

            break;

        case 0x0c:

            gen_op_orl_T0_T1();

            gen_movl_reg_T0(s, rd);

            if (logic_cc)

                gen_op_logic_T0_cc();

            break;

        case 0x0d:

            if (logic_cc && rd == 15) {

                /* MOVS r15, ... is used for exception return.  */

                if (IS_USER(s))

                    goto illegal_op;

                gen_op_movl_T0_T1();

                gen_exception_return(s);

            } else {

                gen_movl_reg_T1(s, rd);

                if (logic_cc)

                    gen_op_logic_T1_cc();

            }

            break;

        case 0x0e:

            gen_op_bicl_T0_T1();

            gen_movl_reg_T0(s, rd);

            if (logic_cc)

                gen_op_logic_T0_cc();

            break;

        default:

        case 0x0f:

            gen_op_notl_T1();

            gen_movl_reg_T1(s, rd);

            if (logic_cc)

                gen_op_logic_T1_cc();

            break;

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

                    if (((insn >> 22) & 3) == 0) {

                        /* 32 bit mul */

                        gen_movl_T0_reg(s, rs);

                        gen_movl_T1_reg(s, rm);

                        gen_op_mul_T0_T1();

                        if (insn & (1 << 21)) {

                            gen_movl_T1_reg(s, rn);

                            gen_op_addl_T0_T1();

                        }

                        if (insn & (1 << 20)) 

                            gen_op_logic_T0_cc();

                        gen_movl_reg_T0(s, rd);

                    } else {

                        /* 64 bit mul */

                        gen_movl_T0_reg(s, rs);

                        gen_movl_T1_reg(s, rm);

                        if (insn & (1 << 22)) 

                            gen_op_imull_T0_T1();

                        else

                            gen_op_mull_T0_T1();

                        if (insn & (1 << 21)) /* mult accumulate */

                            gen_op_addq_T0_T1(rn, rd);

                        if (!(insn & (1 << 23))) { /* double accumulate */

                            ARCH(6);

                            gen_op_addq_lo_T0_T1(rn);

                            gen_op_addq_lo_T0_T1(rd);

                        }

                        if (insn & (1 << 20)) 

                            gen_op_logicq_cc();

                        gen_movl_reg_T0(s, rn);

                        gen_movl_reg_T1(s, rd);

                    }

                } else {

                    rn = (insn >> 16) & 0xf;

                    rd = (insn >> 12) & 0xf;

                    if (insn & (1 << 23)) {

                        /* load/store exclusive */

                        goto illegal_op;

                    } else {

                        /* SWP instruction */

                        rm = (insn) & 0xf;

                        

                        gen_movl_T0_reg(s, rm);

                        gen_movl_T1_reg(s, rn);

                        if (insn & (1 << 22)) {

                            gen_ldst(swpb, s);

                        } else {

                            gen_ldst(swpl, s);

                        }

                        gen_movl_reg_T0(s, rd);

                    }

                }

            } else {

                int address_offset;

                /* Misc load/store */

                rn = (insn >> 16) & 0xf;

                rd = (insn >> 12) & 0xf;

                gen_movl_T1_reg(s, rn);

                if (insn & (1 << 24))

                    gen_add_datah_offset(s, insn, 0);

                address_offset = 0;

                if (insn & (1 << 20)) {

                    /* load */

                    switch(sh) {

                    case 1:

                        gen_ldst(lduw, s);

                        break;

                    case 2:

                        gen_ldst(ldsb, s);

                        break;

                    default:

                    case 3:

                        gen_ldst(ldsw, s);

                        break;

                    }

                    gen_movl_reg_T0(s, rd);

                } else if (sh & 2) {

                    /* doubleword */

                    if (sh & 1) {

                        /* store */

                        gen_movl_T0_reg(s, rd);

                        gen_ldst(stl, s);

                        gen_op_addl_T1_im(4);

                        gen_movl_T0_reg(s, rd + 1);

                        gen_ldst(stl, s);

                    } else {

                        /* load */

                        gen_ldst(ldl, s);

                        gen_movl_reg_T0(s, rd);

                        gen_op_addl_T1_im(4);

                        gen_ldst(ldl, s);

                        gen_movl_reg_T0(s, rd + 1);

                    }

                    address_offset = -4;

                } else {

                    /* store */

                    gen_movl_T0_reg(s, rd);

                    gen_ldst(stw, s);

                }

                if (!(insn & (1 << 24))) {

                    gen_add_datah_offset(s, insn, address_offset);

                    gen_movl_reg_T1(s, rn);

                } else if (insn & (1 << 21)) {

                    if (address_offset)

                        gen_op_addl_T1_im(address_offset);

                    gen_movl_reg_T1(s, rn);

                }

            }

            break;

        case 0x4:

        case 0x5:

        case 0x6:

        case 0x7:

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

            gen_movl_T1_reg(s, rn);

            i = (IS_USER(s) || (insn & 0x01200000) == 0x00200000);

            if (insn & (1 << 24))

                gen_add_data_offset(s, insn);

            if (insn & (1 << 20)) {

                /* load */


#if defined(CONFIG_USER_ONLY)

                if (insn & (1 << 22))

                    gen_op_ldub_raw();

                else

                    gen_op_ldl_raw();

#else

                if (insn & (1 << 22)) {

                    if (i)

                        gen_op_ldub_user();

                    else

                        gen_op_ldub_kernel();

                } else {

                    if (i)

                        gen_op_ldl_user();

                    else

                        gen_op_ldl_kernel();

                }

#endif

                if (rd == 15)

                    gen_bx(s);

                else

                    gen_movl_reg_T0(s, rd);

            } else {

                /* store */

                gen_movl_T0_reg(s, rd);

#if defined(CONFIG_USER_ONLY)

                if (insn & (1 << 22))

                    gen_op_stb_raw();

                else

                    gen_op_stl_raw();

#else

                if (insn & (1 << 22)) {

                    if (i)

                        gen_op_stb_user();

                    else

                        gen_op_stb_kernel();

                } else {

                    if (i)

                        gen_op_stl_user();

                    else

                        gen_op_stl_kernel();

                }

#endif

            }

            if (!(insn & (1 << 24))) {

                gen_add_data_offset(s, insn);

                gen_movl_reg_T1(s, rn);

            } else if (insn & (1 << 21))

                gen_movl_reg_T1(s, rn); {

            }

            break;

        case 0x08:

        case 0x09:

            {

                int j, n, user, loaded_base;

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

                gen_movl_T1_reg(s, rn);

                

                /* compute total size */

                loaded_base = 0;

                n = 0;

                for(i=0;i<16;i++) {

                    if (insn & (1 << i))

                        n++;

                }

                /* XXX: test invalid n == 0 case ? */

                if (insn & (1 << 23)) {

                    if (insn & (1 << 24)) {

                        /* pre increment */

                        gen_op_addl_T1_im(4);

                    } else {

                        /* post increment */

                    }

                } else {

                    if (insn & (1 << 24)) {

                        /* pre decrement */

                        gen_op_addl_T1_im(-(n * 4));

                    } else {

                        /* post decrement */

                        if (n != 1)

                            gen_op_addl_T1_im(-((n - 1) * 4));

                    }

                }

                j = 0;

                for(i=0;i<16;i++) {

                    if (insn & (1 << i)) {

                        if (insn & (1 << 20)) {

                            /* load */

                            gen_ldst(ldl, s);

                            if (i == 15) {

                                gen_bx(s);

                            } else if (user) {

                                gen_op_movl_user_T0(i);

                            } else if (i == rn) {

                                gen_op_movl_T2_T0();

                                loaded_base = 1;

                            } else {

                                gen_movl_reg_T0(s, i);

                            }

                        } else {

                            /* store */

                            if (i == 15) {

                                /* special case: r15 = PC + 12 */

                                val = (long)s->pc + 8;

                                gen_op_movl_TN_im[0](val);

                            } else if (user) {

                                gen_op_movl_T0_user(i);

                            } else {

                                gen_movl_T0_reg(s, i);

                            }

                            gen_ldst(stl, s);

                        }

                        j++;

                        /* no need to add after the last transfer */

                        if (j != n)

                            gen_op_addl_T1_im(4);

                    }

                }

                if (insn & (1 << 21)) {

                    /* write back */

                    if (insn & (1 << 23)) {

                        if (insn & (1 << 24)) {

                            /* pre increment */

                        } else {

                            /* post increment */

                            gen_op_addl_T1_im(4);

                        }

                    } else {

                        if (insn & (1 << 24)) {

                            /* pre decrement */

                            if (n != 1)

                                gen_op_addl_T1_im(-((n - 1) * 4));

                        } else {

                            /* post decrement */

                            gen_op_addl_T1_im(-(n * 4));

                        }

                    }

                    gen_movl_reg_T1(s, rn);

                }

                if (loaded_base) {

                    gen_op_movl_T0_T2();

                    gen_movl_reg_T0(s, rn);

                }

                if ((insn & (1 << 22)) && !user) {

                    /* Restore CPSR from SPSR.  */

                    gen_op_movl_T0_spsr();

                    gen_op_movl_cpsr_T0(0xffffffff);

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

                    gen_op_movl_T0_im(val);

                    gen_op_movl_reg_TN[0][14]();

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

            op1 = (insn >> 8) & 0xf;

            switch (op1) {

            case 10:

            case 11:

                if (disas_vfp_insn (env, s, insn))

                    goto illegal_op;

                break;

            case 15:

                if (disas_cp15_insn (s, insn))

                    goto illegal_op;

                break;

            default:

                /* unknown coprocessor.  */

                goto illegal_op;

            }

            break;

        case 0xf:

            /* swi */

            gen_op_movl_T0_im((long)s->pc);

            gen_op_movl_reg_TN[0][15]();

            gen_op_swi();

            s->is_jmp = DISAS_JUMP;

            break;

        default:

        illegal_op:

            gen_op_movl_T0_im((long)s->pc - 4);

            gen_op_movl_reg_TN[0][15]();

            gen_op_undef_insn();

            s->is_jmp = DISAS_JUMP;

            break;

        }

    }

}