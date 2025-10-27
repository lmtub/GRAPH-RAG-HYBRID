static int disas_coproc_insn(DisasContext *s, uint32_t insn)

{

    int cpnum, is64, crn, crm, opc1, opc2, isread, rt, rt2;

    const ARMCPRegInfo *ri;



    cpnum = (insn >> 8) & 0xf;



    /* First check for coprocessor space used for XScale/iwMMXt insns */

    if (arm_dc_feature(s, ARM_FEATURE_XSCALE) && (cpnum < 2)) {

        if (extract32(s->c15_cpar, cpnum, 1) == 0) {

            return 1;

        }

        if (arm_dc_feature(s, ARM_FEATURE_IWMMXT)) {

            return disas_iwmmxt_insn(s, insn);

        } else if (arm_dc_feature(s, ARM_FEATURE_XSCALE)) {

            return disas_dsp_insn(s, insn);

        }

        return 1;

    }



    /* Otherwise treat as a generic register access */

    is64 = (insn & (1 << 25)) == 0;

    if (!is64 && ((insn & (1 << 4)) == 0)) {

        /* cdp */

        return 1;

    }



    crm = insn & 0xf;

    if (is64) {

        crn = 0;

        opc1 = (insn >> 4) & 0xf;

        opc2 = 0;

        rt2 = (insn >> 16) & 0xf;

    } else {

        crn = (insn >> 16) & 0xf;

        opc1 = (insn >> 21) & 7;

        opc2 = (insn >> 5) & 7;

        rt2 = 0;

    }

    isread = (insn >> 20) & 1;

    rt = (insn >> 12) & 0xf;



    ri = get_arm_cp_reginfo(s->cp_regs,

            ENCODE_CP_REG(cpnum, is64, s->ns, crn, crm, opc1, opc2));

    if (ri) {

        /* Check access permissions */

        if (!cp_access_ok(s->current_el, ri, isread)) {

            return 1;

        }



        if (ri->accessfn ||

            (arm_dc_feature(s, ARM_FEATURE_XSCALE) && cpnum < 14)) {

            /* Emit code to perform further access permissions checks at

             * runtime; this may result in an exception.

             * Note that on XScale all cp0..c13 registers do an access check

             * call in order to handle c15_cpar.

             */

            TCGv_ptr tmpptr;

            TCGv_i32 tcg_syn;

            uint32_t syndrome;



            /* Note that since we are an implementation which takes an

             * exception on a trapped conditional instruction only if the

             * instruction passes its condition code check, we can take

             * advantage of the clause in the ARM ARM that allows us to set

             * the COND field in the instruction to 0xE in all cases.

             * We could fish the actual condition out of the insn (ARM)

             * or the condexec bits (Thumb) but it isn't necessary.

             */

            switch (cpnum) {

            case 14:

                if (is64) {

                    syndrome = syn_cp14_rrt_trap(1, 0xe, opc1, crm, rt, rt2,

                                                 isread, s->thumb);

                } else {

                    syndrome = syn_cp14_rt_trap(1, 0xe, opc1, opc2, crn, crm,

                                                rt, isread, s->thumb);

                }

                break;

            case 15:

                if (is64) {

                    syndrome = syn_cp15_rrt_trap(1, 0xe, opc1, crm, rt, rt2,

                                                 isread, s->thumb);

                } else {

                    syndrome = syn_cp15_rt_trap(1, 0xe, opc1, opc2, crn, crm,

                                                rt, isread, s->thumb);

                }

                break;

            default:

                /* ARMv8 defines that only coprocessors 14 and 15 exist,

                 * so this can only happen if this is an ARMv7 or earlier CPU,

                 * in which case the syndrome information won't actually be

                 * guest visible.

                 */

                assert(!arm_dc_feature(s, ARM_FEATURE_V8));

                syndrome = syn_uncategorized();

                break;

            }



            gen_set_pc_im(s, s->pc);

            tmpptr = tcg_const_ptr(ri);

            tcg_syn = tcg_const_i32(syndrome);

            gen_helper_access_check_cp_reg(cpu_env, tmpptr, tcg_syn);

            tcg_temp_free_ptr(tmpptr);

            tcg_temp_free_i32(tcg_syn);

        }



        /* Handle special cases first */

        switch (ri->type & ~(ARM_CP_FLAG_MASK & ~ARM_CP_SPECIAL)) {

        case ARM_CP_NOP:

            return 0;

        case ARM_CP_WFI:

            if (isread) {

                return 1;

            }

            gen_set_pc_im(s, s->pc);

            s->is_jmp = DISAS_WFI;

            return 0;

        default:

            break;

        }



        if ((s->tb->cflags & CF_USE_ICOUNT) && (ri->type & ARM_CP_IO)) {

            gen_io_start();

        }



        if (isread) {

            /* Read */

            if (is64) {

                TCGv_i64 tmp64;

                TCGv_i32 tmp;

                if (ri->type & ARM_CP_CONST) {

                    tmp64 = tcg_const_i64(ri->resetvalue);

                } else if (ri->readfn) {

                    TCGv_ptr tmpptr;

                    tmp64 = tcg_temp_new_i64();

                    tmpptr = tcg_const_ptr(ri);

                    gen_helper_get_cp_reg64(tmp64, cpu_env, tmpptr);

                    tcg_temp_free_ptr(tmpptr);

                } else {

                    tmp64 = tcg_temp_new_i64();

                    tcg_gen_ld_i64(tmp64, cpu_env, ri->fieldoffset);

                }

                tmp = tcg_temp_new_i32();

                tcg_gen_trunc_i64_i32(tmp, tmp64);

                store_reg(s, rt, tmp);

                tcg_gen_shri_i64(tmp64, tmp64, 32);

                tmp = tcg_temp_new_i32();

                tcg_gen_trunc_i64_i32(tmp, tmp64);

                tcg_temp_free_i64(tmp64);

                store_reg(s, rt2, tmp);

            } else {

                TCGv_i32 tmp;

                if (ri->type & ARM_CP_CONST) {

                    tmp = tcg_const_i32(ri->resetvalue);

                } else if (ri->readfn) {

                    TCGv_ptr tmpptr;

                    tmp = tcg_temp_new_i32();

                    tmpptr = tcg_const_ptr(ri);

                    gen_helper_get_cp_reg(tmp, cpu_env, tmpptr);

                    tcg_temp_free_ptr(tmpptr);

                } else {

                    tmp = load_cpu_offset(ri->fieldoffset);

                }

                if (rt == 15) {

                    /* Destination register of r15 for 32 bit loads sets

                     * the condition codes from the high 4 bits of the value

                     */

                    gen_set_nzcv(tmp);

                    tcg_temp_free_i32(tmp);

                } else {

                    store_reg(s, rt, tmp);

                }

            }

        } else {

            /* Write */

            if (ri->type & ARM_CP_CONST) {

                /* If not forbidden by access permissions, treat as WI */

                return 0;

            }



            if (is64) {

                TCGv_i32 tmplo, tmphi;

                TCGv_i64 tmp64 = tcg_temp_new_i64();

                tmplo = load_reg(s, rt);

                tmphi = load_reg(s, rt2);

                tcg_gen_concat_i32_i64(tmp64, tmplo, tmphi);

                tcg_temp_free_i32(tmplo);

                tcg_temp_free_i32(tmphi);

                if (ri->writefn) {

                    TCGv_ptr tmpptr = tcg_const_ptr(ri);

                    gen_helper_set_cp_reg64(cpu_env, tmpptr, tmp64);

                    tcg_temp_free_ptr(tmpptr);

                } else {

                    tcg_gen_st_i64(tmp64, cpu_env, ri->fieldoffset);

                }

                tcg_temp_free_i64(tmp64);

            } else {

                if (ri->writefn) {

                    TCGv_i32 tmp;

                    TCGv_ptr tmpptr;

                    tmp = load_reg(s, rt);

                    tmpptr = tcg_const_ptr(ri);

                    gen_helper_set_cp_reg(cpu_env, tmpptr, tmp);

                    tcg_temp_free_ptr(tmpptr);

                    tcg_temp_free_i32(tmp);

                } else {

                    TCGv_i32 tmp = load_reg(s, rt);

                    store_cpu_offset(tmp, ri->fieldoffset);

                }

            }

        }



        if ((s->tb->cflags & CF_USE_ICOUNT) && (ri->type & ARM_CP_IO)) {

            /* I/O operations must end the TB here (whether read or write) */

            gen_io_end();

            gen_lookup_tb(s);

        } else if (!isread && !(ri->type & ARM_CP_SUPPRESS_TB_END)) {

            /* We default to ending the TB on a coprocessor register write,

             * but allow this to be suppressed by the register definition

             * (usually only necessary to work around guest bugs).

             */

            gen_lookup_tb(s);

        }



        return 0;

    }



    /* Unknown register; this might be a guest error or a QEMU

     * unimplemented feature.

     */

    if (is64) {

        qemu_log_mask(LOG_UNIMP, "%s access to unsupported AArch32 "

                      "64 bit system register cp:%d opc1: %d crm:%d "

                      "(%s)\n",

                      isread ? "read" : "write", cpnum, opc1, crm,

                      s->ns ? "non-secure" : "secure");

    } else {

        qemu_log_mask(LOG_UNIMP, "%s access to unsupported AArch32 "

                      "system register cp:%d opc1:%d crn:%d crm:%d opc2:%d "

                      "(%s)\n",

                      isread ? "read" : "write", cpnum, opc1, crn, crm, opc2,

                      s->ns ? "non-secure" : "secure");

    }



    return 1;

}
