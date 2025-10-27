static void handle_sys(DisasContext *s, uint32_t insn, bool isread,

                       unsigned int op0, unsigned int op1, unsigned int op2,

                       unsigned int crn, unsigned int crm, unsigned int rt)

{

    const ARMCPRegInfo *ri;

    TCGv_i64 tcg_rt;



    ri = get_arm_cp_reginfo(s->cp_regs,

                            ENCODE_AA64_CP_REG(CP_REG_ARM64_SYSREG_CP,

                                               crn, crm, op0, op1, op2));



    if (!ri) {

        /* Unknown register; this might be a guest error or a QEMU

         * unimplemented feature.

         */

        qemu_log_mask(LOG_UNIMP, "%s access to unsupported AArch64 "

                      "system register op0:%d op1:%d crn:%d crm:%d op2:%d\n",

                      isread ? "read" : "write", op0, op1, crn, crm, op2);

        unallocated_encoding(s);

        return;

    }



    /* Check access permissions */

    if (!cp_access_ok(s->current_pl, ri, isread)) {

        unallocated_encoding(s);

        return;

    }



    if (ri->accessfn) {

        /* Emit code to perform further access permissions checks at

         * runtime; this may result in an exception.

         */

        TCGv_ptr tmpptr;

        gen_a64_set_pc_im(s->pc - 4);

        tmpptr = tcg_const_ptr(ri);

        gen_helper_access_check_cp_reg(cpu_env, tmpptr);

        tcg_temp_free_ptr(tmpptr);

    }



    /* Handle special cases first */

    switch (ri->type & ~(ARM_CP_FLAG_MASK & ~ARM_CP_SPECIAL)) {

    case ARM_CP_NOP:

        return;

    case ARM_CP_NZCV:

        tcg_rt = cpu_reg(s, rt);

        if (isread) {

            gen_get_nzcv(tcg_rt);

        } else {

            gen_set_nzcv(tcg_rt);

        }

        return;

    case ARM_CP_CURRENTEL:

        /* Reads as current EL value from pstate, which is

         * guaranteed to be constant by the tb flags.

         */

        tcg_rt = cpu_reg(s, rt);

        tcg_gen_movi_i64(tcg_rt, s->current_pl << 2);

        return;

    default:

        break;

    }



    if (use_icount && (ri->type & ARM_CP_IO)) {

        gen_io_start();

    }



    tcg_rt = cpu_reg(s, rt);



    if (isread) {

        if (ri->type & ARM_CP_CONST) {

            tcg_gen_movi_i64(tcg_rt, ri->resetvalue);

        } else if (ri->readfn) {

            TCGv_ptr tmpptr;

            tmpptr = tcg_const_ptr(ri);

            gen_helper_get_cp_reg64(tcg_rt, cpu_env, tmpptr);

            tcg_temp_free_ptr(tmpptr);

        } else {

            tcg_gen_ld_i64(tcg_rt, cpu_env, ri->fieldoffset);

        }

    } else {

        if (ri->type & ARM_CP_CONST) {

            /* If not forbidden by access permissions, treat as WI */

            return;

        } else if (ri->writefn) {

            TCGv_ptr tmpptr;

            tmpptr = tcg_const_ptr(ri);

            gen_helper_set_cp_reg64(cpu_env, tmpptr, tcg_rt);

            tcg_temp_free_ptr(tmpptr);

        } else {

            tcg_gen_st_i64(tcg_rt, cpu_env, ri->fieldoffset);

        }

    }



    if (use_icount && (ri->type & ARM_CP_IO)) {

        /* I/O operations must end the TB here (whether read or write) */

        gen_io_end();

        s->is_jmp = DISAS_UPDATE;

    } else if (!isread && !(ri->type & ARM_CP_SUPPRESS_TB_END)) {

        /* We default to ending the TB on a coprocessor register write,

         * but allow this to be suppressed by the register definition

         * (usually only necessary to work around guest bugs).

         */

        s->is_jmp = DISAS_UPDATE;

    }

}
