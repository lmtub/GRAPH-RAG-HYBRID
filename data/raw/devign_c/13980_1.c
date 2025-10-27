static int disas_cp15_insn(CPUState *env, DisasContext *s, uint32_t insn)

{

    uint32_t rd;

    TCGv tmp, tmp2;



    /* M profile cores use memory mapped registers instead of cp15.  */

    if (arm_feature(env, ARM_FEATURE_M))

	return 1;



    if ((insn & (1 << 25)) == 0) {

        if (insn & (1 << 20)) {

            /* mrrc */

            return 1;

        }

        /* mcrr.  Used for block cache operations, so implement as no-op.  */

        return 0;

    }

    if ((insn & (1 << 4)) == 0) {

        /* cdp */

        return 1;

    }

    if (IS_USER(s) && !cp15_user_ok(insn)) {

        return 1;

    }



    /* Pre-v7 versions of the architecture implemented WFI via coprocessor

     * instructions rather than a separate instruction.

     */

    if ((insn & 0x0fff0fff) == 0x0e070f90) {

        /* 0,c7,c0,4: Standard v6 WFI (also used in some pre-v6 cores).

         * In v7, this must NOP.

         */

        if (!arm_feature(env, ARM_FEATURE_V7)) {

            /* Wait for interrupt.  */

            gen_set_pc_im(s->pc);

            s->is_jmp = DISAS_WFI;

        }

        return 0;

    }



    if ((insn & 0x0fff0fff) == 0x0e070f58) {

        /* 0,c7,c8,2: Not all pre-v6 cores implemented this WFI,

         * so this is slightly over-broad.

         */

        if (!arm_feature(env, ARM_FEATURE_V6)) {

            /* Wait for interrupt.  */

            gen_set_pc_im(s->pc);

            s->is_jmp = DISAS_WFI;

            return 0;

        }

        /* Otherwise fall through to handle via helper function.

         * In particular, on v7 and some v6 cores this is one of

         * the VA-PA registers.

         */

    }



    rd = (insn >> 12) & 0xf;



    if (cp15_tls_load_store(env, s, insn, rd))

        return 0;



    tmp2 = tcg_const_i32(insn);

    if (insn & ARM_CP_RW_BIT) {

        tmp = new_tmp();

        gen_helper_get_cp15(tmp, cpu_env, tmp2);

        /* If the destination register is r15 then sets condition codes.  */

        if (rd != 15)

            store_reg(s, rd, tmp);

        else

            dead_tmp(tmp);

    } else {

        tmp = load_reg(s, rd);

        gen_helper_set_cp15(cpu_env, tmp2, tmp);

        dead_tmp(tmp);

        /* Normally we would always end the TB here, but Linux

         * arch/arm/mach-pxa/sleep.S expects two instructions following

         * an MMU enable to execute from cache.  Imitate this behaviour.  */

        if (!arm_feature(env, ARM_FEATURE_XSCALE) ||

                (insn & 0x0fff0fff) != 0x0e010f10)

            gen_lookup_tb(s);

    }

    tcg_temp_free_i32(tmp2);

    return 0;

}
