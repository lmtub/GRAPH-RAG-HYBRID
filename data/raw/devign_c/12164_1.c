static void dec_mul(DisasContext *dc)

{

    if (dc->format == OP_FMT_RI) {

        LOG_DIS("muli r%d, r%d, %d\n", dc->r0, dc->r1,

                sign_extend(dc->imm16, 16));

    } else {

        LOG_DIS("mul r%d, r%d, r%d\n", dc->r2, dc->r0, dc->r1);

    }



    if (!(dc->env->features & LM32_FEATURE_MULTIPLY)) {

        cpu_abort(dc->env, "hardware multiplier is not available\n");

    }



    if (dc->format == OP_FMT_RI) {

        tcg_gen_muli_tl(cpu_R[dc->r1], cpu_R[dc->r0],

                sign_extend(dc->imm16, 16));

    } else {

        tcg_gen_mul_tl(cpu_R[dc->r2], cpu_R[dc->r0], cpu_R[dc->r1]);

    }

}
