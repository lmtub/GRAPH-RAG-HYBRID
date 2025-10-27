static void dec_divu(DisasContext *dc)

{

    int l1;



    LOG_DIS("divu r%d, r%d, r%d\n", dc->r2, dc->r0, dc->r1);



    if (!(dc->env->features & LM32_FEATURE_DIVIDE)) {

        cpu_abort(dc->env, "hardware divider is not available\n");

    }



    l1 = gen_new_label();

    tcg_gen_brcondi_tl(TCG_COND_NE, cpu_R[dc->r1], 0, l1);

    tcg_gen_movi_tl(cpu_pc, dc->pc);

    t_gen_raise_exception(dc, EXCP_DIVIDE_BY_ZERO);

    gen_set_label(l1);

    tcg_gen_divu_tl(cpu_R[dc->r2], cpu_R[dc->r0], cpu_R[dc->r1]);

}
