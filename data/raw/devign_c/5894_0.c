do_gen_eob_worker(DisasContext *s, bool inhibit, bool recheck_tf, TCGv jr)

{

    gen_update_cc_op(s);



    /* If several instructions disable interrupts, only the first does it.  */

    if (inhibit && !(s->flags & HF_INHIBIT_IRQ_MASK)) {

        gen_set_hflag(s, HF_INHIBIT_IRQ_MASK);

    } else {

        gen_reset_hflag(s, HF_INHIBIT_IRQ_MASK);

    }



    if (s->tb->flags & HF_RF_MASK) {

        gen_helper_reset_rf(cpu_env);

    }

    if (s->singlestep_enabled) {

        gen_helper_debug(cpu_env);

    } else if (recheck_tf) {

        gen_helper_rechecking_single_step(cpu_env);

        tcg_gen_exit_tb(0);

    } else if (s->tf) {

        gen_helper_single_step(cpu_env);

    } else if (!TCGV_IS_UNUSED(jr)) {

        TCGv vaddr = tcg_temp_new();



        tcg_gen_add_tl(vaddr, jr, cpu_seg_base[R_CS]);

        tcg_gen_lookup_and_goto_ptr(vaddr);

        tcg_temp_free(vaddr);

    } else {

        tcg_gen_exit_tb(0);

    }

    s->is_jmp = DISAS_TB_JUMP;

}
