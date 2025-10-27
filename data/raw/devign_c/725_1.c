static void do_tb_flush(CPUState *cpu, void *data)

{

    unsigned tb_flush_req = (unsigned) (uintptr_t) data;



    tb_lock();



    /* If it's already been done on request of another CPU,

     * just retry.

     */

    if (tcg_ctx.tb_ctx.tb_flush_count != tb_flush_req) {

        goto done;

    }



#if defined(DEBUG_FLUSH)

    printf("qemu: flush code_size=%ld nb_tbs=%d avg_tb_size=%ld\n",

           (unsigned long)(tcg_ctx.code_gen_ptr - tcg_ctx.code_gen_buffer),

           tcg_ctx.tb_ctx.nb_tbs, tcg_ctx.tb_ctx.nb_tbs > 0 ?

           ((unsigned long)(tcg_ctx.code_gen_ptr - tcg_ctx.code_gen_buffer)) /

           tcg_ctx.tb_ctx.nb_tbs : 0);

#endif

    if ((unsigned long)(tcg_ctx.code_gen_ptr - tcg_ctx.code_gen_buffer)

        > tcg_ctx.code_gen_buffer_size) {

        cpu_abort(cpu, "Internal error: code buffer overflow\n");

    }



    CPU_FOREACH(cpu) {

        int i;



        for (i = 0; i < TB_JMP_CACHE_SIZE; ++i) {

            atomic_set(&cpu->tb_jmp_cache[i], NULL);

        }

    }



    tcg_ctx.tb_ctx.nb_tbs = 0;

    qht_reset_size(&tcg_ctx.tb_ctx.htable, CODE_GEN_HTABLE_SIZE);

    page_flush_tb();



    tcg_ctx.code_gen_ptr = tcg_ctx.code_gen_buffer;

    /* XXX: flush processor icache at this point if cache flush is

       expensive */

    atomic_mb_set(&tcg_ctx.tb_ctx.tb_flush_count,

                  tcg_ctx.tb_ctx.tb_flush_count + 1);



done:

    tb_unlock();

}
