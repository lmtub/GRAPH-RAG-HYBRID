void tb_flush(CPUState *cpu)

{




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


    tcg_ctx.tb_ctx.nb_tbs = 0;



    CPU_FOREACH(cpu) {

        memset(cpu->tb_jmp_cache, 0, sizeof(cpu->tb_jmp_cache));

        cpu->tb_flushed = true;




    qht_reset_size(&tcg_ctx.tb_ctx.htable, CODE_GEN_HTABLE_SIZE);

    page_flush_tb();



    tcg_ctx.code_gen_ptr = tcg_ctx.code_gen_buffer;

    /* XXX: flush processor icache at this point if cache flush is

       expensive */

    tcg_ctx.tb_ctx.tb_flush_count++;
