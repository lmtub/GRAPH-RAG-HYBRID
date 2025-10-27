void tcg_prologue_init(TCGContext *s)

{

    size_t prologue_size, total_size;

    void *buf0, *buf1;



    /* Put the prologue at the beginning of code_gen_buffer.  */

    buf0 = s->code_gen_buffer;

    s->code_ptr = buf0;

    s->code_buf = buf0;

    s->code_gen_prologue = buf0;



    /* Generate the prologue.  */

    tcg_target_qemu_prologue(s);

    buf1 = s->code_ptr;

    flush_icache_range((uintptr_t)buf0, (uintptr_t)buf1);



    /* Deduct the prologue from the buffer.  */

    prologue_size = tcg_current_code_size(s);

    s->code_gen_ptr = buf1;

    s->code_gen_buffer = buf1;

    s->code_buf = buf1;

    total_size = s->code_gen_buffer_size - prologue_size;

    s->code_gen_buffer_size = total_size;



    /* Compute a high-water mark, at which we voluntarily flush the buffer

       and start over.  The size here is arbitrary, significantly larger

       than we expect the code generation for any one opcode to require.  */

    /* ??? We currently have no good estimate for, or checks in,

       tcg_out_tb_finalize.  If there are quite a lot of guest memory ops,

       the number of out-of-line fragments could be quite high.  In the

       short-term, increase the highwater buffer.  */

    s->code_gen_highwater = s->code_gen_buffer + (total_size - 64*1024);



    tcg_register_jit(s->code_gen_buffer, total_size);



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_OUT_ASM)) {

        qemu_log("PROLOGUE: [size=%zu]\n", prologue_size);

        log_disas(buf0, prologue_size);

        qemu_log("\n");

        qemu_log_flush();

    }

#endif

}
