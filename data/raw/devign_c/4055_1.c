void tcg_prologue_init(TCGContext *s)
{
    /* init global prologue and epilogue */
    s->code_buf = s->code_gen_prologue;
    s->code_ptr = s->code_buf;
    tcg_target_qemu_prologue(s);
    flush_icache_range((tcg_target_ulong)s->code_buf,
                       (tcg_target_ulong)s->code_ptr);