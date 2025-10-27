size_t tcg_code_size(void)

{

    unsigned int i;

    size_t total;



    qemu_mutex_lock(&region.lock);

    total = region.agg_size_full;

    for (i = 0; i < n_tcg_ctxs; i++) {

        const TCGContext *s = tcg_ctxs[i];

        size_t size;



        size = atomic_read(&s->code_gen_ptr) - s->code_gen_buffer;

        g_assert(size <= s->code_gen_buffer_size);

        total += size;

    }

    qemu_mutex_unlock(&region.lock);

    return total;

}
