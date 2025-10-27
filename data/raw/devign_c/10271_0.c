void tcg_region_reset_all(void)

{

    unsigned int i;



    qemu_mutex_lock(&region.lock);

    region.current = 0;

    region.agg_size_full = 0;



    for (i = 0; i < n_tcg_ctxs; i++) {

        bool err = tcg_region_initial_alloc__locked(tcg_ctxs[i]);



        g_assert(!err);

    }

    qemu_mutex_unlock(&region.lock);

}
