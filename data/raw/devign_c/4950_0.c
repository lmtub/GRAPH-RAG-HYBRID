void migrate_decompress_threads_create(void)

{

    int i, thread_count;



    thread_count = migrate_decompress_threads();

    decompress_threads = g_new0(QemuThread, thread_count);

    decomp_param = g_new0(DecompressParam, thread_count);

    quit_decomp_thread = false;

    qemu_mutex_init(&decomp_done_lock);

    qemu_cond_init(&decomp_done_cond);

    for (i = 0; i < thread_count; i++) {

        qemu_mutex_init(&decomp_param[i].mutex);

        qemu_cond_init(&decomp_param[i].cond);

        decomp_param[i].compbuf = g_malloc0(compressBound(TARGET_PAGE_SIZE));

        decomp_param[i].done = true;

        qemu_thread_create(decompress_threads + i, "decompress",

                           do_data_decompress, decomp_param + i,

                           QEMU_THREAD_JOINABLE);

    }

}
