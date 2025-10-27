static inline void terminate_compression_threads(void)

{

    int idx, thread_count;



    thread_count = migrate_compress_threads();

    quit_comp_thread = true;

    for (idx = 0; idx < thread_count; idx++) {

        qemu_mutex_lock(&comp_param[idx].mutex);

        qemu_cond_signal(&comp_param[idx].cond);

        qemu_mutex_unlock(&comp_param[idx].mutex);

    }

}
