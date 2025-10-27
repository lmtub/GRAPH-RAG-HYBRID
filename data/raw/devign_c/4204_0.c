static void *do_data_compress(void *opaque)

{

    CompressParam *param = opaque;



    while (!quit_comp_thread) {

        qemu_mutex_lock(&param->mutex);

        /* Re-check the quit_comp_thread in case of

         * terminate_compression_threads is called just before

         * qemu_mutex_lock(&param->mutex) and after

         * while(!quit_comp_thread), re-check it here can make

         * sure the compression thread terminate as expected.

         */

        while (!param->start && !quit_comp_thread) {

            qemu_cond_wait(&param->cond, &param->mutex);

        }

        if (!quit_comp_thread) {

            do_compress_ram_page(param);

        }

        param->start = false;

        qemu_mutex_unlock(&param->mutex);



        qemu_mutex_lock(comp_done_lock);

        param->done = true;

        qemu_cond_signal(comp_done_cond);

        qemu_mutex_unlock(comp_done_lock);

    }



    return NULL;

}
