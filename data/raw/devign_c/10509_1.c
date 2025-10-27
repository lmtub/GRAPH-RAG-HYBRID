static void decompress_data_with_multi_threads(QEMUFile *f,

                                               void *host, int len)

{

    int idx, thread_count;



    thread_count = migrate_decompress_threads();

    while (true) {

        for (idx = 0; idx < thread_count; idx++) {

            if (!decomp_param[idx].start) {

                qemu_get_buffer(f, decomp_param[idx].compbuf, len);

                decomp_param[idx].des = host;

                decomp_param[idx].len = len;

                start_decompression(&decomp_param[idx]);

                break;

            }

        }

        if (idx < thread_count) {

            break;

        }

    }

}
