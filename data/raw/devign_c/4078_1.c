create_iovec(BlockBackend *blk, QEMUIOVector *qiov, char **argv, int nr_iov,

             int pattern)

{

    size_t *sizes = g_new0(size_t, nr_iov);

    size_t count = 0;

    void *buf = NULL;

    void *p;

    int i;



    for (i = 0; i < nr_iov; i++) {

        char *arg = argv[i];

        int64_t len;



        len = cvtnum(arg);

        if (len < 0) {

            print_cvtnum_err(len, arg);

            goto fail;

        }



        if (len > SIZE_MAX) {

            printf("Argument '%s' exceeds maximum size %llu\n", arg,

                   (unsigned long long)SIZE_MAX);

            goto fail;

        }



        sizes[i] = len;

        count += len;

    }



    qemu_iovec_init(qiov, nr_iov);



    buf = p = qemu_io_alloc(blk, count, pattern);



    for (i = 0; i < nr_iov; i++) {

        qemu_iovec_add(qiov, p, sizes[i]);

        p += sizes[i];

    }



fail:

    g_free(sizes);

    return buf;

}
