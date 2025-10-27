static void test_blk_read(BlockBackend *blk, long pattern,

                          int64_t pattern_offset, int64_t pattern_count,

                          int64_t offset, int64_t count,

                          bool expect_failed)

{

    void *pattern_buf = NULL;

    QEMUIOVector qiov;

    void *cmp_buf = NULL;

    int async_ret = NOT_DONE;



    if (pattern) {

        cmp_buf = g_malloc(pattern_count);

        memset(cmp_buf, pattern, pattern_count);

    }



    pattern_buf = g_malloc(count);

    if (pattern) {

        memset(pattern_buf, pattern, count);

    } else {

        memset(pattern_buf, 0x00, count);

    }



    qemu_iovec_init(&qiov, 1);

    qemu_iovec_add(&qiov, pattern_buf, count);



    blk_aio_preadv(blk, offset, &qiov, 0, blk_rw_done, &async_ret);

    while (async_ret == NOT_DONE) {

        main_loop_wait(false);

    }



    if (expect_failed) {

        g_assert(async_ret != 0);

    } else {

        g_assert(async_ret == 0);

        if (pattern) {

            g_assert(memcmp(pattern_buf + pattern_offset,

                            cmp_buf, pattern_count) <= 0);

        }

    }



    g_free(pattern_buf);



}