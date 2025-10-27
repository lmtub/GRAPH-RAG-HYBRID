static int coroutine_fn bdrv_co_do_pwrite_zeroes(BlockDriverState *bs,

    int64_t offset, int bytes, BdrvRequestFlags flags)

{

    BlockDriver *drv = bs->drv;

    QEMUIOVector qiov;

    struct iovec iov = {0};

    int ret = 0;

    bool need_flush = false;

    int head = 0;

    int tail = 0;



    int max_write_zeroes = MIN_NON_ZERO(bs->bl.max_pwrite_zeroes, INT_MAX);

    int alignment = MAX(bs->bl.pwrite_zeroes_alignment,

                        bs->bl.request_alignment);

    int max_transfer = MIN_NON_ZERO(bs->bl.max_transfer,

                                    MAX_WRITE_ZEROES_BOUNCE_BUFFER);



    assert(alignment % bs->bl.request_alignment == 0);

    head = offset % alignment;

    tail = (offset + bytes) % alignment;

    max_write_zeroes = QEMU_ALIGN_DOWN(max_write_zeroes, alignment);

    assert(max_write_zeroes >= bs->bl.request_alignment);



    while (bytes > 0 && !ret) {

        int num = bytes;



        /* Align request.  Block drivers can expect the "bulk" of the request

         * to be aligned, and that unaligned requests do not cross cluster

         * boundaries.

         */

        if (head) {

            /* Make a small request up to the first aligned sector. For

             * convenience, limit this request to max_transfer even if

             * we don't need to fall back to writes.  */

            num = MIN(MIN(bytes, max_transfer), alignment - head);

            head = (head + num) % alignment;

            assert(num < max_write_zeroes);

        } else if (tail && num > alignment) {

            /* Shorten the request to the last aligned sector.  */

            num -= tail;

        }



        /* limit request size */

        if (num > max_write_zeroes) {

            num = max_write_zeroes;

        }



        ret = -ENOTSUP;

        /* First try the efficient write zeroes operation */

        if (drv->bdrv_co_pwrite_zeroes) {

            ret = drv->bdrv_co_pwrite_zeroes(bs, offset, num,

                                             flags & bs->supported_zero_flags);

            if (ret != -ENOTSUP && (flags & BDRV_REQ_FUA) &&

                !(bs->supported_zero_flags & BDRV_REQ_FUA)) {

                need_flush = true;

            }

        } else {

            assert(!bs->supported_zero_flags);

        }



        if (ret == -ENOTSUP) {

            /* Fall back to bounce buffer if write zeroes is unsupported */

            BdrvRequestFlags write_flags = flags & ~BDRV_REQ_ZERO_WRITE;



            if ((flags & BDRV_REQ_FUA) &&

                !(bs->supported_write_flags & BDRV_REQ_FUA)) {

                /* No need for bdrv_driver_pwrite() to do a fallback

                 * flush on each chunk; use just one at the end */

                write_flags &= ~BDRV_REQ_FUA;

                need_flush = true;

            }

            num = MIN(num, max_transfer);

            iov.iov_len = num;

            if (iov.iov_base == NULL) {

                iov.iov_base = qemu_try_blockalign(bs, num);

                if (iov.iov_base == NULL) {

                    ret = -ENOMEM;

                    goto fail;

                }

                memset(iov.iov_base, 0, num);

            }

            qemu_iovec_init_external(&qiov, &iov, 1);



            ret = bdrv_driver_pwritev(bs, offset, num, &qiov, write_flags);



            /* Keep bounce buffer around if it is big enough for all

             * all future requests.

             */

            if (num < max_transfer) {

                qemu_vfree(iov.iov_base);

                iov.iov_base = NULL;

            }

        }



        offset += num;

        bytes -= num;

    }



fail:

    if (ret == 0 && need_flush) {

        ret = bdrv_co_flush(bs);

    }

    qemu_vfree(iov.iov_base);

    return ret;

}
