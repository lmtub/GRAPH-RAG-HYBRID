static int coroutine_fn bdrv_aligned_preadv(BlockDriverState *bs,

    BdrvTrackedRequest *req, int64_t offset, unsigned int bytes,

    int64_t align, QEMUIOVector *qiov, int flags)

{

    int64_t total_bytes, max_bytes;

    int ret = 0;

    uint64_t bytes_remaining = bytes;

    int max_transfer;



    assert(is_power_of_2(align));

    assert((offset & (align - 1)) == 0);

    assert((bytes & (align - 1)) == 0);

    assert(!qiov || bytes == qiov->size);

    assert((bs->open_flags & BDRV_O_NO_IO) == 0);

    max_transfer = QEMU_ALIGN_DOWN(MIN_NON_ZERO(bs->bl.max_transfer, INT_MAX),

                                   align);



    /* TODO: We would need a per-BDS .supported_read_flags and

     * potential fallback support, if we ever implement any read flags

     * to pass through to drivers.  For now, there aren't any

     * passthrough flags.  */

    assert(!(flags & ~(BDRV_REQ_NO_SERIALISING | BDRV_REQ_COPY_ON_READ)));



    /* Handle Copy on Read and associated serialisation */

    if (flags & BDRV_REQ_COPY_ON_READ) {

        /* If we touch the same cluster it counts as an overlap.  This

         * guarantees that allocating writes will be serialized and not race

         * with each other for the same cluster.  For example, in copy-on-read

         * it ensures that the CoR read and write operations are atomic and

         * guest writes cannot interleave between them. */

        mark_request_serialising(req, bdrv_get_cluster_size(bs));

    }



    if (!(flags & BDRV_REQ_NO_SERIALISING)) {

        wait_serialising_requests(req);

    }



    if (flags & BDRV_REQ_COPY_ON_READ) {

        int64_t start_sector = offset >> BDRV_SECTOR_BITS;

        int64_t end_sector = DIV_ROUND_UP(offset + bytes, BDRV_SECTOR_SIZE);

        unsigned int nb_sectors = end_sector - start_sector;

        int pnum;



        ret = bdrv_is_allocated(bs, start_sector, nb_sectors, &pnum);

        if (ret < 0) {

            goto out;

        }



        if (!ret || pnum != nb_sectors) {

            ret = bdrv_co_do_copy_on_readv(bs, offset, bytes, qiov);

            goto out;

        }

    }



    /* Forward the request to the BlockDriver, possibly fragmenting it */

    total_bytes = bdrv_getlength(bs);

    if (total_bytes < 0) {

        ret = total_bytes;

        goto out;

    }



    max_bytes = ROUND_UP(MAX(0, total_bytes - offset), align);

    if (bytes <= max_bytes && bytes <= max_transfer) {

        ret = bdrv_driver_preadv(bs, offset, bytes, qiov, 0);

        goto out;

    }



    while (bytes_remaining) {

        int num;



        if (max_bytes) {

            QEMUIOVector local_qiov;



            num = MIN(bytes_remaining, MIN(max_bytes, max_transfer));

            assert(num);

            qemu_iovec_init(&local_qiov, qiov->niov);

            qemu_iovec_concat(&local_qiov, qiov, bytes - bytes_remaining, num);



            ret = bdrv_driver_preadv(bs, offset + bytes - bytes_remaining,

                                     num, &local_qiov, 0);

            max_bytes -= num;

            qemu_iovec_destroy(&local_qiov);

        } else {

            num = bytes_remaining;

            ret = qemu_iovec_memset(qiov, bytes - bytes_remaining, 0,

                                    bytes_remaining);

        }

        if (ret < 0) {

            goto out;

        }

        bytes_remaining -= num;

    }



out:

    return ret < 0 ? ret : 0;

}
