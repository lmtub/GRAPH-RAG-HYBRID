vdi_co_preadv(BlockDriverState *bs, uint64_t offset, uint64_t bytes,

              QEMUIOVector *qiov, int flags)

{

    BDRVVdiState *s = bs->opaque;

    QEMUIOVector local_qiov;

    uint32_t bmap_entry;

    uint32_t block_index;

    uint32_t offset_in_block;

    uint32_t n_bytes;

    uint64_t bytes_done = 0;

    int ret = 0;



    logout("\n");



    qemu_iovec_init(&local_qiov, qiov->niov);



    while (ret >= 0 && bytes > 0) {

        block_index = offset / s->block_size;

        offset_in_block = offset % s->block_size;

        n_bytes = MIN(bytes, s->block_size - offset_in_block);



        logout("will read %u bytes starting at offset %" PRIu64 "\n",

               n_bytes, offset);



        /* prepare next AIO request */

        bmap_entry = le32_to_cpu(s->bmap[block_index]);

        if (!VDI_IS_ALLOCATED(bmap_entry)) {

            /* Block not allocated, return zeros, no need to wait. */

            qemu_iovec_memset(qiov, bytes_done, 0, n_bytes);

            ret = 0;

        } else {

            uint64_t data_offset = s->header.offset_data +

                                   (uint64_t)bmap_entry * s->block_size +

                                   offset_in_block;



            qemu_iovec_reset(&local_qiov);

            qemu_iovec_concat(&local_qiov, qiov, bytes_done, n_bytes);



            ret = bdrv_co_preadv(bs->file->bs, data_offset, n_bytes,

                                 &local_qiov, 0);

        }

        logout("%u bytes read\n", n_bytes);



        bytes -= n_bytes;

        offset += n_bytes;

        bytes_done += n_bytes;

    }



    qemu_iovec_destroy(&local_qiov);



    return ret;

}
