static int copy_sectors(BlockDriverState *bs, uint64_t start_sect,

                        uint64_t cluster_offset, int n_start, int n_end)

{

    BDRVQcowState *s = bs->opaque;

    int n, ret;

    void *buf;



    /*

     * If this is the last cluster and it is only partially used, we must only

     * copy until the end of the image, or bdrv_check_request will fail for the

     * bdrv_read/write calls below.

     */

    if (start_sect + n_end > bs->total_sectors) {

        n_end = bs->total_sectors - start_sect;

    }



    n = n_end - n_start;

    if (n <= 0) {

        return 0;

    }



    buf = qemu_blockalign(bs, n * BDRV_SECTOR_SIZE);



    BLKDBG_EVENT(bs->file, BLKDBG_COW_READ);

    ret = bdrv_read(bs, start_sect + n_start, buf, n);

    if (ret < 0) {

        goto out;

    }



    if (s->crypt_method) {

        qcow2_encrypt_sectors(s, start_sect + n_start,

                        buf, buf, n, 1,

                        &s->aes_encrypt_key);

    }



    BLKDBG_EVENT(bs->file, BLKDBG_COW_WRITE);

    ret = bdrv_write(bs->file, (cluster_offset >> 9) + n_start, buf, n);

    if (ret < 0) {

        goto out;

    }



    ret = 0;

out:

    qemu_vfree(buf);

    return ret;

}
