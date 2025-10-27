static int copy_sectors(BlockDriverState *bs, uint64_t start_sect,

                        uint64_t cluster_offset, int n_start, int n_end)

{

    BDRVQcowState *s = bs->opaque;

    int n, ret;



    n = n_end - n_start;

    if (n <= 0)

        return 0;

    ret = qcow_read(bs, start_sect + n_start, s->cluster_data, n);

    if (ret < 0)

        return ret;

    if (s->crypt_method) {

        qcow2_encrypt_sectors(s, start_sect + n_start,

                        s->cluster_data,

                        s->cluster_data, n, 1,

                        &s->aes_encrypt_key);

    }

    ret = bdrv_write(s->hd, (cluster_offset >> 9) + n_start,

                     s->cluster_data, n);

    if (ret < 0)

        return ret;

    return 0;

}
