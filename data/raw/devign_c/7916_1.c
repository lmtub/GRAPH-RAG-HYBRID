static void qcow_aio_write_cb(void *opaque, int ret)

{

    QCowAIOCB *acb = opaque;

    BlockDriverState *bs = acb->common.bs;

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster;

    uint64_t cluster_offset;

    const uint8_t *src_buf;

    int n_end;



    acb->hd_aiocb = NULL;



    if (ret < 0) {

    fail:

        acb->common.cb(acb->common.opaque, ret);

        qemu_aio_release(acb);

        return;

    }



    acb->nb_sectors -= acb->n;

    acb->sector_num += acb->n;

    acb->buf += acb->n * 512;



    if (acb->nb_sectors == 0) {

        /* request completed */

        acb->common.cb(acb->common.opaque, 0);

        qemu_aio_release(acb);

        return;

    }



    index_in_cluster = acb->sector_num & (s->cluster_sectors - 1);

    n_end = index_in_cluster + acb->nb_sectors;

    if (s->crypt_method &&

        n_end > QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors)

        n_end = QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors;



    cluster_offset = alloc_cluster_offset(bs, acb->sector_num << 9,

                                          index_in_cluster,

                                          n_end, &acb->n);

    if (!cluster_offset || (cluster_offset & 511) != 0) {

        ret = -EIO;

        goto fail;

    }

    if (s->crypt_method) {

        if (!acb->cluster_data) {

            acb->cluster_data = qemu_mallocz(QCOW_MAX_CRYPT_CLUSTERS *

                                             s->cluster_size);

            if (!acb->cluster_data) {

                ret = -ENOMEM;

                goto fail;

            }

        }

        encrypt_sectors(s, acb->sector_num, acb->cluster_data, acb->buf,

                        acb->n, 1, &s->aes_encrypt_key);

        src_buf = acb->cluster_data;

    } else {

        src_buf = acb->buf;

    }

    acb->hd_aiocb = bdrv_aio_write(s->hd,

                                   (cluster_offset >> 9) + index_in_cluster,

                                   src_buf, acb->n,

                                   qcow_aio_write_cb, acb);

    if (acb->hd_aiocb == NULL)

        goto fail;

}
