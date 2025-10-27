static void qcow_aio_write_cb(void *opaque, int ret)

{

    QCowAIOCB *acb = opaque;

    BlockDriverState *bs = acb->common.bs;

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster;

    const uint8_t *src_buf;

    int n_end;



    acb->hd_aiocb = NULL;



    if (ret >= 0) {

        ret = qcow2_alloc_cluster_link_l2(bs, acb->cluster_offset, &acb->l2meta);

    }



    run_dependent_requests(&acb->l2meta);



    if (ret < 0)

        goto done;



    acb->nb_sectors -= acb->n;

    acb->sector_num += acb->n;

    acb->buf += acb->n * 512;



    if (acb->nb_sectors == 0) {

        /* request completed */

        ret = 0;

        goto done;

    }



    index_in_cluster = acb->sector_num & (s->cluster_sectors - 1);

    n_end = index_in_cluster + acb->nb_sectors;

    if (s->crypt_method &&

        n_end > QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors)

        n_end = QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors;



    acb->cluster_offset = qcow2_alloc_cluster_offset(bs, acb->sector_num << 9,

                                          index_in_cluster,

                                          n_end, &acb->n, &acb->l2meta);



    /* Need to wait for another request? If so, we are done for now. */

    if (!acb->cluster_offset && acb->l2meta.depends_on != NULL) {

        LIST_INSERT_HEAD(&acb->l2meta.depends_on->dependent_requests,

            acb, next_depend);

        return;

    }



    if (!acb->cluster_offset || (acb->cluster_offset & 511) != 0) {

        ret = -EIO;

        goto done;

    }

    if (s->crypt_method) {

        if (!acb->cluster_data) {

            acb->cluster_data = qemu_mallocz(QCOW_MAX_CRYPT_CLUSTERS *

                                             s->cluster_size);

        }

        qcow2_encrypt_sectors(s, acb->sector_num, acb->cluster_data, acb->buf,

                        acb->n, 1, &s->aes_encrypt_key);

        src_buf = acb->cluster_data;

    } else {

        src_buf = acb->buf;

    }

    acb->hd_iov.iov_base = (void *)src_buf;

    acb->hd_iov.iov_len = acb->n * 512;

    qemu_iovec_init_external(&acb->hd_qiov, &acb->hd_iov, 1);

    acb->hd_aiocb = bdrv_aio_writev(s->hd,

                                    (acb->cluster_offset >> 9) + index_in_cluster,

                                    &acb->hd_qiov, acb->n,

                                    qcow_aio_write_cb, acb);

    if (acb->hd_aiocb == NULL)

        goto done;



    return;



done:

    if (acb->qiov->niov > 1)

        qemu_vfree(acb->orig_buf);

    acb->common.cb(acb->common.opaque, ret);

    qemu_aio_release(acb);

}
