static void qcow2_aio_read_cb(void *opaque, int ret)

{

    QCowAIOCB *acb = opaque;

    BlockDriverState *bs = acb->common.bs;

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster, n1;



    acb->hd_aiocb = NULL;

    if (ret < 0)

        goto done;



    /* post process the read buffer */

    if (!acb->cluster_offset) {

        /* nothing to do */

    } else if (acb->cluster_offset & QCOW_OFLAG_COMPRESSED) {

        /* nothing to do */

    } else {

        if (s->crypt_method) {

            qcow2_encrypt_sectors(s, acb->sector_num,  acb->cluster_data,

                acb->cluster_data, acb->cur_nr_sectors, 0, &s->aes_decrypt_key);

            qemu_iovec_reset(&acb->hd_qiov);

            qemu_iovec_copy(&acb->hd_qiov, acb->qiov, acb->bytes_done,

                acb->cur_nr_sectors * 512);

            qemu_iovec_from_buffer(&acb->hd_qiov, acb->cluster_data,

                512 * acb->cur_nr_sectors);

        }

    }



    acb->remaining_sectors -= acb->cur_nr_sectors;

    acb->sector_num += acb->cur_nr_sectors;

    acb->bytes_done += acb->cur_nr_sectors * 512;



    if (acb->remaining_sectors == 0) {

        /* request completed */

        ret = 0;

        goto done;

    }



    /* prepare next AIO request */

    acb->cur_nr_sectors = acb->remaining_sectors;

    if (s->crypt_method) {

        acb->cur_nr_sectors = MIN(acb->cur_nr_sectors,

            QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors);

    }



    ret = qcow2_get_cluster_offset(bs, acb->sector_num << 9,

        &acb->cur_nr_sectors, &acb->cluster_offset);

    if (ret < 0) {

        goto done;

    }



    index_in_cluster = acb->sector_num & (s->cluster_sectors - 1);



    qemu_iovec_reset(&acb->hd_qiov);

    qemu_iovec_copy(&acb->hd_qiov, acb->qiov, acb->bytes_done,

        acb->cur_nr_sectors * 512);



    if (!acb->cluster_offset) {



        if (bs->backing_hd) {

            /* read from the base image */

            n1 = qcow2_backing_read1(bs->backing_hd, &acb->hd_qiov,

                acb->sector_num, acb->cur_nr_sectors);

            if (n1 > 0) {

                BLKDBG_EVENT(bs->file, BLKDBG_READ_BACKING_AIO);

                acb->hd_aiocb = bdrv_aio_readv(bs->backing_hd, acb->sector_num,

                                    &acb->hd_qiov, acb->cur_nr_sectors,

				    qcow2_aio_read_cb, acb);

                if (acb->hd_aiocb == NULL)

                    goto done;

            } else {

                ret = qcow2_schedule_bh(qcow2_aio_read_bh, acb);

                if (ret < 0)

                    goto done;

            }

        } else {

            /* Note: in this case, no need to wait */

            qemu_iovec_memset(&acb->hd_qiov, 0, 512 * acb->cur_nr_sectors);

            ret = qcow2_schedule_bh(qcow2_aio_read_bh, acb);

            if (ret < 0)

                goto done;

        }

    } else if (acb->cluster_offset & QCOW_OFLAG_COMPRESSED) {

        /* add AIO support for compressed blocks ? */

        if (qcow2_decompress_cluster(bs, acb->cluster_offset) < 0)

            goto done;



        qemu_iovec_from_buffer(&acb->hd_qiov,

            s->cluster_cache + index_in_cluster * 512,

            512 * acb->cur_nr_sectors);



        ret = qcow2_schedule_bh(qcow2_aio_read_bh, acb);

        if (ret < 0)

            goto done;

    } else {

        if ((acb->cluster_offset & 511) != 0) {

            ret = -EIO;

            goto done;

        }



        if (s->crypt_method) {

            /*

             * For encrypted images, read everything into a temporary

             * contiguous buffer on which the AES functions can work.

             */

            if (!acb->cluster_data) {

                acb->cluster_data =

                    qemu_mallocz(QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size);

            }



            assert(acb->cur_nr_sectors <=

                QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors);

            qemu_iovec_reset(&acb->hd_qiov);

            qemu_iovec_add(&acb->hd_qiov, acb->cluster_data,

                512 * acb->cur_nr_sectors);

        }



        BLKDBG_EVENT(bs->file, BLKDBG_READ_AIO);

        acb->hd_aiocb = bdrv_aio_readv(bs->file,

                            (acb->cluster_offset >> 9) + index_in_cluster,

                            &acb->hd_qiov, acb->cur_nr_sectors,

                            qcow2_aio_read_cb, acb);

        if (acb->hd_aiocb == NULL) {

            ret = -EIO;

            goto done;

        }

    }



    return;

done:

    acb->common.cb(acb->common.opaque, ret);

    qemu_iovec_destroy(&acb->hd_qiov);

    qemu_aio_release(acb);

}
