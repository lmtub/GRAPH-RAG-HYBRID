static int qcow2_co_writev(BlockDriverState *bs,

                           int64_t sector_num,

                           int remaining_sectors,

                           QEMUIOVector *qiov)

{

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster;

    int n_end;

    int ret;

    int cur_nr_sectors; /* number of sectors in current iteration */

    QCowL2Meta l2meta;

    uint64_t cluster_offset;

    QEMUIOVector hd_qiov;

    uint64_t bytes_done = 0;

    uint8_t *cluster_data = NULL;



    l2meta.nb_clusters = 0;

    qemu_co_queue_init(&l2meta.dependent_requests);



    qemu_iovec_init(&hd_qiov, qiov->niov);



    s->cluster_cache_offset = -1; /* disable compressed cache */



    qemu_co_mutex_lock(&s->lock);



    while (remaining_sectors != 0) {



        index_in_cluster = sector_num & (s->cluster_sectors - 1);

        n_end = index_in_cluster + remaining_sectors;

        if (s->crypt_method &&

            n_end > QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors) {

            n_end = QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors;

        }



        ret = qcow2_alloc_cluster_offset(bs, sector_num << 9,

            index_in_cluster, n_end, &cur_nr_sectors, &l2meta);

        if (ret < 0) {

            goto fail;

        }



        cluster_offset = l2meta.cluster_offset;

        assert((cluster_offset & 511) == 0);



        qemu_iovec_reset(&hd_qiov);

        qemu_iovec_copy(&hd_qiov, qiov, bytes_done,

            cur_nr_sectors * 512);



        if (s->crypt_method) {

            if (!cluster_data) {

                cluster_data = g_malloc0(QCOW_MAX_CRYPT_CLUSTERS *

                                                 s->cluster_size);

            }



            assert(hd_qiov.size <=

                   QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size);

            qemu_iovec_to_buffer(&hd_qiov, cluster_data);



            qcow2_encrypt_sectors(s, sector_num, cluster_data,

                cluster_data, cur_nr_sectors, 1, &s->aes_encrypt_key);



            qemu_iovec_reset(&hd_qiov);

            qemu_iovec_add(&hd_qiov, cluster_data,

                cur_nr_sectors * 512);

        }



        BLKDBG_EVENT(bs->file, BLKDBG_WRITE_AIO);

        qemu_co_mutex_unlock(&s->lock);

        ret = bdrv_co_writev(bs->file,

                             (cluster_offset >> 9) + index_in_cluster,

                             cur_nr_sectors, &hd_qiov);

        qemu_co_mutex_lock(&s->lock);

        if (ret < 0) {

            goto fail;

        }



        ret = qcow2_alloc_cluster_link_l2(bs, &l2meta);



        run_dependent_requests(s, &l2meta);



        if (ret < 0) {

            goto fail;

        }



        remaining_sectors -= cur_nr_sectors;

        sector_num += cur_nr_sectors;

        bytes_done += cur_nr_sectors * 512;

    }

    ret = 0;



fail:

    qemu_co_mutex_unlock(&s->lock);



    qemu_iovec_destroy(&hd_qiov);




    return ret;

}