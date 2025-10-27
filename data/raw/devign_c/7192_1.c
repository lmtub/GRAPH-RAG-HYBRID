static coroutine_fn int qcow2_co_pwritev(BlockDriverState *bs, uint64_t offset,

                                         uint64_t bytes, QEMUIOVector *qiov,

                                         int flags)

{

    BDRVQcow2State *s = bs->opaque;

    int offset_in_cluster;

    int ret;

    unsigned int cur_bytes; /* number of sectors in current iteration */

    uint64_t cluster_offset;

    QEMUIOVector hd_qiov;

    uint64_t bytes_done = 0;

    uint8_t *cluster_data = NULL;

    QCowL2Meta *l2meta = NULL;



    trace_qcow2_writev_start_req(qemu_coroutine_self(), offset, bytes);



    qemu_iovec_init(&hd_qiov, qiov->niov);



    s->cluster_cache_offset = -1; /* disable compressed cache */



    qemu_co_mutex_lock(&s->lock);



    while (bytes != 0) {



        l2meta = NULL;



        trace_qcow2_writev_start_part(qemu_coroutine_self());

        offset_in_cluster = offset_into_cluster(s, offset);

        cur_bytes = MIN(bytes, INT_MAX);

        if (bs->encrypted) {

            cur_bytes = MIN(cur_bytes,

                            QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size

                            - offset_in_cluster);

        }



        ret = qcow2_alloc_cluster_offset(bs, offset, &cur_bytes,

                                         &cluster_offset, &l2meta);

        if (ret < 0) {

            goto fail;

        }



        assert((cluster_offset & 511) == 0);



        qemu_iovec_reset(&hd_qiov);

        qemu_iovec_concat(&hd_qiov, qiov, bytes_done, cur_bytes);



        if (bs->encrypted) {

            Error *err = NULL;

            assert(s->crypto);

            if (!cluster_data) {

                cluster_data = qemu_try_blockalign(bs->file->bs,

                                                   QCOW_MAX_CRYPT_CLUSTERS

                                                   * s->cluster_size);

                if (cluster_data == NULL) {

                    ret = -ENOMEM;

                    goto fail;

                }

            }



            assert(hd_qiov.size <=

                   QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size);

            qemu_iovec_to_buf(&hd_qiov, 0, cluster_data, hd_qiov.size);



            if (qcrypto_block_encrypt(s->crypto, offset >> BDRV_SECTOR_BITS,

                                      cluster_data,

                                      cur_bytes, &err) < 0) {

                error_free(err);

                ret = -EIO;

                goto fail;

            }



            qemu_iovec_reset(&hd_qiov);

            qemu_iovec_add(&hd_qiov, cluster_data, cur_bytes);

        }



        ret = qcow2_pre_write_overlap_check(bs, 0,

                cluster_offset + offset_in_cluster, cur_bytes);

        if (ret < 0) {

            goto fail;

        }



        /* If we need to do COW, check if it's possible to merge the

         * writing of the guest data together with that of the COW regions.

         * If it's not possible (or not necessary) then write the

         * guest data now. */

        if (!merge_cow(offset, cur_bytes, &hd_qiov, l2meta)) {

            qemu_co_mutex_unlock(&s->lock);

            BLKDBG_EVENT(bs->file, BLKDBG_WRITE_AIO);

            trace_qcow2_writev_data(qemu_coroutine_self(),

                                    cluster_offset + offset_in_cluster);

            ret = bdrv_co_pwritev(bs->file,

                                  cluster_offset + offset_in_cluster,

                                  cur_bytes, &hd_qiov, 0);

            qemu_co_mutex_lock(&s->lock);

            if (ret < 0) {

                goto fail;

            }

        }



        while (l2meta != NULL) {

            QCowL2Meta *next;



            ret = qcow2_alloc_cluster_link_l2(bs, l2meta);

            if (ret < 0) {

                goto fail;

            }



            /* Take the request off the list of running requests */

            if (l2meta->nb_clusters != 0) {

                QLIST_REMOVE(l2meta, next_in_flight);

            }



            qemu_co_queue_restart_all(&l2meta->dependent_requests);



            next = l2meta->next;

            g_free(l2meta);

            l2meta = next;

        }



        bytes -= cur_bytes;

        offset += cur_bytes;

        bytes_done += cur_bytes;

        trace_qcow2_writev_done_part(qemu_coroutine_self(), cur_bytes);

    }

    ret = 0;



fail:

    qemu_co_mutex_unlock(&s->lock);



    while (l2meta != NULL) {

        QCowL2Meta *next;



        if (l2meta->nb_clusters != 0) {

            QLIST_REMOVE(l2meta, next_in_flight);

        }

        qemu_co_queue_restart_all(&l2meta->dependent_requests);



        next = l2meta->next;

        g_free(l2meta);

        l2meta = next;

    }



    qemu_iovec_destroy(&hd_qiov);

    qemu_vfree(cluster_data);

    trace_qcow2_writev_done_req(qemu_coroutine_self(), ret);



    return ret;

}
