static int coroutine_fn backup_do_cow(BlockDriverState *bs,

                                      int64_t sector_num, int nb_sectors,

                                      bool *error_is_read,

                                      bool is_write_notifier)

{

    BackupBlockJob *job = (BackupBlockJob *)bs->job;

    CowRequest cow_request;

    struct iovec iov;

    QEMUIOVector bounce_qiov;

    void *bounce_buffer = NULL;

    int ret = 0;

    int64_t sectors_per_cluster = cluster_size_sectors(job);

    int64_t start, end;

    int n;



    qemu_co_rwlock_rdlock(&job->flush_rwlock);



    start = sector_num / sectors_per_cluster;

    end = DIV_ROUND_UP(sector_num + nb_sectors, sectors_per_cluster);



    trace_backup_do_cow_enter(job, start, sector_num, nb_sectors);



    wait_for_overlapping_requests(job, start, end);

    cow_request_begin(&cow_request, job, start, end);



    for (; start < end; start++) {

        if (hbitmap_get(job->bitmap, start)) {

            trace_backup_do_cow_skip(job, start);

            continue; /* already copied */

        }



        trace_backup_do_cow_process(job, start);



        n = MIN(sectors_per_cluster,

                job->common.len / BDRV_SECTOR_SIZE -

                start * sectors_per_cluster);



        if (!bounce_buffer) {

            bounce_buffer = qemu_blockalign(bs, job->cluster_size);

        }

        iov.iov_base = bounce_buffer;

        iov.iov_len = n * BDRV_SECTOR_SIZE;

        qemu_iovec_init_external(&bounce_qiov, &iov, 1);



        if (is_write_notifier) {

            ret = bdrv_co_readv_no_serialising(bs,

                                           start * sectors_per_cluster,

                                           n, &bounce_qiov);

        } else {

            ret = bdrv_co_readv(bs, start * sectors_per_cluster, n,

                                &bounce_qiov);

        }

        if (ret < 0) {

            trace_backup_do_cow_read_fail(job, start, ret);

            if (error_is_read) {

                *error_is_read = true;

            }

            goto out;

        }



        if (buffer_is_zero(iov.iov_base, iov.iov_len)) {

            ret = bdrv_co_write_zeroes(job->target,

                                       start * sectors_per_cluster,

                                       n, BDRV_REQ_MAY_UNMAP);

        } else {

            ret = bdrv_co_writev(job->target,

                                 start * sectors_per_cluster, n,

                                 &bounce_qiov);

        }

        if (ret < 0) {

            trace_backup_do_cow_write_fail(job, start, ret);

            if (error_is_read) {

                *error_is_read = false;

            }

            goto out;

        }



        hbitmap_set(job->bitmap, start, 1);



        /* Publish progress, guest I/O counts as progress too.  Note that the

         * offset field is an opaque progress value, it is not a disk offset.

         */

        job->sectors_read += n;

        job->common.offset += n * BDRV_SECTOR_SIZE;

    }



out:

    if (bounce_buffer) {

        qemu_vfree(bounce_buffer);

    }



    cow_request_end(&cow_request);



    trace_backup_do_cow_return(job, sector_num, nb_sectors, ret);



    qemu_co_rwlock_unlock(&job->flush_rwlock);



    return ret;

}
