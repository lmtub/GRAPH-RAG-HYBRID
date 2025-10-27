static int coroutine_fn backup_run_incremental(BackupBlockJob *job)

{

    bool error_is_read;

    int ret = 0;

    int clusters_per_iter;

    uint32_t granularity;

    int64_t sector;

    int64_t cluster;

    int64_t end;

    int64_t last_cluster = -1;

    int64_t sectors_per_cluster = cluster_size_sectors(job);

    BdrvDirtyBitmapIter *dbi;



    granularity = bdrv_dirty_bitmap_granularity(job->sync_bitmap);

    clusters_per_iter = MAX((granularity / job->cluster_size), 1);

    dbi = bdrv_dirty_iter_new(job->sync_bitmap, 0);



    /* Find the next dirty sector(s) */

    while ((sector = bdrv_dirty_iter_next(dbi)) != -1) {

        cluster = sector / sectors_per_cluster;



        /* Fake progress updates for any clusters we skipped */

        if (cluster != last_cluster + 1) {

            job->common.offset += ((cluster - last_cluster - 1) *

                                   job->cluster_size);

        }



        for (end = cluster + clusters_per_iter; cluster < end; cluster++) {

            do {

                if (yield_and_check(job)) {

                    goto out;

                }

                ret = backup_do_cow(job, cluster * job->cluster_size,

                                    job->cluster_size, &error_is_read,

                                    false);

                if ((ret < 0) &&

                    backup_error_action(job, error_is_read, -ret) ==

                    BLOCK_ERROR_ACTION_REPORT) {

                    goto out;

                }

            } while (ret < 0);

        }



        /* If the bitmap granularity is smaller than the backup granularity,

         * we need to advance the iterator pointer to the next cluster. */

        if (granularity < job->cluster_size) {

            bdrv_set_dirty_iter(dbi, cluster * sectors_per_cluster);

        }



        last_cluster = cluster - 1;

    }



    /* Play some final catchup with the progress meter */

    end = DIV_ROUND_UP(job->common.len, job->cluster_size);

    if (last_cluster + 1 < end) {

        job->common.offset += ((end - last_cluster - 1) * job->cluster_size);

    }



out:

    bdrv_dirty_iter_free(dbi);

    return ret;

}
