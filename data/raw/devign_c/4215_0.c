static void init_blk_migration_it(void *opaque, BlockDriverState *bs)

{

    Monitor *mon = opaque;

    BlkMigDevState *bmds;

    int64_t sectors;



    if (!bdrv_is_read_only(bs)) {

        sectors = bdrv_getlength(bs) >> BDRV_SECTOR_BITS;

        if (sectors <= 0) {

            return;

        }



        bmds = g_malloc0(sizeof(BlkMigDevState));

        bmds->bs = bs;

        bmds->bulk_completed = 0;

        bmds->total_sectors = sectors;

        bmds->completed_sectors = 0;

        bmds->shared_base = block_mig_state.shared_base;

        alloc_aio_bitmap(bmds);

        drive_get_ref(drive_get_by_blockdev(bs));

        bdrv_set_in_use(bs, 1);



        block_mig_state.total_sector_sum += sectors;



        if (bmds->shared_base) {

            monitor_printf(mon, "Start migration for %s with shared base "

                                "image\n",

                           bs->device_name);

        } else {

            monitor_printf(mon, "Start full migration for %s\n",

                           bs->device_name);

        }



        QSIMPLEQ_INSERT_TAIL(&block_mig_state.bmds_list, bmds, entry);

    }

}
