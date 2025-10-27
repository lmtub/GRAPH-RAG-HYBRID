static int blk_mig_save_bulked_block(Monitor *mon, QEMUFile *f)

{

    int64_t completed_sector_sum = 0;

    BlkMigDevState *bmds;

    int progress;

    int ret = 0;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        if (bmds->bulk_completed == 0) {

            if (mig_save_device_bulk(mon, f, bmds) == 1) {

                /* completed bulk section for this device */

                bmds->bulk_completed = 1;

            }

            completed_sector_sum += bmds->completed_sectors;

            ret = 1;

            break;

        } else {

            completed_sector_sum += bmds->completed_sectors;

        }

    }



    progress = completed_sector_sum * 100 / block_mig_state.total_sector_sum;

    if (progress != block_mig_state.prev_progress) {

        block_mig_state.prev_progress = progress;

        qemu_put_be64(f, (progress << BDRV_SECTOR_BITS)

                         | BLK_MIG_FLAG_PROGRESS);

        monitor_printf(mon, "Completed %d %%\r", progress);

        monitor_flush(mon);

    }



    return ret;

}
