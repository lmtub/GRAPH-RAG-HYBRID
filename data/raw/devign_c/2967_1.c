static void hmp_migrate_status_cb(void *opaque)

{

    MigrationStatus *status = opaque;

    MigrationInfo *info;



    info = qmp_query_migrate(NULL);

    if (!info->has_status || strcmp(info->status, "active") == 0) {

        if (info->has_disk) {

            int progress;



            if (info->disk->remaining) {

                progress = info->disk->transferred * 100 / info->disk->total;

            } else {

                progress = 100;

            }



            monitor_printf(status->mon, "Completed %d %%\r", progress);

            monitor_flush(status->mon);

        }



        timer_mod(status->timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1000);

    } else {

        if (status->is_block_migration) {

            monitor_printf(status->mon, "\n");

        }

        monitor_resume(status->mon);

        timer_del(status->timer);

        g_free(status);

    }



    qapi_free_MigrationInfo(info);

}
