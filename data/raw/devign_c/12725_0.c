MigrationInfo *qmp_query_migrate(Error **errp)

{

    MigrationInfo *info = g_malloc0(sizeof(*info));

    MigrationState *s = migrate_get_current();



    switch (s->state) {

    case MIGRATION_STATUS_NONE:

        /* no migration has happened ever */

        break;

    case MIGRATION_STATUS_SETUP:

        info->has_status = true;

        info->status = MIGRATION_STATUS_SETUP;

        info->has_total_time = false;

        break;

    case MIGRATION_STATUS_ACTIVE:

    case MIGRATION_STATUS_CANCELLING:

        info->has_status = true;

        info->status = MIGRATION_STATUS_ACTIVE;

        info->has_total_time = true;

        info->total_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME)

            - s->total_time;

        info->has_expected_downtime = true;

        info->expected_downtime = s->expected_downtime;

        info->has_setup_time = true;

        info->setup_time = s->setup_time;



        info->has_ram = true;

        info->ram = g_malloc0(sizeof(*info->ram));

        info->ram->transferred = ram_bytes_transferred();

        info->ram->remaining = ram_bytes_remaining();

        info->ram->total = ram_bytes_total();

        info->ram->duplicate = dup_mig_pages_transferred();

        info->ram->skipped = skipped_mig_pages_transferred();

        info->ram->normal = norm_mig_pages_transferred();

        info->ram->normal_bytes = norm_mig_bytes_transferred();

        info->ram->dirty_pages_rate = s->dirty_pages_rate;

        info->ram->mbps = s->mbps;

        info->ram->dirty_sync_count = s->dirty_sync_count;



        if (blk_mig_active()) {

            info->has_disk = true;

            info->disk = g_malloc0(sizeof(*info->disk));

            info->disk->transferred = blk_mig_bytes_transferred();

            info->disk->remaining = blk_mig_bytes_remaining();

            info->disk->total = blk_mig_bytes_total();

        }



        get_xbzrle_cache_stats(info);

        break;

    case MIGRATION_STATUS_COMPLETED:

        get_xbzrle_cache_stats(info);



        info->has_status = true;

        info->status = MIGRATION_STATUS_COMPLETED;

        info->has_total_time = true;

        info->total_time = s->total_time;

        info->has_downtime = true;

        info->downtime = s->downtime;

        info->has_setup_time = true;

        info->setup_time = s->setup_time;



        info->has_ram = true;

        info->ram = g_malloc0(sizeof(*info->ram));

        info->ram->transferred = ram_bytes_transferred();

        info->ram->remaining = 0;

        info->ram->total = ram_bytes_total();

        info->ram->duplicate = dup_mig_pages_transferred();

        info->ram->skipped = skipped_mig_pages_transferred();

        info->ram->normal = norm_mig_pages_transferred();

        info->ram->normal_bytes = norm_mig_bytes_transferred();

        info->ram->mbps = s->mbps;

        info->ram->dirty_sync_count = s->dirty_sync_count;

        break;

    case MIGRATION_STATUS_FAILED:

        info->has_status = true;

        info->status = MIGRATION_STATUS_FAILED;

        break;

    case MIGRATION_STATUS_CANCELLED:

        info->has_status = true;

        info->status = MIGRATION_STATUS_CANCELLED;

        break;

    }



    return info;

}
