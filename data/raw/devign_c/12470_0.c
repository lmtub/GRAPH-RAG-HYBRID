static bool migrate_caps_check(bool *cap_list,

                               MigrationCapabilityStatusList *params,

                               Error **errp)

{

    MigrationCapabilityStatusList *cap;

    bool old_postcopy_cap;



    old_postcopy_cap = cap_list[MIGRATION_CAPABILITY_POSTCOPY_RAM];



    for (cap = params; cap; cap = cap->next) {

        cap_list[cap->value->capability] = cap->value->state;

    }



#ifndef CONFIG_LIVE_BLOCK_MIGRATION

    if (cap_list[MIGRATION_CAPABILITY_BLOCK]) {

        error_setg(errp, "QEMU compiled without old-style (blk/-b, inc/-i) "

                   "block migration");

        error_append_hint(errp, "Use drive_mirror+NBD instead.\n");

        return false;

    }

#endif



    if (cap_list[MIGRATION_CAPABILITY_POSTCOPY_RAM]) {

        if (cap_list[MIGRATION_CAPABILITY_COMPRESS]) {

            /* The decompression threads asynchronously write into RAM

             * rather than use the atomic copies needed to avoid

             * userfaulting.  It should be possible to fix the decompression

             * threads for compatibility in future.

             */

            error_setg(errp, "Postcopy is not currently compatible "

                       "with compression");

            return false;

        }



        /* This check is reasonably expensive, so only when it's being

         * set the first time, also it's only the destination that needs

         * special support.

         */

        if (!old_postcopy_cap && runstate_check(RUN_STATE_INMIGRATE) &&

            !postcopy_ram_supported_by_host()) {

            /* postcopy_ram_supported_by_host will have emitted a more

             * detailed message

             */

            error_setg(errp, "Postcopy is not supported");

            return false;

        }

    }



    return true;

}
