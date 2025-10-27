static int ram_save_target_page(MigrationState *ms, QEMUFile *f,

                                RAMBlock *block, ram_addr_t offset,

                                bool last_stage,

                                uint64_t *bytes_transferred,

                                ram_addr_t dirty_ram_abs)

{

    int res = 0;



    /* Check the pages is dirty and if it is send it */

    if (migration_bitmap_clear_dirty(dirty_ram_abs)) {

        unsigned long *unsentmap;

        if (compression_switch && migrate_use_compression()) {

            res = ram_save_compressed_page(f, block, offset,

                                           last_stage,

                                           bytes_transferred);

        } else {

            res = ram_save_page(f, block, offset, last_stage,

                                bytes_transferred);

        }



        if (res < 0) {

            return res;

        }

        unsentmap = atomic_rcu_read(&migration_bitmap_rcu)->unsentmap;

        if (unsentmap) {

            clear_bit(dirty_ram_abs >> TARGET_PAGE_BITS, unsentmap);

        }

        last_sent_block = block;

    }



    return res;

}
