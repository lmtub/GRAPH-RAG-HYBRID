static void migration_bitmap_sync(void)

{

    uint64_t num_dirty_pages_init = ram_list.dirty_pages;



    trace_migration_bitmap_sync_start();

    memory_global_sync_dirty_bitmap(get_system_memory());

    trace_migration_bitmap_sync_end(ram_list.dirty_pages

                                    - num_dirty_pages_init);

}
