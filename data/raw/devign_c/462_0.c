static inline bool migration_bitmap_test_and_reset_dirty(MemoryRegion *mr,

                                                         ram_addr_t offset)

{

    bool ret;

    int nr = (mr->ram_addr + offset) >> TARGET_PAGE_BITS;



    ret = test_and_clear_bit(nr, migration_bitmap);



    if (ret) {

        migration_dirty_pages--;

    }

    return ret;

}
