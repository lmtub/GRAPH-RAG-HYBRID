ram_addr_t migration_bitmap_find_and_reset_dirty(MemoryRegion *mr,

                                                 ram_addr_t start)

{

    unsigned long base = mr->ram_addr >> TARGET_PAGE_BITS;

    unsigned long nr = base + (start >> TARGET_PAGE_BITS);

    uint64_t mr_size = TARGET_PAGE_ALIGN(memory_region_size(mr));

    unsigned long size = base + (mr_size >> TARGET_PAGE_BITS);



    unsigned long next;



    if (ram_bulk_stage && nr > base) {

        next = nr + 1;

    } else {

        next = find_next_bit(migration_bitmap, size, nr);

    }



    if (next < size) {

        clear_bit(next, migration_bitmap);

        migration_dirty_pages--;

    }

    return (next - base) << TARGET_PAGE_BITS;

}
