static inline bool migration_bitmap_clear_dirty(RAMState *rs, ram_addr_t addr)

{

    bool ret;

    int nr = addr >> TARGET_PAGE_BITS;

    unsigned long *bitmap = atomic_rcu_read(&rs->ram_bitmap)->bmap;



    ret = test_and_clear_bit(nr, bitmap);



    if (ret) {

        rs->migration_dirty_pages--;

    }

    return ret;

}
