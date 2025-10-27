static uint64_t empty_slot_read(void *opaque, target_phys_addr_t addr,

                                unsigned size)

{

    DPRINTF("read from " TARGET_FMT_plx "\n", addr);

    return 0;

}
