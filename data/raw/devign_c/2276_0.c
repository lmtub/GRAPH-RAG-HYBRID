static void empty_slot_write(void *opaque, target_phys_addr_t addr,

                             uint64_t val, unsigned size)

{

    DPRINTF("write 0x%x to " TARGET_FMT_plx "\n", (unsigned)val, addr);

}
