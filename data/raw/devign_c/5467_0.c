static uint32_t hpet_ram_readb(void *opaque, target_phys_addr_t addr)

{

    printf("qemu: hpet_read b at %" PRIx64 "\n", addr);

    return 0;

}
