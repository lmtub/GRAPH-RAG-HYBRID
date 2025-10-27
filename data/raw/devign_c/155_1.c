static void ehci_mem_writew(void *ptr, target_phys_addr_t addr, uint32_t val)

{

    fprintf(stderr, "EHCI doesn't handle 16-bit writes to MMIO\n");

    exit(1);

}
