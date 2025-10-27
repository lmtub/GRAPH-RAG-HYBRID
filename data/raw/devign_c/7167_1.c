static uint32_t ehci_mem_readb(void *ptr, target_phys_addr_t addr)

{

    EHCIState *s = ptr;

    uint32_t val;



    val = s->mmio[addr];



    return val;

}
