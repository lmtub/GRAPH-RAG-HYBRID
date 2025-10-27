static uint32_t ehci_mem_readw(void *ptr, target_phys_addr_t addr)

{

    EHCIState *s = ptr;

    uint32_t val;



    val = s->mmio[addr] | (s->mmio[addr+1] << 8);



    return val;

}
