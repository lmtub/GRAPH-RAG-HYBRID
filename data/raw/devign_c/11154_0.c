static uint64_t macio_nvram_readb(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    MacIONVRAMState *s = opaque;

    uint32_t value;



    addr = (addr >> s->it_shift) & (s->size - 1);

    value = s->data[addr];

    NVR_DPRINTF("readb addr %04x val %x\n", (int)addr, value);



    return value;

}
