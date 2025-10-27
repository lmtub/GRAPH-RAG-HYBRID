static void lsi_ram_write(void *opaque, target_phys_addr_t addr,

                          uint64_t val, unsigned size)

{

    LSIState *s = opaque;

    uint32_t newval;

    uint32_t mask;

    int shift;



    newval = s->script_ram[addr >> 2];

    shift = (addr & 3) * 8;

    mask = ((uint64_t)1 << (size * 8)) - 1;

    newval &= ~(mask << shift);

    newval |= val << shift;

    s->script_ram[addr >> 2] = newval;

}
