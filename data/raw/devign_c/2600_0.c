static uint32_t dp8393x_readw(void *opaque, target_phys_addr_t addr)

{

    dp8393xState *s = opaque;

    int reg;



    if ((addr & ((1 << s->it_shift) - 1)) != 0) {

        return 0;

    }



    reg = addr >> s->it_shift;

    return read_register(s, reg);

}
