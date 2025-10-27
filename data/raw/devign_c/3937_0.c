static void dp8393x_writew(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    dp8393xState *s = opaque;

    int reg;



    if ((addr & ((1 << s->it_shift) - 1)) != 0) {

        return;

    }



    reg = addr >> s->it_shift;



    write_register(s, reg, (uint16_t)val);

}
