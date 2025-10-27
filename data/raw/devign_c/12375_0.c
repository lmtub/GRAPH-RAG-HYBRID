static void gpio_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                       unsigned size)

{

    struct gpio_state_t *s = opaque;

    D(printf("%s %x=%x\n", __func__, addr, (unsigned)value));



    addr >>= 2;

    switch (addr)

    {

        case RW_PA_DOUT:

            /* Decode nand pins.  */

            s->nand->ale = !!(value & (1 << 6));

            s->nand->cle = !!(value & (1 << 5));

            s->nand->ce  = !!(value & (1 << 4));



            s->regs[addr] = value;

            break;



        case RW_PD_DOUT:

            /* Temp sensor clk.  */

            if ((s->regs[addr] ^ value) & 2)

                tempsensor_clkedge(&s->tempsensor, !!(value & 2),

                                   !!(value & 16));

            s->regs[addr] = value;

            break;



        default:

            s->regs[addr] = value;

            break;

    }

}
