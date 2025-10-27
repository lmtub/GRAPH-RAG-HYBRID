static void pic_write(void *opaque, target_phys_addr_t addr,

                      uint64_t value, unsigned size)

{

    HeathrowPICS *s = opaque;

    HeathrowPIC *pic;

    unsigned int n;



    n = ((addr & 0xfff) - 0x10) >> 4;

    PIC_DPRINTF("writel: " TARGET_FMT_plx " %u: %08x\n", addr, n, value);

    if (n >= 2)

        return;

    pic = &s->pics[n];

    switch(addr & 0xf) {

    case 0x04:

        pic->mask = value;

        heathrow_pic_update(s);

        break;

    case 0x08:

        /* do not reset level triggered IRQs */

        value &= ~pic->level_triggered;

        pic->events &= ~value;

        heathrow_pic_update(s);

        break;

    default:

        break;

    }

}
