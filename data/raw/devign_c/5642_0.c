static uint32_t cuda_readb(void *opaque, target_phys_addr_t addr)

{

    CUDAState *s = opaque;

    uint32_t val;



    addr = (addr >> 9) & 0xf;

    switch(addr) {

    case 0:

        val = s->b;

        break;

    case 1:

        val = s->a;

        break;

    case 2:

        val = s->dirb;

        break;

    case 3:

        val = s->dira;

        break;

    case 4:

        val = get_counter(&s->timers[0]) & 0xff;

        s->ifr &= ~T1_INT;

        cuda_update_irq(s);

        break;

    case 5:

        val = get_counter(&s->timers[0]) >> 8;

        cuda_update_irq(s);

        break;

    case 6:

        val = s->timers[0].latch & 0xff;

        break;

    case 7:

        /* XXX: check this */

        val = (s->timers[0].latch >> 8) & 0xff;

        break;

    case 8:

        val = get_counter(&s->timers[1]) & 0xff;

        s->ifr &= ~T2_INT;

        break;

    case 9:

        val = get_counter(&s->timers[1]) >> 8;

        break;

    case 10:

        val = s->sr;

        s->ifr &= ~SR_INT;

        cuda_update_irq(s);

        break;

    case 11:

        val = s->acr;

        break;

    case 12:

        val = s->pcr;

        break;

    case 13:

        val = s->ifr;

        if (s->ifr & s->ier)

            val |= 0x80;

        break;

    case 14:

        val = s->ier | 0x80;

        break;

    default:

    case 15:

        val = s->anh;

        break;

    }

    if (addr != 13 || val != 0) {

        CUDA_DPRINTF("read: reg=0x%x val=%02x\n", (int)addr, val);

    }



    return val;

}
