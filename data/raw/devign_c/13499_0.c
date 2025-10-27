static void mcf_intc_write(void *opaque, target_phys_addr_t addr,

                           uint64_t val, unsigned size)

{

    int offset;

    mcf_intc_state *s = (mcf_intc_state *)opaque;

    offset = addr & 0xff;

    if (offset >= 0x40 && offset < 0x80) {

        int n = offset - 0x40;

        s->icr[n] = val;

        if (val == 0)

            s->enabled &= ~(1ull << n);

        else

            s->enabled |= (1ull << n);

        mcf_intc_update(s);

        return;

    }

    switch (offset) {

    case 0x00: case 0x04:

        /* Ignore IPR writes.  */

        return;

    case 0x08:

        s->imr = (s->imr & 0xffffffff) | ((uint64_t)val << 32);

        break;

    case 0x0c:

        s->imr = (s->imr & 0xffffffff00000000ull) | (uint32_t)val;

        break;

    default:

        hw_error("mcf_intc_write: Bad write offset %d\n", offset);

        break;

    }

    mcf_intc_update(s);

}
