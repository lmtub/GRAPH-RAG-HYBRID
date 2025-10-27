static void mpcore_scu_write(void *opaque, target_phys_addr_t offset,

                             uint64_t value, unsigned size)

{

    mpcore_priv_state *s = (mpcore_priv_state *)opaque;

    /* SCU */

    switch (offset) {

    case 0: /* Control register.  */

        s->scu_control = value & 1;

        break;

    case 0x0c: /* Invalidate all.  */

        /* This is a no-op as cache is not emulated.  */

        break;

    default:

        hw_error("mpcore_priv_read: Bad offset %x\n", (int)offset);

    }

}
