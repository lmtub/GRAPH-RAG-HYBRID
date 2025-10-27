static uint64_t mpcore_scu_read(void *opaque, target_phys_addr_t offset,

                                unsigned size)

{

    mpcore_priv_state *s = (mpcore_priv_state *)opaque;

    int id;

    /* SCU */

    switch (offset) {

    case 0x00: /* Control.  */

        return s->scu_control;

    case 0x04: /* Configuration.  */

        id = ((1 << s->num_cpu) - 1) << 4;

        return id | (s->num_cpu - 1);

    case 0x08: /* CPU status.  */

        return 0;

    case 0x0c: /* Invalidate all.  */

        return 0;

    default:

        hw_error("mpcore_priv_read: Bad offset %x\n", (int)offset);

    }

}
