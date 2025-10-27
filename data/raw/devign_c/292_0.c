static uint64_t sp804_read(void *opaque, target_phys_addr_t offset,

                           unsigned size)

{

    sp804_state *s = (sp804_state *)opaque;



    if (offset < 0x20) {

        return arm_timer_read(s->timer[0], offset);

    }

    if (offset < 0x40) {

        return arm_timer_read(s->timer[1], offset - 0x20);

    }



    /* TimerPeriphID */

    if (offset >= 0xfe0 && offset <= 0xffc) {

        return sp804_ids[(offset - 0xfe0) >> 2];

    }



    switch (offset) {

    /* Integration Test control registers, which we won't support */

    case 0xf00: /* TimerITCR */

    case 0xf04: /* TimerITOP (strictly write only but..) */

        return 0;

    }



    hw_error("%s: Bad offset %x\n", __func__, (int)offset);

    return 0;

}
