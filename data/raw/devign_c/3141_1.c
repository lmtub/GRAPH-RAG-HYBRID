void apic_deliver_pic_intr(DeviceState *dev, int level)

{

    APICCommonState *s = APIC_COMMON(dev);



    if (level) {

        apic_local_deliver(s, APIC_LVT_LINT0);

    } else {

        uint32_t lvt = s->lvt[APIC_LVT_LINT0];



        switch ((lvt >> 8) & 7) {

        case APIC_DM_FIXED:

            if (!(lvt & APIC_LVT_LEVEL_TRIGGER))

                break;

            apic_reset_bit(s->irr, lvt & 0xff);

            /* fall through */

        case APIC_DM_EXTINT:

            cpu_reset_interrupt(CPU(s->cpu), CPU_INTERRUPT_HARD);

            break;

        }

    }

}
