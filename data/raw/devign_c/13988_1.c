static void pm_ioport_read(IORange *ioport, uint64_t addr, unsigned width,

                            uint64_t *data)

{

    PIIX4PMState *s = container_of(ioport, PIIX4PMState, ioport);

    uint32_t val;



    switch(addr) {

    case 0x00:

        val = acpi_pm1_evt_get_sts(&s->ar, s->ar.tmr.overflow_time);

        break;

    case 0x02:

        val = s->ar.pm1.evt.en;

        break;

    case 0x04:

        val = s->ar.pm1.cnt.cnt;

        break;

    case 0x08:

        val = acpi_pm_tmr_get(&s->ar);

        break;

    default:

        val = 0;

        break;

    }

    PIIX4_DPRINTF("PM readw port=0x%04x val=0x%04x\n", (unsigned int)addr, val);

    *data = val;

}
