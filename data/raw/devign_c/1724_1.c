static int ehci_register_companion(USBBus *bus, USBPort *ports[],

                                   uint32_t portcount, uint32_t firstport)

{

    EHCIState *s = container_of(bus, EHCIState, bus);

    uint32_t i;



    if (firstport + portcount > NB_PORTS) {

        qerror_report(QERR_INVALID_PARAMETER_VALUE, "firstport",

                      "firstport on masterbus");

        error_printf_unless_qmp(

            "firstport value of %u makes companion take ports %u - %u, which "

            "is outside of the valid range of 0 - %u\n", firstport, firstport,

            firstport + portcount - 1, NB_PORTS - 1);

        return -1;

    }



    for (i = 0; i < portcount; i++) {

        if (s->companion_ports[firstport + i]) {

            qerror_report(QERR_INVALID_PARAMETER_VALUE, "masterbus",

                          "an USB masterbus");

            error_printf_unless_qmp(

                "port %u on masterbus %s already has a companion assigned\n",

                firstport + i, bus->qbus.name);

            return -1;

        }

    }



    for (i = 0; i < portcount; i++) {

        s->companion_ports[firstport + i] = ports[i];

        s->ports[firstport + i].speedmask |=

            USB_SPEED_MASK_LOW | USB_SPEED_MASK_FULL;

        /* Ensure devs attached before the initial reset go to the companion */

        s->portsc[firstport + i] = PORTSC_POWNER;

    }



    s->companion_count++;

    s->mmio[0x05] = (s->companion_count << 4) | portcount;



    return 0;

}
