static uint32_t uhci_ioport_readw(void *opaque, uint32_t addr)

{

    UHCIState *s = opaque;

    uint32_t val;



    addr &= 0x1f;

    switch(addr) {

    case 0x00:

        val = s->cmd;

        break;

    case 0x02:

        val = s->status;

        break;

    case 0x04:

        val = s->intr;

        break;

    case 0x06:

        val = s->frnum;

        break;

    case 0x10 ... 0x1f:

        {

            UHCIPort *port;

            int n;

            n = (addr >> 1) & 7;

            if (n >= NB_PORTS)

                goto read_default;

            port = &s->ports[n];

            val = port->ctrl;

        }

        break;

    default:

    read_default:

        val = 0xff7f; /* disabled port */

        break;

    }

#ifdef DEBUG

    printf("uhci readw port=0x%04x val=0x%04x\n", addr, val);

#endif

    return val;

}
