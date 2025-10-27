static void uhci_ioport_writel(void *opaque, uint32_t addr, uint32_t val)

{

    UHCIState *s = opaque;



    addr &= 0x1f;

#ifdef DEBUG

    printf("uhci writel port=0x%04x val=0x%08x\n", addr, val);

#endif

    switch(addr) {

    case 0x08:

        s->fl_base_addr = val & ~0xfff;

        break;

    }

}
