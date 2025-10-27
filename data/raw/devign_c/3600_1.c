static uint32_t rtl8139_io_readb(void *opaque, uint8_t addr)

{

    RTL8139State *s = opaque;

    int ret;



    addr &= 0xff;



    switch (addr)

    {

        case MAC0 ... MAC0+5:

            ret = s->phys[addr - MAC0];

            break;

        case MAC0+6 ... MAC0+7:

            ret = 0;

            break;

        case MAR0 ... MAR0+7:

            ret = s->mult[addr - MAR0];

            break;

        case ChipCmd:

            ret = rtl8139_ChipCmd_read(s);

            break;

        case Cfg9346:

            ret = rtl8139_Cfg9346_read(s);

            break;

        case Config0:

            ret = rtl8139_Config0_read(s);

            break;

        case Config1:

            ret = rtl8139_Config1_read(s);

            break;

        case Config3:

            ret = rtl8139_Config3_read(s);

            break;

        case Config4:

            ret = rtl8139_Config4_read(s);

            break;

        case Config5:

            ret = rtl8139_Config5_read(s);

            break;



        case MediaStatus:

            ret = 0xd0;

            DPRINTF("MediaStatus read 0x%x\n", ret);

            break;



        case HltClk:

            ret = s->clock_enabled;

            DPRINTF("HltClk read 0x%x\n", ret);

            break;



        case PCIRevisionID:

            ret = RTL8139_PCI_REVID;

            DPRINTF("PCI Revision ID read 0x%x\n", ret);

            break;



        case TxThresh:

            ret = s->TxThresh;

            DPRINTF("C+ TxThresh read(b) val=0x%02x\n", ret);

            break;



        case 0x43: /* Part of TxConfig register. Windows driver tries to read it */

            ret = s->TxConfig >> 24;

            DPRINTF("RTL8139C TxConfig at 0x43 read(b) val=0x%02x\n", ret);

            break;



        default:

            DPRINTF("not implemented read(b) addr=0x%x\n", addr);

            ret = 0;

            break;

    }



    return ret;

}
