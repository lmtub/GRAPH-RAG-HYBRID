static void rtl8139_io_writew(void *opaque, uint8_t addr, uint32_t val)

{

    RTL8139State *s = opaque;



    addr &= 0xfe;



    switch (addr)

    {

        case IntrMask:

            rtl8139_IntrMask_write(s, val);

            break;



        case IntrStatus:

            rtl8139_IntrStatus_write(s, val);

            break;



        case MultiIntr:

            rtl8139_MultiIntr_write(s, val);

            break;



        case RxBufPtr:

            rtl8139_RxBufPtr_write(s, val);

            break;



        case BasicModeCtrl:

            rtl8139_BasicModeCtrl_write(s, val);

            break;

        case BasicModeStatus:

            rtl8139_BasicModeStatus_write(s, val);

            break;

        case NWayAdvert:

            DPRINTF("NWayAdvert write(w) val=0x%04x\n", val);

            s->NWayAdvert = val;

            break;

        case NWayLPAR:

            DPRINTF("forbidden NWayLPAR write(w) val=0x%04x\n", val);

            break;

        case NWayExpansion:

            DPRINTF("NWayExpansion write(w) val=0x%04x\n", val);

            s->NWayExpansion = val;

            break;



        case CpCmd:

            rtl8139_CpCmd_write(s, val);

            break;



        case IntrMitigate:

            rtl8139_IntrMitigate_write(s, val);

            break;



        default:

            DPRINTF("ioport write(w) addr=0x%x val=0x%04x via write(b)\n",

                addr, val);



            rtl8139_io_writeb(opaque, addr, val & 0xff);

            rtl8139_io_writeb(opaque, addr + 1, (val >> 8) & 0xff);

            break;

    }

}
