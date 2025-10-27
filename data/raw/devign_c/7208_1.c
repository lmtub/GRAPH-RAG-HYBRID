static void lsi_reg_writeb(LSIState *s, int offset, uint8_t val)

{

#define CASE_SET_REG24(name, addr) \

    case addr    : s->name &= 0xffffff00; s->name |= val;       break; \

    case addr + 1: s->name &= 0xffff00ff; s->name |= val << 8;  break; \

    case addr + 2: s->name &= 0xff00ffff; s->name |= val << 16; break;



#define CASE_SET_REG32(name, addr) \

    case addr    : s->name &= 0xffffff00; s->name |= val;       break; \

    case addr + 1: s->name &= 0xffff00ff; s->name |= val << 8;  break; \

    case addr + 2: s->name &= 0xff00ffff; s->name |= val << 16; break; \

    case addr + 3: s->name &= 0x00ffffff; s->name |= val << 24; break;



#ifdef DEBUG_LSI_REG

    DPRINTF("Write reg %x = %02x\n", offset, val);

#endif

    switch (offset) {

    case 0x00: /* SCNTL0 */

        s->scntl0 = val;

        if (val & LSI_SCNTL0_START) {

            BADF("Start sequence not implemented\n");

        }

        break;

    case 0x01: /* SCNTL1 */

        s->scntl1 = val & ~LSI_SCNTL1_SST;

        if (val & LSI_SCNTL1_IARB) {

            BADF("Immediate Arbritration not implemented\n");

        }

        if (val & LSI_SCNTL1_RST) {

            if (!(s->sstat0 & LSI_SSTAT0_RST)) {

                BusChild *kid;



                QTAILQ_FOREACH(kid, &s->bus.qbus.children, sibling) {

                    DeviceState *dev = kid->child;

                    device_reset(dev);

                }

                s->sstat0 |= LSI_SSTAT0_RST;

                lsi_script_scsi_interrupt(s, LSI_SIST0_RST, 0);

            }

        } else {

            s->sstat0 &= ~LSI_SSTAT0_RST;

        }

        break;

    case 0x02: /* SCNTL2 */

        val &= ~(LSI_SCNTL2_WSR | LSI_SCNTL2_WSS);

        s->scntl2 = val;

        break;

    case 0x03: /* SCNTL3 */

        s->scntl3 = val;

        break;

    case 0x04: /* SCID */

        s->scid = val;

        break;

    case 0x05: /* SXFER */

        s->sxfer = val;

        break;

    case 0x06: /* SDID */

        if ((val & 0xf) != (s->ssid & 0xf))

            BADF("Destination ID does not match SSID\n");

        s->sdid = val & 0xf;

        break;

    case 0x07: /* GPREG0 */

        break;

    case 0x08: /* SFBR */

        /* The CPU is not allowed to write to this register.  However the

           SCRIPTS register move instructions are.  */

        s->sfbr = val;

        break;

    case 0x0a: case 0x0b:

        /* Openserver writes to these readonly registers on startup */

	return;

    case 0x0c: case 0x0d: case 0x0e: case 0x0f:

        /* Linux writes to these readonly registers on startup.  */

        return;

    CASE_SET_REG32(dsa, 0x10)

    case 0x14: /* ISTAT0 */

        s->istat0 = (s->istat0 & 0x0f) | (val & 0xf0);

        if (val & LSI_ISTAT0_ABRT) {

            lsi_script_dma_interrupt(s, LSI_DSTAT_ABRT);

        }

        if (val & LSI_ISTAT0_INTF) {

            s->istat0 &= ~LSI_ISTAT0_INTF;

            lsi_update_irq(s);

        }

        if (s->waiting == 1 && val & LSI_ISTAT0_SIGP) {

            DPRINTF("Woken by SIGP\n");

            s->waiting = 0;

            s->dsp = s->dnad;

            lsi_execute_script(s);

        }

        if (val & LSI_ISTAT0_SRST) {

            lsi_soft_reset(s);

        }

        break;

    case 0x16: /* MBOX0 */

        s->mbox0 = val;

        break;

    case 0x17: /* MBOX1 */

        s->mbox1 = val;

        break;

    case 0x1a: /* CTEST2 */

	s->ctest2 = val & LSI_CTEST2_PCICIE;

	break;

    case 0x1b: /* CTEST3 */

        s->ctest3 = val & 0x0f;

        break;

    CASE_SET_REG32(temp, 0x1c)

    case 0x21: /* CTEST4 */

        if (val & 7) {

           BADF("Unimplemented CTEST4-FBL 0x%x\n", val);

        }

        s->ctest4 = val;

        break;

    case 0x22: /* CTEST5 */

        if (val & (LSI_CTEST5_ADCK | LSI_CTEST5_BBCK)) {

            BADF("CTEST5 DMA increment not implemented\n");

        }

        s->ctest5 = val;

        break;

    CASE_SET_REG24(dbc, 0x24)

    CASE_SET_REG32(dnad, 0x28)

    case 0x2c: /* DSP[0:7] */

        s->dsp &= 0xffffff00;

        s->dsp |= val;

        break;

    case 0x2d: /* DSP[8:15] */

        s->dsp &= 0xffff00ff;

        s->dsp |= val << 8;

        break;

    case 0x2e: /* DSP[16:23] */

        s->dsp &= 0xff00ffff;

        s->dsp |= val << 16;

        break;

    case 0x2f: /* DSP[24:31] */

        s->dsp &= 0x00ffffff;

        s->dsp |= val << 24;

        if ((s->dmode & LSI_DMODE_MAN) == 0

            && (s->istat1 & LSI_ISTAT1_SRUN) == 0)

            lsi_execute_script(s);

        break;

    CASE_SET_REG32(dsps, 0x30)

    CASE_SET_REG32(scratch[0], 0x34)

    case 0x38: /* DMODE */

        if (val & (LSI_DMODE_SIOM | LSI_DMODE_DIOM)) {

            BADF("IO mappings not implemented\n");

        }

        s->dmode = val;

        break;

    case 0x39: /* DIEN */

        s->dien = val;

        lsi_update_irq(s);

        break;

    case 0x3a: /* SBR */

        s->sbr = val;

        break;

    case 0x3b: /* DCNTL */

        s->dcntl = val & ~(LSI_DCNTL_PFF | LSI_DCNTL_STD);

        if ((val & LSI_DCNTL_STD) && (s->istat1 & LSI_ISTAT1_SRUN) == 0)

            lsi_execute_script(s);

        break;

    case 0x40: /* SIEN0 */

        s->sien0 = val;

        lsi_update_irq(s);

        break;

    case 0x41: /* SIEN1 */

        s->sien1 = val;

        lsi_update_irq(s);

        break;

    case 0x47: /* GPCNTL0 */

        break;

    case 0x48: /* STIME0 */

        s->stime0 = val;

        break;

    case 0x49: /* STIME1 */

        if (val & 0xf) {

            DPRINTF("General purpose timer not implemented\n");

            /* ??? Raising the interrupt immediately seems to be sufficient

               to keep the FreeBSD driver happy.  */

            lsi_script_scsi_interrupt(s, 0, LSI_SIST1_GEN);

        }

        break;

    case 0x4a: /* RESPID0 */

        s->respid0 = val;

        break;

    case 0x4b: /* RESPID1 */

        s->respid1 = val;

        break;

    case 0x4d: /* STEST1 */

        s->stest1 = val;

        break;

    case 0x4e: /* STEST2 */

        if (val & 1) {

            BADF("Low level mode not implemented\n");

        }

        s->stest2 = val;

        break;

    case 0x4f: /* STEST3 */

        if (val & 0x41) {

            BADF("SCSI FIFO test mode not implemented\n");

        }

        s->stest3 = val;

        break;

    case 0x56: /* CCNTL0 */

        s->ccntl0 = val;

        break;

    case 0x57: /* CCNTL1 */

        s->ccntl1 = val;

        break;

    CASE_SET_REG32(mmrs, 0xa0)

    CASE_SET_REG32(mmws, 0xa4)

    CASE_SET_REG32(sfs, 0xa8)

    CASE_SET_REG32(drs, 0xac)

    CASE_SET_REG32(sbms, 0xb0)

    CASE_SET_REG32(dbms, 0xb4)

    CASE_SET_REG32(dnad64, 0xb8)

    CASE_SET_REG32(pmjad1, 0xc0)

    CASE_SET_REG32(pmjad2, 0xc4)

    CASE_SET_REG32(rbc, 0xc8)

    CASE_SET_REG32(ua, 0xcc)

    CASE_SET_REG32(ia, 0xd4)

    CASE_SET_REG32(sbc, 0xd8)

    CASE_SET_REG32(csbc, 0xdc)

    default:

        if (offset >= 0x5c && offset < 0xa0) {

            int n;

            int shift;

            n = (offset - 0x58) >> 2;

            shift = (offset & 3) * 8;

            s->scratch[n] &= ~(0xff << shift);

            s->scratch[n] |= (val & 0xff) << shift;

        } else {

            BADF("Unhandled writeb 0x%x = 0x%x\n", offset, val);

        }

    }

#undef CASE_SET_REG24

#undef CASE_SET_REG32

}
