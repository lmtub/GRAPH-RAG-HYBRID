static void cchip_write(void *opaque, target_phys_addr_t addr,

                        uint64_t v32, unsigned size)

{

    TyphoonState *s = opaque;

    uint64_t val, oldval, newval;



    if (addr & 4) {

        val = v32 << 32 | s->latch_tmp;

        addr ^= 4;

    } else {

        s->latch_tmp = v32;

        return;

    }



    switch (addr) {

    case 0x0000:

        /* CSC: Cchip System Configuration Register.  */

        /* All sorts of data here; nothing relevant RW.  */

        break;



    case 0x0040:

        /* MTR: Memory Timing Register.  */

        /* All sorts of stuff related to real DRAM.  */

        break;



    case 0x0080:

        /* MISC: Miscellaneous Register.  */

        newval = oldval = s->cchip.misc;

        newval &= ~(val & 0x10000ff0);     /* W1C fields */

        if (val & 0x100000) {

            newval &= ~0xff0000ull;        /* ACL clears ABT and ABW */

        } else {

            newval |= val & 0x00f00000;    /* ABT field is W1S */

            if ((newval & 0xf0000) == 0) {

                newval |= val & 0xf0000;   /* ABW field is W1S iff zero */

            }

        }

        newval |= (val & 0xf000) >> 4;     /* IPREQ field sets IPINTR.  */



        newval &= ~0xf0000000000ull;       /* WO and RW fields */

        newval |= val & 0xf0000000000ull;

        s->cchip.misc = newval;



        /* Pass on changes to IPI and ITI state.  */

        if ((newval ^ oldval) & 0xff0) {

            int i;

            for (i = 0; i < 4; ++i) {

                CPUAlphaState *env = s->cchip.cpu[i];

                if (env) {

                    /* IPI can be either cleared or set by the write.  */

                    if (newval & (1 << (i + 8))) {

                        cpu_interrupt(env, CPU_INTERRUPT_SMP);

                    } else {

                        cpu_reset_interrupt(env, CPU_INTERRUPT_SMP);

                    }



                    /* ITI can only be cleared by the write.  */

                    if ((newval & (1 << (i + 4))) == 0) {

                        cpu_reset_interrupt(env, CPU_INTERRUPT_TIMER);

                    }

                }

            }

        }

        break;



    case 0x00c0:

        /* MPD: Memory Presence Detect Register.  */

        break;



    case 0x0100: /* AAR0 */

    case 0x0140: /* AAR1 */

    case 0x0180: /* AAR2 */

    case 0x01c0: /* AAR3 */

        /* AAR: Array Address Register.  */

        /* All sorts of information about DRAM.  */

        break;



    case 0x0200: /* DIM0 */

        /* DIM: Device Interrupt Mask Register, CPU0.  */

        s->cchip.dim[0] = val;

        cpu_irq_change(s->cchip.cpu[0], val & s->cchip.drir);

        break;

    case 0x0240: /* DIM1 */

        /* DIM: Device Interrupt Mask Register, CPU1.  */

        s->cchip.dim[0] = val;

        cpu_irq_change(s->cchip.cpu[1], val & s->cchip.drir);

        break;



    case 0x0280: /* DIR0 (RO) */

    case 0x02c0: /* DIR1 (RO) */

    case 0x0300: /* DRIR (RO) */

        break;



    case 0x0340:

        /* PRBEN: Probe Enable Register.  */

        break;



    case 0x0380: /* IIC0 */

        s->cchip.iic[0] = val & 0xffffff;

        break;

    case 0x03c0: /* IIC1 */

        s->cchip.iic[1] = val & 0xffffff;

        break;



    case 0x0400: /* MPR0 */

    case 0x0440: /* MPR1 */

    case 0x0480: /* MPR2 */

    case 0x04c0: /* MPR3 */

        /* MPR: Memory Programming Register.  */

        break;



    case 0x0580:

        /* TTR: TIGbus Timing Register.  */

        /* All sorts of stuff related to interrupt delivery timings.  */

        break;

    case 0x05c0:

        /* TDR: TIGbug Device Timing Register.  */

        break;



    case 0x0600:

        /* DIM2: Device Interrupt Mask Register, CPU2.  */

        s->cchip.dim[2] = val;

        cpu_irq_change(s->cchip.cpu[2], val & s->cchip.drir);

        break;

    case 0x0640:

        /* DIM3: Device Interrupt Mask Register, CPU3.  */

        s->cchip.dim[3] = val;

        cpu_irq_change(s->cchip.cpu[3], val & s->cchip.drir);

        break;



    case 0x0680: /* DIR2 (RO) */

    case 0x06c0: /* DIR3 (RO) */

        break;



    case 0x0700: /* IIC2 */

        s->cchip.iic[2] = val & 0xffffff;

        break;

    case 0x0740: /* IIC3 */

        s->cchip.iic[3] = val & 0xffffff;

        break;



    case 0x0780:

        /* PWR: Power Management Control.   */

        break;

    

    case 0x0c00: /* CMONCTLA */

    case 0x0c40: /* CMONCTLB */

    case 0x0c80: /* CMONCNT01 */

    case 0x0cc0: /* CMONCNT23 */

        break;



    default:

        cpu_unassigned_access(cpu_single_env, addr, 1, 0, 0, size);

        return;

    }

}
