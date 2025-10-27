static uint64_t cchip_read(void *opaque, hwaddr addr, unsigned size)

{

    CPUState *cpu = current_cpu;

    TyphoonState *s = opaque;

    uint64_t ret = 0;



    if (addr & 4) {

        return s->latch_tmp;

    }



    switch (addr) {

    case 0x0000:

        /* CSC: Cchip System Configuration Register.  */

        /* All sorts of data here; probably the only thing relevant is

           PIP<14> Pchip 1 Present = 0.  */

        break;



    case 0x0040:

        /* MTR: Memory Timing Register.  */

        /* All sorts of stuff related to real DRAM.  */

        break;



    case 0x0080:

        /* MISC: Miscellaneous Register.  */

        ret = s->cchip.misc | (cpu->cpu_index & 3);

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



    case 0x0200:

        /* DIM0: Device Interrupt Mask Register, CPU0.  */

        ret = s->cchip.dim[0];

        break;

    case 0x0240:

        /* DIM1: Device Interrupt Mask Register, CPU1.  */

        ret = s->cchip.dim[1];

        break;

    case 0x0280:

        /* DIR0: Device Interrupt Request Register, CPU0.  */

        ret = s->cchip.dim[0] & s->cchip.drir;

        break;

    case 0x02c0:

        /* DIR1: Device Interrupt Request Register, CPU1.  */

        ret = s->cchip.dim[1] & s->cchip.drir;

        break;

    case 0x0300:

        /* DRIR: Device Raw Interrupt Request Register.  */

        ret = s->cchip.drir;

        break;



    case 0x0340:

        /* PRBEN: Probe Enable Register.  */

        break;



    case 0x0380:

        /* IIC0: Interval Ignore Count Register, CPU0.  */

        ret = s->cchip.iic[0];

        break;

    case 0x03c0:

        /* IIC1: Interval Ignore Count Register, CPU1.  */

        ret = s->cchip.iic[1];

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

        ret = s->cchip.dim[2];

        break;

    case 0x0640:

        /* DIM3: Device Interrupt Mask Register, CPU3.  */

        ret = s->cchip.dim[3];

        break;

    case 0x0680:

        /* DIR2: Device Interrupt Request Register, CPU2.  */

        ret = s->cchip.dim[2] & s->cchip.drir;

        break;

    case 0x06c0:

        /* DIR3: Device Interrupt Request Register, CPU3.  */

        ret = s->cchip.dim[3] & s->cchip.drir;

        break;



    case 0x0700:

        /* IIC2: Interval Ignore Count Register, CPU2.  */

        ret = s->cchip.iic[2];

        break;

    case 0x0740:

        /* IIC3: Interval Ignore Count Register, CPU3.  */

        ret = s->cchip.iic[3];

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

        cpu_unassigned_access(cpu, addr, false, false, 0, size);

        return -1;

    }



    s->latch_tmp = ret >> 32;

    return ret;

}
