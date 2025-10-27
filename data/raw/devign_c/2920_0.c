static void pchip_write(void *opaque, target_phys_addr_t addr,

                        uint64_t v32, unsigned size)

{

    TyphoonState *s = opaque;

    uint64_t val, oldval;



    if (addr & 4) {

        val = v32 << 32 | s->latch_tmp;

        addr ^= 4;

    } else {

        s->latch_tmp = v32;

        return;

    }



    switch (addr) {

    case 0x0000:

        /* WSBA0: Window Space Base Address Register.  */

        s->pchip.win[0].base_addr = val;

        break;

    case 0x0040:

        /* WSBA1 */

        s->pchip.win[1].base_addr = val;

        break;

    case 0x0080:

        /* WSBA2 */

        s->pchip.win[2].base_addr = val;

        break;

    case 0x00c0:

        /* WSBA3 */

        s->pchip.win[3].base_addr = val;

        break;



    case 0x0100:

        /* WSM0: Window Space Mask Register.  */

        s->pchip.win[0].mask = val;

        break;

    case 0x0140:

        /* WSM1 */

        s->pchip.win[1].mask = val;

        break;

    case 0x0180:

        /* WSM2 */

        s->pchip.win[2].mask = val;

        break;

    case 0x01c0:

        /* WSM3 */

        s->pchip.win[3].mask = val;

        break;



    case 0x0200:

        /* TBA0: Translated Base Address Register.  */

        s->pchip.win[0].translated_base_pfn = val >> 10;

        break;

    case 0x0240:

        /* TBA1 */

        s->pchip.win[1].translated_base_pfn = val >> 10;

        break;

    case 0x0280:

        /* TBA2 */

        s->pchip.win[2].translated_base_pfn = val >> 10;

        break;

    case 0x02c0:

        /* TBA3 */

        s->pchip.win[3].translated_base_pfn = val >> 10;

        break;



    case 0x0300:

        /* PCTL: Pchip Control Register.  */

        oldval = s->pchip.ctl;

        oldval &= ~0x00001cff0fc7ffull;       /* RW fields */

        oldval |= val & 0x00001cff0fc7ffull;



        s->pchip.ctl = oldval;

        break;



    case 0x0340:

        /* PLAT: Pchip Master Latency Register.  */

        break;

    case 0x03c0:

        /* PERROR: Pchip Error Register.  */

        break;

    case 0x0400:

        /* PERRMASK: Pchip Error Mask Register.  */

        break;

    case 0x0440:

        /* PERRSET: Pchip Error Set Register.  */

        break;



    case 0x0480:

        /* TLBIV: Translation Buffer Invalidate Virtual Register.  */

        break;



    case 0x04c0:

        /* TLBIA: Translation Buffer Invalidate All Register (WO).  */

        break;



    case 0x0500:

        /* PMONCTL */

    case 0x0540:

        /* PMONCNT */

    case 0x0800:

        /* SPRST */

        break;



    default:

        cpu_unassigned_access(cpu_single_env, addr, 1, 0, 0, size);

        return;

    }

}
