static uint64_t pchip_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    TyphoonState *s = opaque;

    uint64_t ret = 0;



    if (addr & 4) {

        return s->latch_tmp;

    }



    switch (addr) {

    case 0x0000:

        /* WSBA0: Window Space Base Address Register.  */

        ret = s->pchip.win[0].base_addr;

        break;

    case 0x0040:

        /* WSBA1 */

        ret = s->pchip.win[1].base_addr;

        break;

    case 0x0080:

        /* WSBA2 */

        ret = s->pchip.win[2].base_addr;

        break;

    case 0x00c0:

        /* WSBA3 */

        ret = s->pchip.win[3].base_addr;

        break;



    case 0x0100:

        /* WSM0: Window Space Mask Register.  */

        ret = s->pchip.win[0].mask;

        break;

    case 0x0140:

        /* WSM1 */

        ret = s->pchip.win[1].mask;

        break;

    case 0x0180:

        /* WSM2 */

        ret = s->pchip.win[2].mask;

        break;

    case 0x01c0:

        /* WSM3 */

        ret = s->pchip.win[3].mask;

        break;



    case 0x0200:

        /* TBA0: Translated Base Address Register.  */

        ret = (uint64_t)s->pchip.win[0].translated_base_pfn << 10;

        break;

    case 0x0240:

        /* TBA1 */

        ret = (uint64_t)s->pchip.win[1].translated_base_pfn << 10;

        break;

    case 0x0280:

        /* TBA2 */

        ret = (uint64_t)s->pchip.win[2].translated_base_pfn << 10;

        break;

    case 0x02c0:

        /* TBA3 */

        ret = (uint64_t)s->pchip.win[3].translated_base_pfn << 10;

        break;



    case 0x0300:

        /* PCTL: Pchip Control Register.  */

        ret = s->pchip.ctl;

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

        /* TLBIV: Translation Buffer Invalidate Virtual Register (WO).  */

        break;

    case 0x04c0:

        /* TLBIA: Translation Buffer Invalidate All Register (WO).  */

        break;

    case 0x0500: /* PMONCTL */

    case 0x0540: /* PMONCNT */

    case 0x0800: /* SPRST */

        break;



    default:

        cpu_unassigned_access(cpu_single_env, addr, 0, 0, 0, size);

        return -1;

    }



    s->latch_tmp = ret >> 32;

    return ret;

}
