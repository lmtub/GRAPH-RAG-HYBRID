static int get_phys_addr_v6(CPUARMState *env, uint32_t address, int access_type,

                            ARMMMUIdx mmu_idx, hwaddr *phys_ptr,

                            int *prot, target_ulong *page_size)

{

    CPUState *cs = CPU(arm_env_get_cpu(env));

    int code;

    uint32_t table;

    uint32_t desc;

    uint32_t xn;

    uint32_t pxn = 0;

    int type;

    int ap;

    int domain = 0;

    int domain_prot;

    hwaddr phys_addr;

    uint32_t dacr;



    /* Pagetable walk.  */

    /* Lookup l1 descriptor.  */

    if (!get_level1_table_address(env, mmu_idx, &table, address)) {

        /* Section translation fault if page walk is disabled by PD0 or PD1 */

        code = 5;

        goto do_fault;

    }

    desc = ldl_phys(cs->as, table);

    type = (desc & 3);

    if (type == 0 || (type == 3 && !arm_feature(env, ARM_FEATURE_PXN))) {

        /* Section translation fault, or attempt to use the encoding

         * which is Reserved on implementations without PXN.

         */

        code = 5;

        goto do_fault;

    }

    if ((type == 1) || !(desc & (1 << 18))) {

        /* Page or Section.  */

        domain = (desc >> 5) & 0x0f;

    }

    if (regime_el(env, mmu_idx) == 1) {

        dacr = env->cp15.dacr_ns;

    } else {

        dacr = env->cp15.dacr_s;

    }

    domain_prot = (dacr >> (domain * 2)) & 3;

    if (domain_prot == 0 || domain_prot == 2) {

        if (type != 1) {

            code = 9; /* Section domain fault.  */

        } else {

            code = 11; /* Page domain fault.  */

        }

        goto do_fault;

    }

    if (type != 1) {

        if (desc & (1 << 18)) {

            /* Supersection.  */

            phys_addr = (desc & 0xff000000) | (address & 0x00ffffff);

            *page_size = 0x1000000;

        } else {

            /* Section.  */

            phys_addr = (desc & 0xfff00000) | (address & 0x000fffff);

            *page_size = 0x100000;

        }

        ap = ((desc >> 10) & 3) | ((desc >> 13) & 4);

        xn = desc & (1 << 4);

        pxn = desc & 1;

        code = 13;

    } else {

        if (arm_feature(env, ARM_FEATURE_PXN)) {

            pxn = (desc >> 2) & 1;

        }

        /* Lookup l2 entry.  */

        table = (desc & 0xfffffc00) | ((address >> 10) & 0x3fc);

        desc = ldl_phys(cs->as, table);

        ap = ((desc >> 4) & 3) | ((desc >> 7) & 4);

        switch (desc & 3) {

        case 0: /* Page translation fault.  */

            code = 7;

            goto do_fault;

        case 1: /* 64k page.  */

            phys_addr = (desc & 0xffff0000) | (address & 0xffff);

            xn = desc & (1 << 15);

            *page_size = 0x10000;

            break;

        case 2: case 3: /* 4k page.  */

            phys_addr = (desc & 0xfffff000) | (address & 0xfff);

            xn = desc & 1;

            *page_size = 0x1000;

            break;

        default:

            /* Never happens, but compiler isn't smart enough to tell.  */

            abort();

        }

        code = 15;

    }

    if (domain_prot == 3) {

        *prot = PAGE_READ | PAGE_WRITE | PAGE_EXEC;

    } else {

        if (pxn && !regime_is_user(env, mmu_idx)) {

            xn = 1;

        }

        if (xn && access_type == 2)

            goto do_fault;



        /* The simplified model uses AP[0] as an access control bit.  */

        if ((regime_sctlr(env, mmu_idx) & SCTLR_AFE)

                && (ap & 1) == 0) {

            /* Access flag fault.  */

            code = (code == 15) ? 6 : 3;

            goto do_fault;

        }

        *prot = check_ap(env, mmu_idx, ap, domain_prot, access_type);

        if (!*prot) {

            /* Access permission fault.  */

            goto do_fault;

        }

        if (!xn) {

            *prot |= PAGE_EXEC;

        }

    }

    *phys_ptr = phys_addr;

    return 0;

do_fault:

    return code | (domain << 4);

}
