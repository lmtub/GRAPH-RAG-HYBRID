static int get_phys_addr_v6(CPUState *env, uint32_t address, int access_type,

			    int is_user, uint32_t *phys_ptr, int *prot)

{

    int code;

    uint32_t table;

    uint32_t desc;

    uint32_t xn;

    int type;

    int ap;

    int domain;

    uint32_t phys_addr;



    /* Pagetable walk.  */

    /* Lookup l1 descriptor.  */

    table = get_level1_table_address(env, address);

    desc = ldl_phys(table);

    type = (desc & 3);

    if (type == 0) {

        /* Section translation fault.  */

        code = 5;

        domain = 0;

        goto do_fault;

    } else if (type == 2 && (desc & (1 << 18))) {

        /* Supersection.  */

        domain = 0;

    } else {

        /* Section or page.  */

        domain = (desc >> 4) & 0x1e;

    }

    domain = (env->cp15.c3 >> domain) & 3;

    if (domain == 0 || domain == 2) {

        if (type == 2)

            code = 9; /* Section domain fault.  */

        else

            code = 11; /* Page domain fault.  */

        goto do_fault;

    }

    if (type == 2) {

        if (desc & (1 << 18)) {

            /* Supersection.  */

            phys_addr = (desc & 0xff000000) | (address & 0x00ffffff);

        } else {

            /* Section.  */

            phys_addr = (desc & 0xfff00000) | (address & 0x000fffff);

        }

        ap = ((desc >> 10) & 3) | ((desc >> 13) & 4);

        xn = desc & (1 << 4);

        code = 13;

    } else {

        /* Lookup l2 entry.  */

        table = (desc & 0xfffffc00) | ((address >> 10) & 0x3fc);

        desc = ldl_phys(table);

        ap = ((desc >> 4) & 3) | ((desc >> 7) & 4);

        switch (desc & 3) {

        case 0: /* Page translation fault.  */

            code = 7;

            goto do_fault;

        case 1: /* 64k page.  */

            phys_addr = (desc & 0xffff0000) | (address & 0xffff);

            xn = desc & (1 << 15);

            break;

        case 2: case 3: /* 4k page.  */

            phys_addr = (desc & 0xfffff000) | (address & 0xfff);

            xn = desc & 1;

            break;

        default:

            /* Never happens, but compiler isn't smart enough to tell.  */

            abort();

        }

        code = 15;

    }

    if (xn && access_type == 2)

        goto do_fault;



    /* The simplified model uses AP[0] as an access control bit.  */

    if ((env->cp15.c1_sys & (1 << 29)) && (ap & 1) == 0) {

        /* Access flag fault.  */

        code = (code == 15) ? 6 : 3;

        goto do_fault;

    }

    *prot = check_ap(env, ap, domain, access_type, is_user);

    if (!*prot) {

        /* Access permission fault.  */

        goto do_fault;

    }

    *phys_ptr = phys_addr;

    return 0;

do_fault:

    return code | (domain << 4);

}
