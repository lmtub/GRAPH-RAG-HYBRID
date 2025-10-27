static int get_phys_addr_v5(CPUState *env, uint32_t address, int access_type,

			    int is_user, uint32_t *phys_ptr, int *prot)

{

    int code;

    uint32_t table;

    uint32_t desc;

    int type;

    int ap;

    int domain;

    uint32_t phys_addr;



    /* Pagetable walk.  */

    /* Lookup l1 descriptor.  */

    table = get_level1_table_address(env, address);

    desc = ldl_phys(table);

    type = (desc & 3);

    domain = (env->cp15.c3 >> ((desc >> 4) & 0x1e)) & 3;

    if (type == 0) {

        /* Section translation fault.  */

        code = 5;

        goto do_fault;

    }

    if (domain == 0 || domain == 2) {

        if (type == 2)

            code = 9; /* Section domain fault.  */

        else

            code = 11; /* Page domain fault.  */

        goto do_fault;

    }

    if (type == 2) {

        /* 1Mb section.  */

        phys_addr = (desc & 0xfff00000) | (address & 0x000fffff);

        ap = (desc >> 10) & 3;

        code = 13;

    } else {

        /* Lookup l2 entry.  */

	if (type == 1) {

	    /* Coarse pagetable.  */

	    table = (desc & 0xfffffc00) | ((address >> 10) & 0x3fc);

	} else {

	    /* Fine pagetable.  */

	    table = (desc & 0xfffff000) | ((address >> 8) & 0xffc);

	}

        desc = ldl_phys(table);

        switch (desc & 3) {

        case 0: /* Page translation fault.  */

            code = 7;

            goto do_fault;

        case 1: /* 64k page.  */

            phys_addr = (desc & 0xffff0000) | (address & 0xffff);

            ap = (desc >> (4 + ((address >> 13) & 6))) & 3;

            break;

        case 2: /* 4k page.  */

            phys_addr = (desc & 0xfffff000) | (address & 0xfff);

            ap = (desc >> (4 + ((address >> 13) & 6))) & 3;

            break;

        case 3: /* 1k page.  */

	    if (type == 1) {

		if (arm_feature(env, ARM_FEATURE_XSCALE)) {

		    phys_addr = (desc & 0xfffff000) | (address & 0xfff);

		} else {

		    /* Page translation fault.  */

		    code = 7;

		    goto do_fault;

		}

	    } else {

		phys_addr = (desc & 0xfffffc00) | (address & 0x3ff);

	    }

            ap = (desc >> 4) & 3;

            break;

        default:

            /* Never happens, but compiler isn't smart enough to tell.  */

            abort();

        }

        code = 15;

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
