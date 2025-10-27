static int get_phys_addr_lpae(CPUARMState *env, target_ulong address,

                              int access_type, int is_user,

                              hwaddr *phys_ptr, int *prot,

                              target_ulong *page_size_ptr)

{

    CPUState *cs = CPU(arm_env_get_cpu(env));

    /* Read an LPAE long-descriptor translation table. */

    MMUFaultType fault_type = translation_fault;

    uint32_t level = 1;

    uint32_t epd;

    int32_t tsz;

    uint32_t tg;

    uint64_t ttbr;

    int ttbr_select;

    hwaddr descaddr, descmask;

    uint32_t tableattrs;

    target_ulong page_size;

    uint32_t attrs;

    int32_t granule_sz = 9;

    int32_t va_size = 32;

    int32_t tbi = 0;



    if (arm_el_is_aa64(env, 1)) {

        va_size = 64;

        if (extract64(address, 55, 1))

            tbi = extract64(env->cp15.c2_control, 38, 1);

        else

            tbi = extract64(env->cp15.c2_control, 37, 1);

        tbi *= 8;

    }



    /* Determine whether this address is in the region controlled by

     * TTBR0 or TTBR1 (or if it is in neither region and should fault).

     * This is a Non-secure PL0/1 stage 1 translation, so controlled by

     * TTBCR/TTBR0/TTBR1 in accordance with ARM ARM DDI0406C table B-32:

     */

    uint32_t t0sz = extract32(env->cp15.c2_control, 0, 6);

    if (arm_el_is_aa64(env, 1)) {

        t0sz = MIN(t0sz, 39);

        t0sz = MAX(t0sz, 16);

    }

    uint32_t t1sz = extract32(env->cp15.c2_control, 16, 6);

    if (arm_el_is_aa64(env, 1)) {

        t1sz = MIN(t1sz, 39);

        t1sz = MAX(t1sz, 16);

    }

    if (t0sz && !extract64(address, va_size - t0sz, t0sz - tbi)) {

        /* there is a ttbr0 region and we are in it (high bits all zero) */

        ttbr_select = 0;

    } else if (t1sz && !extract64(~address, va_size - t1sz, t1sz - tbi)) {

        /* there is a ttbr1 region and we are in it (high bits all one) */

        ttbr_select = 1;

    } else if (!t0sz) {

        /* ttbr0 region is "everything not in the ttbr1 region" */

        ttbr_select = 0;

    } else if (!t1sz) {

        /* ttbr1 region is "everything not in the ttbr0 region" */

        ttbr_select = 1;

    } else {

        /* in the gap between the two regions, this is a Translation fault */

        fault_type = translation_fault;

        goto do_fault;

    }



    /* Note that QEMU ignores shareability and cacheability attributes,

     * so we don't need to do anything with the SH, ORGN, IRGN fields

     * in the TTBCR.  Similarly, TTBCR:A1 selects whether we get the

     * ASID from TTBR0 or TTBR1, but QEMU's TLB doesn't currently

     * implement any ASID-like capability so we can ignore it (instead

     * we will always flush the TLB any time the ASID is changed).

     */

    if (ttbr_select == 0) {

        ttbr = env->cp15.ttbr0_el1;

        epd = extract32(env->cp15.c2_control, 7, 1);

        tsz = t0sz;



        tg = extract32(env->cp15.c2_control, 14, 2);

        if (tg == 1) { /* 64KB pages */

            granule_sz = 13;

        }

        if (tg == 2) { /* 16KB pages */

            granule_sz = 11;

        }

    } else {

        ttbr = env->cp15.ttbr1_el1;

        epd = extract32(env->cp15.c2_control, 23, 1);

        tsz = t1sz;



        tg = extract32(env->cp15.c2_control, 30, 2);

        if (tg == 3)  { /* 64KB pages */

            granule_sz = 13;

        }

        if (tg == 1) { /* 16KB pages */

            granule_sz = 11;

        }

    }



    if (epd) {

        /* Translation table walk disabled => Translation fault on TLB miss */

        goto do_fault;

    }



    /* The starting level depends on the virtual address size which can be

     * up to 48-bits and the translation granule size.

     */

    if ((va_size - tsz) > (granule_sz * 4 + 3)) {

        level = 0;

    } else if ((va_size - tsz) > (granule_sz * 3 + 3)) {

        level = 1;

    } else {

        level = 2;

    }



    /* Clear the vaddr bits which aren't part of the within-region address,

     * so that we don't have to special case things when calculating the

     * first descriptor address.

     */

    if (tsz) {

        address &= (1ULL << (va_size - tsz)) - 1;

    }



    descmask = (1ULL << (granule_sz + 3)) - 1;



    /* Now we can extract the actual base address from the TTBR */

    descaddr = extract64(ttbr, 0, 48);

    descaddr &= ~((1ULL << (va_size - tsz - (granule_sz * (4 - level)))) - 1);



    tableattrs = 0;

    for (;;) {

        uint64_t descriptor;



        descaddr |= (address >> (granule_sz * (4 - level))) & descmask;

        descaddr &= ~7ULL;

        descriptor = ldq_phys(cs->as, descaddr);

        if (!(descriptor & 1) ||

            (!(descriptor & 2) && (level == 3))) {

            /* Invalid, or the Reserved level 3 encoding */

            goto do_fault;

        }

        descaddr = descriptor & 0xfffffff000ULL;



        if ((descriptor & 2) && (level < 3)) {

            /* Table entry. The top five bits are attributes which  may

             * propagate down through lower levels of the table (and

             * which are all arranged so that 0 means "no effect", so

             * we can gather them up by ORing in the bits at each level).

             */

            tableattrs |= extract64(descriptor, 59, 5);

            level++;

            continue;

        }

        /* Block entry at level 1 or 2, or page entry at level 3.

         * These are basically the same thing, although the number

         * of bits we pull in from the vaddr varies.

         */

        page_size = (1 << ((granule_sz * (4 - level)) + 3));

        descaddr |= (address & (page_size - 1));

        /* Extract attributes from the descriptor and merge with table attrs */

        if (arm_feature(env, ARM_FEATURE_V8)) {

            attrs = extract64(descriptor, 2, 10)

                | (extract64(descriptor, 53, 11) << 10);

        } else {

            attrs = extract64(descriptor, 2, 10)

                | (extract64(descriptor, 52, 12) << 10);

        }

        attrs |= extract32(tableattrs, 0, 2) << 11; /* XN, PXN */

        attrs |= extract32(tableattrs, 3, 1) << 5; /* APTable[1] => AP[2] */

        /* The sense of AP[1] vs APTable[0] is reversed, as APTable[0] == 1

         * means "force PL1 access only", which means forcing AP[1] to 0.

         */

        if (extract32(tableattrs, 2, 1)) {

            attrs &= ~(1 << 4);

        }

        /* Since we're always in the Non-secure state, NSTable is ignored. */

        break;

    }

    /* Here descaddr is the final physical address, and attributes

     * are all in attrs.

     */

    fault_type = access_fault;

    if ((attrs & (1 << 8)) == 0) {

        /* Access flag */

        goto do_fault;

    }

    fault_type = permission_fault;

    if (is_user && !(attrs & (1 << 4))) {

        /* Unprivileged access not enabled */

        goto do_fault;

    }

    *prot = PAGE_READ | PAGE_WRITE | PAGE_EXEC;

    if (attrs & (1 << 12) || (!is_user && (attrs & (1 << 11)))) {

        /* XN or PXN */

        if (access_type == 2) {

            goto do_fault;

        }

        *prot &= ~PAGE_EXEC;

    }

    if (attrs & (1 << 5)) {

        /* Write access forbidden */

        if (access_type == 1) {

            goto do_fault;

        }

        *prot &= ~PAGE_WRITE;

    }



    *phys_ptr = descaddr;

    *page_size_ptr = page_size;

    return 0;



do_fault:

    /* Long-descriptor format IFSR/DFSR value */

    return (1 << 9) | (fault_type << 2) | level;

}
