static bool get_phys_addr_lpae(CPUARMState *env, target_ulong address,

                               int access_type, ARMMMUIdx mmu_idx,

                               hwaddr *phys_ptr, MemTxAttrs *txattrs, int *prot,

                               target_ulong *page_size_ptr, uint32_t *fsr,

                               ARMMMUFaultInfo *fi)

{

    ARMCPU *cpu = arm_env_get_cpu(env);

    CPUState *cs = CPU(cpu);

    /* Read an LPAE long-descriptor translation table. */

    MMUFaultType fault_type = translation_fault;

    uint32_t level;

    uint32_t epd = 0;

    int32_t t0sz, t1sz;

    uint32_t tg;

    uint64_t ttbr;

    int ttbr_select;

    hwaddr descaddr, indexmask, indexmask_grainsize;

    uint32_t tableattrs;

    target_ulong page_size;

    uint32_t attrs;

    int32_t stride = 9;

    int32_t va_size;

    int inputsize;

    int32_t tbi = 0;

    TCR *tcr = regime_tcr(env, mmu_idx);

    int ap, ns, xn, pxn;

    uint32_t el = regime_el(env, mmu_idx);

    bool ttbr1_valid = true;

    uint64_t descaddrmask;



    /* TODO:

     * This code does not handle the different format TCR for VTCR_EL2.

     * This code also does not support shareability levels.

     * Attribute and permission bit handling should also be checked when adding

     * support for those page table walks.

     */

    if (arm_el_is_aa64(env, el)) {

        level = 0;

        va_size = 64;

        if (el > 1) {

            if (mmu_idx != ARMMMUIdx_S2NS) {

                tbi = extract64(tcr->raw_tcr, 20, 1);

            }

        } else {

            if (extract64(address, 55, 1)) {

                tbi = extract64(tcr->raw_tcr, 38, 1);

            } else {

                tbi = extract64(tcr->raw_tcr, 37, 1);

            }

        }

        tbi *= 8;



        /* If we are in 64-bit EL2 or EL3 then there is no TTBR1, so mark it

         * invalid.

         */

        if (el > 1) {

            ttbr1_valid = false;

        }

    } else {

        level = 1;

        va_size = 32;

        /* There is no TTBR1 for EL2 */

        if (el == 2) {

            ttbr1_valid = false;

        }

    }



    /* Determine whether this address is in the region controlled by

     * TTBR0 or TTBR1 (or if it is in neither region and should fault).

     * This is a Non-secure PL0/1 stage 1 translation, so controlled by

     * TTBCR/TTBR0/TTBR1 in accordance with ARM ARM DDI0406C table B-32:

     */

    if (va_size == 64) {

        /* AArch64 translation.  */

        t0sz = extract32(tcr->raw_tcr, 0, 6);

        t0sz = MIN(t0sz, 39);

        t0sz = MAX(t0sz, 16);

    } else if (mmu_idx != ARMMMUIdx_S2NS) {

        /* AArch32 stage 1 translation.  */

        t0sz = extract32(tcr->raw_tcr, 0, 3);

    } else {

        /* AArch32 stage 2 translation.  */

        bool sext = extract32(tcr->raw_tcr, 4, 1);

        bool sign = extract32(tcr->raw_tcr, 3, 1);

        t0sz = sextract32(tcr->raw_tcr, 0, 4);



        /* If the sign-extend bit is not the same as t0sz[3], the result

         * is unpredictable. Flag this as a guest error.  */

        if (sign != sext) {

            qemu_log_mask(LOG_GUEST_ERROR,

                          "AArch32: VTCR.S / VTCR.T0SZ[3] missmatch\n");

        }

    }

    t1sz = extract32(tcr->raw_tcr, 16, 6);

    if (va_size == 64) {

        t1sz = MIN(t1sz, 39);

        t1sz = MAX(t1sz, 16);

    }

    if (t0sz && !extract64(address, va_size - t0sz, t0sz - tbi)) {

        /* there is a ttbr0 region and we are in it (high bits all zero) */

        ttbr_select = 0;

    } else if (ttbr1_valid && t1sz &&

               !extract64(~address, va_size - t1sz, t1sz - tbi)) {

        /* there is a ttbr1 region and we are in it (high bits all one) */

        ttbr_select = 1;

    } else if (!t0sz) {

        /* ttbr0 region is "everything not in the ttbr1 region" */

        ttbr_select = 0;

    } else if (!t1sz && ttbr1_valid) {

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

        ttbr = regime_ttbr(env, mmu_idx, 0);

        if (el < 2) {

            epd = extract32(tcr->raw_tcr, 7, 1);

        }

        inputsize = va_size - t0sz;



        tg = extract32(tcr->raw_tcr, 14, 2);

        if (tg == 1) { /* 64KB pages */

            stride = 13;

        }

        if (tg == 2) { /* 16KB pages */

            stride = 11;

        }

    } else {

        /* We should only be here if TTBR1 is valid */

        assert(ttbr1_valid);



        ttbr = regime_ttbr(env, mmu_idx, 1);

        epd = extract32(tcr->raw_tcr, 23, 1);

        inputsize = va_size - t1sz;



        tg = extract32(tcr->raw_tcr, 30, 2);

        if (tg == 3)  { /* 64KB pages */

            stride = 13;

        }

        if (tg == 1) { /* 16KB pages */

            stride = 11;

        }

    }



    /* Here we should have set up all the parameters for the translation:

     * va_size, inputsize, ttbr, epd, stride, tbi

     */



    if (epd) {

        /* Translation table walk disabled => Translation fault on TLB miss

         * Note: This is always 0 on 64-bit EL2 and EL3.

         */

        goto do_fault;

    }



    if (mmu_idx != ARMMMUIdx_S2NS) {

        /* The starting level depends on the virtual address size (which can

         * be up to 48 bits) and the translation granule size. It indicates

         * the number of strides (stride bits at a time) needed to

         * consume the bits of the input address. In the pseudocode this is:

         *  level = 4 - RoundUp((inputsize - grainsize) / stride)

         * where their 'inputsize' is our 'inputsize', 'grainsize' is

         * our 'stride + 3' and 'stride' is our 'stride'.

         * Applying the usual "rounded up m/n is (m+n-1)/n" and simplifying:

         * = 4 - (inputsize - stride - 3 + stride - 1) / stride

         * = 4 - (inputsize - 4) / stride;

         */

        level = 4 - (inputsize - 4) / stride;

    } else {

        /* For stage 2 translations the starting level is specified by the

         * VTCR_EL2.SL0 field (whose interpretation depends on the page size)

         */

        uint32_t sl0 = extract32(tcr->raw_tcr, 6, 2);

        uint32_t startlevel;

        bool ok;



        if (va_size == 32 || stride == 9) {

            /* AArch32 or 4KB pages */

            startlevel = 2 - sl0;

        } else {

            /* 16KB or 64KB pages */

            startlevel = 3 - sl0;

        }



        /* Check that the starting level is valid. */

        ok = check_s2_mmu_setup(cpu, va_size == 64, startlevel,

                                inputsize, stride);

        if (!ok) {

            fault_type = translation_fault;

            goto do_fault;

        }

        level = startlevel;

    }



    indexmask_grainsize = (1ULL << (stride + 3)) - 1;

    indexmask = (1ULL << (inputsize - (stride * (4 - level)))) - 1;



    /* Now we can extract the actual base address from the TTBR */

    descaddr = extract64(ttbr, 0, 48);

    descaddr &= ~indexmask;



    /* The address field in the descriptor goes up to bit 39 for ARMv7

     * but up to bit 47 for ARMv8, but we use the descaddrmask

     * up to bit 39 for AArch32, because we don't need other bits in that case

     * to construct next descriptor address (anyway they should be all zeroes).

     */

    descaddrmask = ((1ull << (va_size == 64 ? 48 : 40)) - 1) &

                   ~indexmask_grainsize;



    /* Secure accesses start with the page table in secure memory and

     * can be downgraded to non-secure at any step. Non-secure accesses

     * remain non-secure. We implement this by just ORing in the NSTable/NS

     * bits at each step.

     */

    tableattrs = regime_is_secure(env, mmu_idx) ? 0 : (1 << 4);

    for (;;) {

        uint64_t descriptor;

        bool nstable;



        descaddr |= (address >> (stride * (4 - level))) & indexmask;

        descaddr &= ~7ULL;

        nstable = extract32(tableattrs, 4, 1);

        descriptor = arm_ldq_ptw(cs, descaddr, !nstable, mmu_idx, fsr, fi);

        if (fi->s1ptw) {

            goto do_fault;

        }



        if (!(descriptor & 1) ||

            (!(descriptor & 2) && (level == 3))) {

            /* Invalid, or the Reserved level 3 encoding */

            goto do_fault;

        }

        descaddr = descriptor & descaddrmask;



        if ((descriptor & 2) && (level < 3)) {

            /* Table entry. The top five bits are attributes which  may

             * propagate down through lower levels of the table (and

             * which are all arranged so that 0 means "no effect", so

             * we can gather them up by ORing in the bits at each level).

             */

            tableattrs |= extract64(descriptor, 59, 5);

            level++;

            indexmask = indexmask_grainsize;

            continue;

        }

        /* Block entry at level 1 or 2, or page entry at level 3.

         * These are basically the same thing, although the number

         * of bits we pull in from the vaddr varies.

         */

        page_size = (1ULL << ((stride * (4 - level)) + 3));

        descaddr |= (address & (page_size - 1));

        /* Extract attributes from the descriptor */

        attrs = extract64(descriptor, 2, 10)

            | (extract64(descriptor, 52, 12) << 10);



        if (mmu_idx == ARMMMUIdx_S2NS) {

            /* Stage 2 table descriptors do not include any attribute fields */

            break;

        }

        /* Merge in attributes from table descriptors */

        attrs |= extract32(tableattrs, 0, 2) << 11; /* XN, PXN */

        attrs |= extract32(tableattrs, 3, 1) << 5; /* APTable[1] => AP[2] */

        /* The sense of AP[1] vs APTable[0] is reversed, as APTable[0] == 1

         * means "force PL1 access only", which means forcing AP[1] to 0.

         */

        if (extract32(tableattrs, 2, 1)) {

            attrs &= ~(1 << 4);

        }

        attrs |= nstable << 3; /* NS */

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



    ap = extract32(attrs, 4, 2);

    xn = extract32(attrs, 12, 1);



    if (mmu_idx == ARMMMUIdx_S2NS) {

        ns = true;

        *prot = get_S2prot(env, ap, xn);

    } else {

        ns = extract32(attrs, 3, 1);

        pxn = extract32(attrs, 11, 1);

        *prot = get_S1prot(env, mmu_idx, va_size == 64, ap, ns, xn, pxn);

    }



    fault_type = permission_fault;

    if (!(*prot & (1 << access_type))) {

        goto do_fault;

    }



    if (ns) {

        /* The NS bit will (as required by the architecture) have no effect if

         * the CPU doesn't support TZ or this is a non-secure translation

         * regime, because the attribute will already be non-secure.

         */

        txattrs->secure = false;

    }

    *phys_ptr = descaddr;

    *page_size_ptr = page_size;

    return false;



do_fault:

    /* Long-descriptor format IFSR/DFSR value */

    *fsr = (1 << 9) | (fault_type << 2) | level;

    /* Tag the error as S2 for failed S1 PTW at S2 or ordinary S2.  */

    fi->stage2 = fi->s1ptw || (mmu_idx == ARMMMUIdx_S2NS);

    return true;

}
