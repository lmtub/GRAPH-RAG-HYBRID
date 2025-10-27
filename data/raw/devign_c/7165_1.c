uint64_t helper_ld_asi(target_ulong addr, int asi, int size, int sign)

{

    uint64_t ret = 0;

#if defined(DEBUG_ASI)

    target_ulong last_addr = addr;

#endif



    asi &= 0xff;



    if ((asi < 0x80 && (env->pstate & PS_PRIV) == 0)

        || ((env->def->features & CPU_FEATURE_HYPV)

            && asi >= 0x30 && asi < 0x80

            && !(env->hpstate & HS_PRIV)))

        raise_exception(TT_PRIV_ACT);



    helper_check_align(addr, size - 1);

    switch (asi) {

    case 0x82: // Primary no-fault

    case 0x8a: // Primary no-fault LE

    case 0x83: // Secondary no-fault

    case 0x8b: // Secondary no-fault LE

        {

            /* secondary space access has lowest asi bit equal to 1 */

            int access_mmu_idx = ( asi & 1 ) ? MMU_KERNEL_IDX

                                             : MMU_KERNEL_SECONDARY_IDX;



            if (cpu_get_phys_page_nofault(env, addr, access_mmu_idx) == -1ULL) {

#ifdef DEBUG_ASI

                dump_asi("read ", last_addr, asi, size, ret);

#endif

                return 0;

            }

        }

        // Fall through

    case 0x10: // As if user primary

    case 0x11: // As if user secondary

    case 0x18: // As if user primary LE

    case 0x19: // As if user secondary LE

    case 0x80: // Primary

    case 0x81: // Secondary

    case 0x88: // Primary LE

    case 0x89: // Secondary LE

    case 0xe2: // UA2007 Primary block init

    case 0xe3: // UA2007 Secondary block init

        if ((asi & 0x80) && (env->pstate & PS_PRIV)) {

            if ((env->def->features & CPU_FEATURE_HYPV)

                && env->hpstate & HS_PRIV) {

                switch(size) {

                case 1:

                    ret = ldub_hypv(addr);

                    break;

                case 2:

                    ret = lduw_hypv(addr);

                    break;

                case 4:

                    ret = ldl_hypv(addr);

                    break;

                default:

                case 8:

                    ret = ldq_hypv(addr);

                    break;

                }

            } else {

                /* secondary space access has lowest asi bit equal to 1 */

                if (asi & 1) {

                    switch(size) {

                    case 1:

                        ret = ldub_kernel_secondary(addr);

                        break;

                    case 2:

                        ret = lduw_kernel_secondary(addr);

                        break;

                    case 4:

                        ret = ldl_kernel_secondary(addr);

                        break;

                    default:

                    case 8:

                        ret = ldq_kernel_secondary(addr);

                        break;

                    }

                } else {

                    switch(size) {

                    case 1:

                        ret = ldub_kernel(addr);

                        break;

                    case 2:

                        ret = lduw_kernel(addr);

                        break;

                    case 4:

                        ret = ldl_kernel(addr);

                        break;

                    default:

                    case 8:

                        ret = ldq_kernel(addr);

                        break;

                    }

                }

            }

        } else {

            /* secondary space access has lowest asi bit equal to 1 */

            if (asi & 1) {

                switch(size) {

                case 1:

                    ret = ldub_user_secondary(addr);

                    break;

                case 2:

                    ret = lduw_user_secondary(addr);

                    break;

                case 4:

                    ret = ldl_user_secondary(addr);

                    break;

                default:

                case 8:

                    ret = ldq_user_secondary(addr);

                    break;

                }

            } else {

                switch(size) {

                case 1:

                    ret = ldub_user(addr);

                    break;

                case 2:

                    ret = lduw_user(addr);

                    break;

                case 4:

                    ret = ldl_user(addr);

                    break;

                default:

                case 8:

                    ret = ldq_user(addr);

                    break;

                }

            }

        }

        break;

    case 0x14: // Bypass

    case 0x15: // Bypass, non-cacheable

    case 0x1c: // Bypass LE

    case 0x1d: // Bypass, non-cacheable LE

        {

            switch(size) {

            case 1:

                ret = ldub_phys(addr);

                break;

            case 2:

                ret = lduw_phys(addr);

                break;

            case 4:

                ret = ldl_phys(addr);

                break;

            default:

            case 8:

                ret = ldq_phys(addr);

                break;

            }

            break;

        }

    case 0x24: // Nucleus quad LDD 128 bit atomic

    case 0x2c: // Nucleus quad LDD 128 bit atomic LE

        //  Only ldda allowed

        raise_exception(TT_ILL_INSN);

        return 0;

    case 0x04: // Nucleus

    case 0x0c: // Nucleus Little Endian (LE)

    {

        switch(size) {

        case 1:

            ret = ldub_nucleus(addr);

            break;

        case 2:

            ret = lduw_nucleus(addr);

            break;

        case 4:

            ret = ldl_nucleus(addr);

            break;

        default:

        case 8:

            ret = ldq_nucleus(addr);

            break;

        }

        break;

    }

    case 0x4a: // UPA config

        // XXX

        break;

    case 0x45: // LSU

        ret = env->lsu;

        break;

    case 0x50: // I-MMU regs

        {

            int reg = (addr >> 3) & 0xf;



            if (reg == 0) {

                // I-TSB Tag Target register

                ret = ultrasparc_tag_target(env->immu.tag_access);

            } else {

                ret = env->immuregs[reg];

            }



            break;

        }

    case 0x51: // I-MMU 8k TSB pointer

        {

            // env->immuregs[5] holds I-MMU TSB register value

            // env->immuregs[6] holds I-MMU Tag Access register value

            ret = ultrasparc_tsb_pointer(env->immu.tsb, env->immu.tag_access,

                                         8*1024);

            break;

        }

    case 0x52: // I-MMU 64k TSB pointer

        {

            // env->immuregs[5] holds I-MMU TSB register value

            // env->immuregs[6] holds I-MMU Tag Access register value

            ret = ultrasparc_tsb_pointer(env->immu.tsb, env->immu.tag_access,

                                         64*1024);

            break;

        }

    case 0x55: // I-MMU data access

        {

            int reg = (addr >> 3) & 0x3f;



            ret = env->itlb[reg].tte;

            break;

        }

    case 0x56: // I-MMU tag read

        {

            int reg = (addr >> 3) & 0x3f;



            ret = env->itlb[reg].tag;

            break;

        }

    case 0x58: // D-MMU regs

        {

            int reg = (addr >> 3) & 0xf;



            if (reg == 0) {

                // D-TSB Tag Target register

                ret = ultrasparc_tag_target(env->dmmu.tag_access);

            } else {

                ret = env->dmmuregs[reg];

            }

            break;

        }

    case 0x59: // D-MMU 8k TSB pointer

        {

            // env->dmmuregs[5] holds D-MMU TSB register value

            // env->dmmuregs[6] holds D-MMU Tag Access register value

            ret = ultrasparc_tsb_pointer(env->dmmu.tsb, env->dmmu.tag_access,

                                         8*1024);

            break;

        }

    case 0x5a: // D-MMU 64k TSB pointer

        {

            // env->dmmuregs[5] holds D-MMU TSB register value

            // env->dmmuregs[6] holds D-MMU Tag Access register value

            ret = ultrasparc_tsb_pointer(env->dmmu.tsb, env->dmmu.tag_access,

                                         64*1024);

            break;

        }

    case 0x5d: // D-MMU data access

        {

            int reg = (addr >> 3) & 0x3f;



            ret = env->dtlb[reg].tte;

            break;

        }

    case 0x5e: // D-MMU tag read

        {

            int reg = (addr >> 3) & 0x3f;



            ret = env->dtlb[reg].tag;

            break;

        }

    case 0x46: // D-cache data

    case 0x47: // D-cache tag access

    case 0x4b: // E-cache error enable

    case 0x4c: // E-cache asynchronous fault status

    case 0x4d: // E-cache asynchronous fault address

    case 0x4e: // E-cache tag data

    case 0x66: // I-cache instruction access

    case 0x67: // I-cache tag access

    case 0x6e: // I-cache predecode

    case 0x6f: // I-cache LRU etc.

    case 0x76: // E-cache tag

    case 0x7e: // E-cache tag

        break;

    case 0x5b: // D-MMU data pointer

    case 0x48: // Interrupt dispatch, RO

    case 0x49: // Interrupt data receive

    case 0x7f: // Incoming interrupt vector, RO

        // XXX

        break;

    case 0x54: // I-MMU data in, WO

    case 0x57: // I-MMU demap, WO

    case 0x5c: // D-MMU data in, WO

    case 0x5f: // D-MMU demap, WO

    case 0x77: // Interrupt vector, WO

    default:

        do_unassigned_access(addr, 0, 0, 1, size);

        ret = 0;

        break;

    }



    /* Convert from little endian */

    switch (asi) {

    case 0x0c: // Nucleus Little Endian (LE)

    case 0x18: // As if user primary LE

    case 0x19: // As if user secondary LE

    case 0x1c: // Bypass LE

    case 0x1d: // Bypass, non-cacheable LE

    case 0x88: // Primary LE

    case 0x89: // Secondary LE

    case 0x8a: // Primary no-fault LE

    case 0x8b: // Secondary no-fault LE

        switch(size) {

        case 2:

            ret = bswap16(ret);

            break;

        case 4:

            ret = bswap32(ret);

            break;

        case 8:

            ret = bswap64(ret);

            break;

        default:

            break;

        }

    default:

        break;

    }



    /* Convert to signed number */

    if (sign) {

        switch(size) {

        case 1:

            ret = (int8_t) ret;

            break;

        case 2:

            ret = (int16_t) ret;

            break;

        case 4:

            ret = (int32_t) ret;

            break;

        default:

            break;

        }

    }

#ifdef DEBUG_ASI

    dump_asi("read ", last_addr, asi, size, ret);

#endif

    return ret;

}
