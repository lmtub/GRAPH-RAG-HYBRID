void helper_st_asi(target_ulong addr, target_ulong val, int asi, int size)

{

#ifdef DEBUG_ASI

    dump_asi("write", addr, asi, size, val);

#endif



    asi &= 0xff;



    if ((asi < 0x80 && (env->pstate & PS_PRIV) == 0)

        || ((env->def->features & CPU_FEATURE_HYPV)

            && asi >= 0x30 && asi < 0x80

            && !(env->hpstate & HS_PRIV)))

        raise_exception(TT_PRIV_ACT);



    helper_check_align(addr, size - 1);

    /* Convert to little endian */

    switch (asi) {

    case 0x0c: // Nucleus Little Endian (LE)

    case 0x18: // As if user primary LE

    case 0x19: // As if user secondary LE

    case 0x1c: // Bypass LE

    case 0x1d: // Bypass, non-cacheable LE

    case 0x88: // Primary LE

    case 0x89: // Secondary LE

        switch(size) {

        case 2:

            val = bswap16(val);

            break;

        case 4:

            val = bswap32(val);

            break;

        case 8:

            val = bswap64(val);

            break;

        default:

            break;

        }

    default:

        break;

    }



    switch(asi) {

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

                    stb_hypv(addr, val);

                    break;

                case 2:

                    stw_hypv(addr, val);

                    break;

                case 4:

                    stl_hypv(addr, val);

                    break;

                case 8:

                default:

                    stq_hypv(addr, val);

                    break;

                }

            } else {

                /* secondary space access has lowest asi bit equal to 1 */

                if (asi & 1) {

                    switch(size) {

                    case 1:

                        stb_kernel_secondary(addr, val);

                        break;

                    case 2:

                        stw_kernel_secondary(addr, val);

                        break;

                    case 4:

                        stl_kernel_secondary(addr, val);

                        break;

                    case 8:

                    default:

                        stq_kernel_secondary(addr, val);

                        break;

                    }

                } else {

                    switch(size) {

                    case 1:

                        stb_kernel(addr, val);

                        break;

                    case 2:

                        stw_kernel(addr, val);

                        break;

                    case 4:

                        stl_kernel(addr, val);

                        break;

                    case 8:

                    default:

                        stq_kernel(addr, val);

                        break;

                    }

                }

            }

        } else {

            /* secondary space access has lowest asi bit equal to 1 */

            if (asi & 1) {

                switch(size) {

                case 1:

                    stb_user_secondary(addr, val);

                    break;

                case 2:

                    stw_user_secondary(addr, val);

                    break;

                case 4:

                    stl_user_secondary(addr, val);

                    break;

                case 8:

                default:

                    stq_user_secondary(addr, val);

                    break;

                }

            } else {

                switch(size) {

                case 1:

                    stb_user(addr, val);

                    break;

                case 2:

                    stw_user(addr, val);

                    break;

                case 4:

                    stl_user(addr, val);

                    break;

                case 8:

                default:

                    stq_user(addr, val);

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

                stb_phys(addr, val);

                break;

            case 2:

                stw_phys(addr, val);

                break;

            case 4:

                stl_phys(addr, val);

                break;

            case 8:

            default:

                stq_phys(addr, val);

                break;

            }

        }

        return;

    case 0x24: // Nucleus quad LDD 128 bit atomic

    case 0x2c: // Nucleus quad LDD 128 bit atomic LE

        //  Only ldda allowed

        raise_exception(TT_ILL_INSN);

        return;

    case 0x04: // Nucleus

    case 0x0c: // Nucleus Little Endian (LE)

    {

        switch(size) {

        case 1:

            stb_nucleus(addr, val);

            break;

        case 2:

            stw_nucleus(addr, val);

            break;

        case 4:

            stl_nucleus(addr, val);

            break;

        default:

        case 8:

            stq_nucleus(addr, val);

            break;

        }

        break;

    }



    case 0x4a: // UPA config

        // XXX

        return;

    case 0x45: // LSU

        {

            uint64_t oldreg;



            oldreg = env->lsu;

            env->lsu = val & (DMMU_E | IMMU_E);

            // Mappings generated during D/I MMU disabled mode are

            // invalid in normal mode

            if (oldreg != env->lsu) {

                DPRINTF_MMU("LSU change: 0x%" PRIx64 " -> 0x%" PRIx64 "\n",

                            oldreg, env->lsu);

#ifdef DEBUG_MMU

                dump_mmu(env);

#endif

                tlb_flush(env, 1);

            }

            return;

        }

    case 0x50: // I-MMU regs

        {

            int reg = (addr >> 3) & 0xf;

            uint64_t oldreg;



            oldreg = env->immuregs[reg];

            switch(reg) {

            case 0: // RO

                return;

            case 1: // Not in I-MMU

            case 2:

                return;

            case 3: // SFSR

                if ((val & 1) == 0)

                    val = 0; // Clear SFSR

                env->immu.sfsr = val;

                break;

            case 4: // RO

                return;

            case 5: // TSB access

                DPRINTF_MMU("immu TSB write: 0x%016" PRIx64 " -> 0x%016"

                            PRIx64 "\n", env->immu.tsb, val);

                env->immu.tsb = val;

                break;

            case 6: // Tag access

                env->immu.tag_access = val;

                break;

            case 7:

            case 8:

                return;

            default:

                break;

            }



            if (oldreg != env->immuregs[reg]) {

                DPRINTF_MMU("immu change reg[%d]: 0x%016" PRIx64 " -> 0x%016"

                            PRIx64 "\n", reg, oldreg, env->immuregs[reg]);

            }

#ifdef DEBUG_MMU

            dump_mmu(env);

#endif

            return;

        }

    case 0x54: // I-MMU data in

        replace_tlb_1bit_lru(env->itlb, env->immu.tag_access, val, "immu", env);

        return;

    case 0x55: // I-MMU data access

        {

            // TODO: auto demap



            unsigned int i = (addr >> 3) & 0x3f;



            replace_tlb_entry(&env->itlb[i], env->immu.tag_access, val, env);



#ifdef DEBUG_MMU

            DPRINTF_MMU("immu data access replaced entry [%i]\n", i);

            dump_mmu(env);

#endif

            return;

        }

    case 0x57: // I-MMU demap

        demap_tlb(env->itlb, addr, "immu", env);

        return;

    case 0x58: // D-MMU regs

        {

            int reg = (addr >> 3) & 0xf;

            uint64_t oldreg;



            oldreg = env->dmmuregs[reg];

            switch(reg) {

            case 0: // RO

            case 4:

                return;

            case 3: // SFSR

                if ((val & 1) == 0) {

                    val = 0; // Clear SFSR, Fault address

                    env->dmmu.sfar = 0;

                }

                env->dmmu.sfsr = val;

                break;

            case 1: // Primary context

                env->dmmu.mmu_primary_context = val;

                break;

            case 2: // Secondary context

                env->dmmu.mmu_secondary_context = val;

                break;

            case 5: // TSB access

                DPRINTF_MMU("dmmu TSB write: 0x%016" PRIx64 " -> 0x%016"

                            PRIx64 "\n", env->dmmu.tsb, val);

                env->dmmu.tsb = val;

                break;

            case 6: // Tag access

                env->dmmu.tag_access = val;

                break;

            case 7: // Virtual Watchpoint

            case 8: // Physical Watchpoint

            default:

                env->dmmuregs[reg] = val;

                break;

            }



            if (oldreg != env->dmmuregs[reg]) {

                DPRINTF_MMU("dmmu change reg[%d]: 0x%016" PRIx64 " -> 0x%016"

                            PRIx64 "\n", reg, oldreg, env->dmmuregs[reg]);

            }

#ifdef DEBUG_MMU

            dump_mmu(env);

#endif

            return;

        }

    case 0x5c: // D-MMU data in

        replace_tlb_1bit_lru(env->dtlb, env->dmmu.tag_access, val, "dmmu", env);

        return;

    case 0x5d: // D-MMU data access

        {

            unsigned int i = (addr >> 3) & 0x3f;



            replace_tlb_entry(&env->dtlb[i], env->dmmu.tag_access, val, env);



#ifdef DEBUG_MMU

            DPRINTF_MMU("dmmu data access replaced entry [%i]\n", i);

            dump_mmu(env);

#endif

            return;

        }

    case 0x5f: // D-MMU demap

        demap_tlb(env->dtlb, addr, "dmmu", env);

        return;

    case 0x49: // Interrupt data receive

        // XXX

        return;

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

        return;

    case 0x51: // I-MMU 8k TSB pointer, RO

    case 0x52: // I-MMU 64k TSB pointer, RO

    case 0x56: // I-MMU tag read, RO

    case 0x59: // D-MMU 8k TSB pointer, RO

    case 0x5a: // D-MMU 64k TSB pointer, RO

    case 0x5b: // D-MMU data pointer, RO

    case 0x5e: // D-MMU tag read, RO

    case 0x48: // Interrupt dispatch, RO

    case 0x7f: // Incoming interrupt vector, RO

    case 0x82: // Primary no-fault, RO

    case 0x83: // Secondary no-fault, RO

    case 0x8a: // Primary no-fault LE, RO

    case 0x8b: // Secondary no-fault LE, RO

    default:

        do_unassigned_access(addr, 1, 0, 1, size);

        return;

    }

}
