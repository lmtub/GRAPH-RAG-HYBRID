void cpu_x86_cpuid(CPUX86State *env, uint32_t index, uint32_t count,

                   uint32_t *eax, uint32_t *ebx,

                   uint32_t *ecx, uint32_t *edx)

{

    X86CPU *cpu = x86_env_get_cpu(env);

    CPUState *cs = CPU(cpu);

    uint32_t pkg_offset;



    /* test if maximum index reached */

    if (index & 0x80000000) {

        if (index > env->cpuid_xlevel) {

            if (env->cpuid_xlevel2 > 0) {

                /* Handle the Centaur's CPUID instruction. */

                if (index > env->cpuid_xlevel2) {

                    index = env->cpuid_xlevel2;

                } else if (index < 0xC0000000) {

                    index = env->cpuid_xlevel;

                }

            } else {

                /* Intel documentation states that invalid EAX input will

                 * return the same information as EAX=cpuid_level

                 * (Intel SDM Vol. 2A - Instruction Set Reference - CPUID)

                 */

                index =  env->cpuid_level;

            }

        }

    } else {

        if (index > env->cpuid_level)

            index = env->cpuid_level;

    }



    switch(index) {

    case 0:

        *eax = env->cpuid_level;

        *ebx = env->cpuid_vendor1;

        *edx = env->cpuid_vendor2;

        *ecx = env->cpuid_vendor3;

        break;

    case 1:

        *eax = env->cpuid_version;

        *ebx = (cpu->apic_id << 24) |

               8 << 8; /* CLFLUSH size in quad words, Linux wants it. */

        *ecx = env->features[FEAT_1_ECX];

        if ((*ecx & CPUID_EXT_XSAVE) && (env->cr[4] & CR4_OSXSAVE_MASK)) {

            *ecx |= CPUID_EXT_OSXSAVE;

        }

        *edx = env->features[FEAT_1_EDX];

        if (cs->nr_cores * cs->nr_threads > 1) {

            *ebx |= (cs->nr_cores * cs->nr_threads) << 16;

            *edx |= CPUID_HT;

        }

        break;

    case 2:

        /* cache info: needed for Pentium Pro compatibility */

        if (cpu->cache_info_passthrough) {

            host_cpuid(index, 0, eax, ebx, ecx, edx);

            break;

        }

        *eax = 1; /* Number of CPUID[EAX=2] calls required */

        *ebx = 0;

        if (!cpu->enable_l3_cache) {

            *ecx = 0;

        } else {

            *ecx = L3_N_DESCRIPTOR;

        }

        *edx = (L1D_DESCRIPTOR << 16) | \

               (L1I_DESCRIPTOR <<  8) | \

               (L2_DESCRIPTOR);

        break;

    case 4:

        /* cache info: needed for Core compatibility */

        if (cpu->cache_info_passthrough) {

            host_cpuid(index, count, eax, ebx, ecx, edx);

            *eax &= ~0xFC000000;

        } else {

            *eax = 0;

            switch (count) {

            case 0: /* L1 dcache info */

                *eax |= CPUID_4_TYPE_DCACHE | \

                        CPUID_4_LEVEL(1) | \

                        CPUID_4_SELF_INIT_LEVEL;

                *ebx = (L1D_LINE_SIZE - 1) | \

                       ((L1D_PARTITIONS - 1) << 12) | \

                       ((L1D_ASSOCIATIVITY - 1) << 22);

                *ecx = L1D_SETS - 1;

                *edx = CPUID_4_NO_INVD_SHARING;

                break;

            case 1: /* L1 icache info */

                *eax |= CPUID_4_TYPE_ICACHE | \

                        CPUID_4_LEVEL(1) | \

                        CPUID_4_SELF_INIT_LEVEL;

                *ebx = (L1I_LINE_SIZE - 1) | \

                       ((L1I_PARTITIONS - 1) << 12) | \

                       ((L1I_ASSOCIATIVITY - 1) << 22);

                *ecx = L1I_SETS - 1;

                *edx = CPUID_4_NO_INVD_SHARING;

                break;

            case 2: /* L2 cache info */

                *eax |= CPUID_4_TYPE_UNIFIED | \

                        CPUID_4_LEVEL(2) | \

                        CPUID_4_SELF_INIT_LEVEL;

                if (cs->nr_threads > 1) {

                    *eax |= (cs->nr_threads - 1) << 14;

                }

                *ebx = (L2_LINE_SIZE - 1) | \

                       ((L2_PARTITIONS - 1) << 12) | \

                       ((L2_ASSOCIATIVITY - 1) << 22);

                *ecx = L2_SETS - 1;

                *edx = CPUID_4_NO_INVD_SHARING;

                break;

            case 3: /* L3 cache info */

                if (!cpu->enable_l3_cache) {

                    *eax = 0;

                    *ebx = 0;

                    *ecx = 0;

                    *edx = 0;

                    break;

                }

                *eax |= CPUID_4_TYPE_UNIFIED | \

                        CPUID_4_LEVEL(3) | \

                        CPUID_4_SELF_INIT_LEVEL;

                pkg_offset = apicid_pkg_offset(cs->nr_cores, cs->nr_threads);

                *eax |= ((1 << pkg_offset) - 1) << 14;

                *ebx = (L3_N_LINE_SIZE - 1) | \

                       ((L3_N_PARTITIONS - 1) << 12) | \

                       ((L3_N_ASSOCIATIVITY - 1) << 22);

                *ecx = L3_N_SETS - 1;

                *edx = CPUID_4_INCLUSIVE | CPUID_4_COMPLEX_IDX;

                break;

            default: /* end of info */

                *eax = 0;

                *ebx = 0;

                *ecx = 0;

                *edx = 0;

                break;

            }

        }



        /* QEMU gives out its own APIC IDs, never pass down bits 31..26.  */

        if ((*eax & 31) && cs->nr_cores > 1) {

            *eax |= (cs->nr_cores - 1) << 26;

        }

        break;

    case 5:

        /* mwait info: needed for Core compatibility */

        *eax = 0; /* Smallest monitor-line size in bytes */

        *ebx = 0; /* Largest monitor-line size in bytes */

        *ecx = CPUID_MWAIT_EMX | CPUID_MWAIT_IBE;

        *edx = 0;

        break;

    case 6:

        /* Thermal and Power Leaf */

        *eax = env->features[FEAT_6_EAX];

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    case 7:

        /* Structured Extended Feature Flags Enumeration Leaf */

        if (count == 0) {

            *eax = 0; /* Maximum ECX value for sub-leaves */

            *ebx = env->features[FEAT_7_0_EBX]; /* Feature flags */

            *ecx = env->features[FEAT_7_0_ECX]; /* Feature flags */

            if ((*ecx & CPUID_7_0_ECX_PKU) && env->cr[4] & CR4_PKE_MASK) {

                *ecx |= CPUID_7_0_ECX_OSPKE;

            }

            *edx = 0; /* Reserved */

        } else {

            *eax = 0;

            *ebx = 0;

            *ecx = 0;

            *edx = 0;

        }

        break;

    case 9:

        /* Direct Cache Access Information Leaf */

        *eax = 0; /* Bits 0-31 in DCA_CAP MSR */

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    case 0xA:

        /* Architectural Performance Monitoring Leaf */

        if (kvm_enabled() && cpu->enable_pmu) {

            KVMState *s = cs->kvm_state;



            *eax = kvm_arch_get_supported_cpuid(s, 0xA, count, R_EAX);

            *ebx = kvm_arch_get_supported_cpuid(s, 0xA, count, R_EBX);

            *ecx = kvm_arch_get_supported_cpuid(s, 0xA, count, R_ECX);

            *edx = kvm_arch_get_supported_cpuid(s, 0xA, count, R_EDX);

        } else {

            *eax = 0;

            *ebx = 0;

            *ecx = 0;

            *edx = 0;

        }

        break;

    case 0xB:

        /* Extended Topology Enumeration Leaf */

        if (!cpu->enable_cpuid_0xb) {

                *eax = *ebx = *ecx = *edx = 0;

                break;

        }



        *ecx = count & 0xff;

        *edx = cpu->apic_id;



        switch (count) {

        case 0:

            *eax = apicid_core_offset(cs->nr_cores, cs->nr_threads);

            *ebx = cs->nr_threads;

            *ecx |= CPUID_TOPOLOGY_LEVEL_SMT;

            break;

        case 1:

            *eax = apicid_pkg_offset(cs->nr_cores, cs->nr_threads);

            *ebx = cs->nr_cores * cs->nr_threads;

            *ecx |= CPUID_TOPOLOGY_LEVEL_CORE;

            break;

        default:

            *eax = 0;

            *ebx = 0;

            *ecx |= CPUID_TOPOLOGY_LEVEL_INVALID;

        }



        assert(!(*eax & ~0x1f));

        *ebx &= 0xffff; /* The count doesn't need to be reliable. */

        break;

    case 0xD: {

        KVMState *s = cs->kvm_state;

        uint64_t ena_mask;

        int i;



        /* Processor Extended State */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        if (!(env->features[FEAT_1_ECX] & CPUID_EXT_XSAVE)) {

            break;

        }

        if (kvm_enabled()) {

            ena_mask = kvm_arch_get_supported_cpuid(s, 0xd, 0, R_EDX);

            ena_mask <<= 32;

            ena_mask |= kvm_arch_get_supported_cpuid(s, 0xd, 0, R_EAX);

        } else {

            ena_mask = -1;

        }



        if (count == 0) {

            *ecx = 0x240;

            for (i = 2; i < ARRAY_SIZE(x86_ext_save_areas); i++) {

                const ExtSaveArea *esa = &x86_ext_save_areas[i];

                if ((env->features[esa->feature] & esa->bits) == esa->bits

                    && ((ena_mask >> i) & 1) != 0) {

                    if (i < 32) {

                        *eax |= 1u << i;

                    } else {

                        *edx |= 1u << (i - 32);

                    }

                    *ecx = MAX(*ecx, esa->offset + esa->size);

                }

            }

            *eax |= ena_mask & (XSTATE_FP_MASK | XSTATE_SSE_MASK);

            *ebx = *ecx;

        } else if (count == 1) {

            *eax = env->features[FEAT_XSAVE];

        } else if (count < ARRAY_SIZE(x86_ext_save_areas)) {

            const ExtSaveArea *esa = &x86_ext_save_areas[count];

            if ((env->features[esa->feature] & esa->bits) == esa->bits

                && ((ena_mask >> count) & 1) != 0) {

                *eax = esa->size;

                *ebx = esa->offset;

            }

        }

        break;

    }

    case 0x80000000:

        *eax = env->cpuid_xlevel;

        *ebx = env->cpuid_vendor1;

        *edx = env->cpuid_vendor2;

        *ecx = env->cpuid_vendor3;

        break;

    case 0x80000001:

        *eax = env->cpuid_version;

        *ebx = 0;

        *ecx = env->features[FEAT_8000_0001_ECX];

        *edx = env->features[FEAT_8000_0001_EDX];



        /* The Linux kernel checks for the CMPLegacy bit and

         * discards multiple thread information if it is set.

         * So don't set it here for Intel to make Linux guests happy.

         */

        if (cs->nr_cores * cs->nr_threads > 1) {

            if (env->cpuid_vendor1 != CPUID_VENDOR_INTEL_1 ||

                env->cpuid_vendor2 != CPUID_VENDOR_INTEL_2 ||

                env->cpuid_vendor3 != CPUID_VENDOR_INTEL_3) {

                *ecx |= 1 << 1;    /* CmpLegacy bit */

            }

        }

        break;

    case 0x80000002:

    case 0x80000003:

    case 0x80000004:

        *eax = env->cpuid_model[(index - 0x80000002) * 4 + 0];

        *ebx = env->cpuid_model[(index - 0x80000002) * 4 + 1];

        *ecx = env->cpuid_model[(index - 0x80000002) * 4 + 2];

        *edx = env->cpuid_model[(index - 0x80000002) * 4 + 3];

        break;

    case 0x80000005:

        /* cache info (L1 cache) */

        if (cpu->cache_info_passthrough) {

            host_cpuid(index, 0, eax, ebx, ecx, edx);

            break;

        }

        *eax = (L1_DTLB_2M_ASSOC << 24) | (L1_DTLB_2M_ENTRIES << 16) | \

               (L1_ITLB_2M_ASSOC <<  8) | (L1_ITLB_2M_ENTRIES);

        *ebx = (L1_DTLB_4K_ASSOC << 24) | (L1_DTLB_4K_ENTRIES << 16) | \

               (L1_ITLB_4K_ASSOC <<  8) | (L1_ITLB_4K_ENTRIES);

        *ecx = (L1D_SIZE_KB_AMD << 24) | (L1D_ASSOCIATIVITY_AMD << 16) | \

               (L1D_LINES_PER_TAG << 8) | (L1D_LINE_SIZE);

        *edx = (L1I_SIZE_KB_AMD << 24) | (L1I_ASSOCIATIVITY_AMD << 16) | \

               (L1I_LINES_PER_TAG << 8) | (L1I_LINE_SIZE);

        break;

    case 0x80000006:

        /* cache info (L2 cache) */

        if (cpu->cache_info_passthrough) {

            host_cpuid(index, 0, eax, ebx, ecx, edx);

            break;

        }

        *eax = (AMD_ENC_ASSOC(L2_DTLB_2M_ASSOC) << 28) | \

               (L2_DTLB_2M_ENTRIES << 16) | \

               (AMD_ENC_ASSOC(L2_ITLB_2M_ASSOC) << 12) | \

               (L2_ITLB_2M_ENTRIES);

        *ebx = (AMD_ENC_ASSOC(L2_DTLB_4K_ASSOC) << 28) | \

               (L2_DTLB_4K_ENTRIES << 16) | \

               (AMD_ENC_ASSOC(L2_ITLB_4K_ASSOC) << 12) | \

               (L2_ITLB_4K_ENTRIES);

        *ecx = (L2_SIZE_KB_AMD << 16) | \

               (AMD_ENC_ASSOC(L2_ASSOCIATIVITY) << 12) | \

               (L2_LINES_PER_TAG << 8) | (L2_LINE_SIZE);

        if (!cpu->enable_l3_cache) {

            *edx = ((L3_SIZE_KB / 512) << 18) | \

                   (AMD_ENC_ASSOC(L3_ASSOCIATIVITY) << 12) | \

                   (L3_LINES_PER_TAG << 8) | (L3_LINE_SIZE);

        } else {

            *edx = ((L3_N_SIZE_KB_AMD / 512) << 18) | \

                   (AMD_ENC_ASSOC(L3_N_ASSOCIATIVITY) << 12) | \

                   (L3_N_LINES_PER_TAG << 8) | (L3_N_LINE_SIZE);

        }

        break;

    case 0x80000007:

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = env->features[FEAT_8000_0007_EDX];

        break;

    case 0x80000008:

        /* virtual & phys address size in low 2 bytes. */

        if (env->features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM) {

            /* 64 bit processor, 48 bits virtual, configurable

             * physical bits.

             */

            *eax = 0x00003000 + cpu->phys_bits;

        } else {

            *eax = cpu->phys_bits;

        }

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        if (cs->nr_cores * cs->nr_threads > 1) {

            *ecx |= (cs->nr_cores * cs->nr_threads) - 1;

        }

        break;

    case 0x8000000A:

        if (env->features[FEAT_8000_0001_ECX] & CPUID_EXT3_SVM) {

            *eax = 0x00000001; /* SVM Revision */

            *ebx = 0x00000010; /* nr of ASIDs */

            *ecx = 0;

            *edx = env->features[FEAT_SVM]; /* optional features */

        } else {

            *eax = 0;

            *ebx = 0;

            *ecx = 0;

            *edx = 0;

        }

        break;

    case 0xC0000000:

        *eax = env->cpuid_xlevel2;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    case 0xC0000001:

        /* Support for VIA CPU's CPUID instruction */

        *eax = env->cpuid_version;

        *ebx = 0;

        *ecx = 0;

        *edx = env->features[FEAT_C000_0001_EDX];

        break;

    case 0xC0000002:

    case 0xC0000003:

    case 0xC0000004:

        /* Reserved for the future, and now filled with zero */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    default:

        /* reserved values: zero */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    }

}
