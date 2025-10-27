static void kqemu_update_cpuid(CPUState *env)

{

    int critical_features_mask, features, ext_features, ext_features_mask;

    uint32_t eax, ebx, ecx, edx;



    /* the following features are kept identical on the host and

       target cpus because they are important for user code. Strictly

       speaking, only SSE really matters because the OS must support

       it if the user code uses it. */

    critical_features_mask =

        CPUID_CMOV | CPUID_CX8 |

        CPUID_FXSR | CPUID_MMX | CPUID_SSE |

        CPUID_SSE2 | CPUID_SEP;

    ext_features_mask = CPUID_EXT_SSE3 | CPUID_EXT_MONITOR;

    if (!is_cpuid_supported()) {

        features = 0;

        ext_features = 0;

    } else {

        cpuid(1, eax, ebx, ecx, edx);

        features = edx;

        ext_features = ecx;

    }

#ifdef __x86_64__

    /* NOTE: on x86_64 CPUs, SYSENTER is not supported in

       compatibility mode, so in order to have the best performances

       it is better not to use it */

    features &= ~CPUID_SEP;

#endif

    env->cpuid_features = (env->cpuid_features & ~critical_features_mask) |

        (features & critical_features_mask);

    env->cpuid_ext_features = (env->cpuid_ext_features & ~ext_features_mask) |

        (ext_features & ext_features_mask);

    /* XXX: we could update more of the target CPUID state so that the

       non accelerated code sees exactly the same CPU features as the

       accelerated code */

}
