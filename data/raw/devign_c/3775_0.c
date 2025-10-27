void helper_cpuid(void)

{

    if (EAX == 0) {

        EAX = 1; /* max EAX index supported */

        EBX = 0x756e6547;

        ECX = 0x6c65746e;

        EDX = 0x49656e69;

    } else {

        /* EAX = 1 info */

        EAX = 0x52b;

        EBX = 0;

        ECX = 0;

        EDX = CPUID_FP87 | CPUID_VME | CPUID_DE | CPUID_PSE |

            CPUID_TSC | CPUID_MSR | CPUID_MCE |

            CPUID_CX8;

    }

}
