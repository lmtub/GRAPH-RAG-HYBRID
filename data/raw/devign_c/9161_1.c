void cpu_x86_cpuid(CPUX86State *env, uint32_t index, uint32_t count,

                   uint32_t *eax, uint32_t *ebx,

                   uint32_t *ecx, uint32_t *edx)

{

    X86CPU *cpu = x86_env_get_cpu(env);

    CPUState *cs = CPU(cpu);



    /* test if maximum index reached */

    if (index & 0x80000000) {

        if (index > env->cpuid_xlevel) {

            if (env->cpuid_xlevel2 > 0) {

                /* Handle the Centaur's CPUID instruction. */

                if (index > env->cpuid_xlevel2) {

                    index = env->cpuid_xlevel2;

                } else if (index < 0xC0000000) {

                    index = env->cpuid_xlevel;


            } else {

                /* Intel documentation states that invalid EAX input will

                 * return the same information as EAX=cpuid_level

                 * (Intel SDM Vol. 2A - Instruction Set Reference - CPUID)

                 */

                index =  env->cpuid_level;



    } else {

        if (index > env->cpuid_level)

            index = env->cpuid_level;




    switch(index) {

    case 0:

        *eax = env->cpuid_level;

        get_cpuid_vendor(env, ebx, ecx, edx);


    case 1:

        *eax = env->cpuid_version;

        *ebx = (env->cpuid_apic_id << 24) | 8 << 8; /* CLFLUSH size in quad words, Linux wants it. */

        *ecx = env->features[FEAT_1_ECX];

        *edx = env->features[FEAT_1_EDX];

        if (cs->nr_cores * cs->nr_threads > 1) {

            *ebx |= (cs->nr_cores * cs->nr_threads) << 16;

            *edx |= 1 << 28;    /* HTT bit */



    case 2:

        /* cache info: needed for Pentium Pro compatibility */


            host_cpuid(index, 0, eax, ebx, ecx, edx);



        *eax = 1; /* Number of CPUID[EAX=2] calls required */

        *ebx = 0;

        *ecx = 0;

        *edx = (L1D_DESCRIPTOR << 16) | \

               (L1I_DESCRIPTOR <<  8) | \

               (L2_DESCRIPTOR);


    case 4:

        /* cache info: needed for Core compatibility */





        if (cs->nr_cores > 1) {

            *eax = (cs->nr_cores - 1) << 26;

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


            case 1: /* L1 icache info */

                *eax |= CPUID_4_TYPE_ICACHE | \

                        CPUID_4_LEVEL(1) | \

                        CPUID_4_SELF_INIT_LEVEL;

                *ebx = (L1I_LINE_SIZE - 1) | \

                       ((L1I_PARTITIONS - 1) << 12) | \

                       ((L1I_ASSOCIATIVITY - 1) << 22);

                *ecx = L1I_SETS - 1;

                *edx = CPUID_4_NO_INVD_SHARING;


            case 2: /* L2 cache info */

                *eax |= CPUID_4_TYPE_UNIFIED | \

                        CPUID_4_LEVEL(2) | \

                        CPUID_4_SELF_INIT_LEVEL;

                if (cs->nr_threads > 1) {

                    *eax |= (cs->nr_threads - 1) << 14;


                *ebx = (L2_LINE_SIZE - 1) | \

                       ((L2_PARTITIONS - 1) << 12) | \

                       ((L2_ASSOCIATIVITY - 1) << 22);

                *ecx = L2_SETS - 1;

                *edx = CPUID_4_NO_INVD_SHARING;


            default: /* end of info */

                *eax = 0;

                *ebx = 0;

                *ecx = 0;

                *edx = 0;




    case 5:

        /* mwait info: needed for Core compatibility */

        *eax = 0; /* Smallest monitor-line size in bytes */

        *ebx = 0; /* Largest monitor-line size in bytes */

        *ecx = CPUID_MWAIT_EMX | CPUID_MWAIT_IBE;

        *edx = 0;


    case 6:

        /* Thermal and Power Leaf */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;


    case 7:

        /* Structured Extended Feature Flags Enumeration Leaf */

        if (count == 0) {

            *eax = 0; /* Maximum ECX value for sub-leaves */

            *ebx = env->features[FEAT_7_0_EBX]; /* Feature flags */

            *ecx = 0; /* Reserved */

            *edx = 0; /* Reserved */

        } else {

            *eax = 0;

            *ebx = 0;

            *ecx = 0;

            *edx = 0;



    case 9:

        /* Direct Cache Access Information Leaf */

        *eax = 0; /* Bits 0-31 in DCA_CAP MSR */

        *ebx = 0;

        *ecx = 0;

        *edx = 0;


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



    case 0xD:

        /* Processor Extended State */

        if (!(env->features[FEAT_1_ECX] & CPUID_EXT_XSAVE)) {

            *eax = 0;

            *ebx = 0;

            *ecx = 0;

            *edx = 0;



        if (kvm_enabled()) {

            KVMState *s = cs->kvm_state;



            *eax = kvm_arch_get_supported_cpuid(s, 0xd, count, R_EAX);

            *ebx = kvm_arch_get_supported_cpuid(s, 0xd, count, R_EBX);

            *ecx = kvm_arch_get_supported_cpuid(s, 0xd, count, R_ECX);

            *edx = kvm_arch_get_supported_cpuid(s, 0xd, count, R_EDX);

        } else {

            *eax = 0;

            *ebx = 0;

            *ecx = 0;

            *edx = 0;



    case 0x80000000:

        *eax = env->cpuid_xlevel;

        *ebx = env->cpuid_vendor1;

        *edx = env->cpuid_vendor2;

        *ecx = env->cpuid_vendor3;


    case 0x80000001:

        *eax = env->cpuid_version;

        *ebx = 0;

        *ecx = env->features[FEAT_8000_0001_ECX];

        *edx = env->features[FEAT_8000_0001_EDX];



        /* The Linux kernel checks for the CMPLegacy bit and

         * discards multiple thread information if it is set.

         * So dont set it here for Intel to make Linux guests happy.

         */

        if (cs->nr_cores * cs->nr_threads > 1) {

            uint32_t tebx, tecx, tedx;

            get_cpuid_vendor(env, &tebx, &tecx, &tedx);

            if (tebx != CPUID_VENDOR_INTEL_1 ||

                tedx != CPUID_VENDOR_INTEL_2 ||

                tecx != CPUID_VENDOR_INTEL_3) {

                *ecx |= 1 << 1;    /* CmpLegacy bit */




    case 0x80000002:

    case 0x80000003:

    case 0x80000004:

        *eax = env->cpuid_model[(index - 0x80000002) * 4 + 0];

        *ebx = env->cpuid_model[(index - 0x80000002) * 4 + 1];

        *ecx = env->cpuid_model[(index - 0x80000002) * 4 + 2];

        *edx = env->cpuid_model[(index - 0x80000002) * 4 + 3];


    case 0x80000005:

        /* cache info (L1 cache) */


            host_cpuid(index, 0, eax, ebx, ecx, edx);



        *eax = (L1_DTLB_2M_ASSOC << 24) | (L1_DTLB_2M_ENTRIES << 16) | \

               (L1_ITLB_2M_ASSOC <<  8) | (L1_ITLB_2M_ENTRIES);

        *ebx = (L1_DTLB_4K_ASSOC << 24) | (L1_DTLB_4K_ENTRIES << 16) | \

               (L1_ITLB_4K_ASSOC <<  8) | (L1_ITLB_4K_ENTRIES);

        *ecx = (L1D_SIZE_KB_AMD << 24) | (L1D_ASSOCIATIVITY_AMD << 16) | \

               (L1D_LINES_PER_TAG << 8) | (L1D_LINE_SIZE);

        *edx = (L1I_SIZE_KB_AMD << 24) | (L1I_ASSOCIATIVITY_AMD << 16) | \

               (L1I_LINES_PER_TAG << 8) | (L1I_LINE_SIZE);


    case 0x80000006:

        /* cache info (L2 cache) */


            host_cpuid(index, 0, eax, ebx, ecx, edx);



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

        *edx = ((L3_SIZE_KB/512) << 18) | \

               (AMD_ENC_ASSOC(L3_ASSOCIATIVITY) << 12) | \

               (L3_LINES_PER_TAG << 8) | (L3_LINE_SIZE);


    case 0x80000008:

        /* virtual & phys address size in low 2 bytes. */

/* XXX: This value must match the one used in the MMU code. */

        if (env->features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM) {

            /* 64 bit processor */

/* XXX: The physical address space is limited to 42 bits in exec.c. */

            *eax = 0x00003028; /* 48 bits virtual, 40 bits physical */

        } else {

            if (env->features[FEAT_1_EDX] & CPUID_PSE36) {

                *eax = 0x00000024; /* 36 bits physical */

            } else {

                *eax = 0x00000020; /* 32 bits physical */



        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        if (cs->nr_cores * cs->nr_threads > 1) {

            *ecx |= (cs->nr_cores * cs->nr_threads) - 1;



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



    case 0xC0000000:

        *eax = env->cpuid_xlevel2;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;


    case 0xC0000001:

        /* Support for VIA CPU's CPUID instruction */

        *eax = env->cpuid_version;

        *ebx = 0;

        *ecx = 0;

        *edx = env->features[FEAT_C000_0001_EDX];


    case 0xC0000002:

    case 0xC0000003:

    case 0xC0000004:

        /* Reserved for the future, and now filled with zero */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;


    default:

        /* reserved values: zero */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;


