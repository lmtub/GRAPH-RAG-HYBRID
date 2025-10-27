static void init_proc_e500 (CPUPPCState *env, int version)

{

    uint32_t tlbncfg[2];

    uint64_t ivor_mask = 0x0000000F0000FFFFULL;

    uint32_t l1cfg0 = 0x3800  /* 8 ways */

                    | 0x0020; /* 32 kb */

#if !defined(CONFIG_USER_ONLY)

    int i;

#endif



    /* Time base */

    gen_tbl(env);

    /*

     * XXX The e500 doesn't implement IVOR7 and IVOR9, but doesn't

     *     complain when accessing them.

     * gen_spr_BookE(env, 0x0000000F0000FD7FULL);

     */

    if (version == fsl_e500mc) {

        ivor_mask = 0x000003FE0000FFFFULL;

    }

    gen_spr_BookE(env, ivor_mask);

    /* Processor identification */

    spr_register(env, SPR_BOOKE_PIR, "PIR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_pir,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_BOOKE_SPEFSCR, "SPEFSCR",

                 &spr_read_spefscr, &spr_write_spefscr,

                 &spr_read_spefscr, &spr_write_spefscr,

                 0x00000000);

    /* Memory management */

#if defined(CONFIG_USER_ONLY)

    env->dcache_line_size = 32;

    env->icache_line_size = 32;

#else /* !defined(CONFIG_USER_ONLY) */

    env->nb_pids = 3;

    env->nb_ways = 2;

    env->id_tlbs = 0;

    switch (version) {

    case fsl_e500v1:

        /* e500v1 */

        tlbncfg[0] = gen_tlbncfg(2, 1, 1, 0, 256);

        tlbncfg[1] = gen_tlbncfg(16, 1, 9, TLBnCFG_AVAIL | TLBnCFG_IPROT, 16);

        env->dcache_line_size = 32;

        env->icache_line_size = 32;

        break;

    case fsl_e500v2:

        /* e500v2 */

        tlbncfg[0] = gen_tlbncfg(4, 1, 1, 0, 512);

        tlbncfg[1] = gen_tlbncfg(16, 1, 12, TLBnCFG_AVAIL | TLBnCFG_IPROT, 16);

        env->dcache_line_size = 32;

        env->icache_line_size = 32;

        break;

    case fsl_e500mc:

        /* e500mc */

        tlbncfg[0] = gen_tlbncfg(4, 1, 1, 0, 512);

        tlbncfg[1] = gen_tlbncfg(64, 1, 12, TLBnCFG_AVAIL | TLBnCFG_IPROT, 64);

        env->dcache_line_size = 64;

        env->icache_line_size = 64;

        l1cfg0 |= 0x1000000; /* 64 byte cache block size */

        break;

    default:

        cpu_abort(env, "Unknown CPU: " TARGET_FMT_lx "\n", env->spr[SPR_PVR]);

    }

#endif

    gen_spr_BookE206(env, 0x000000DF, tlbncfg);

    /* XXX : not implemented */

    spr_register(env, SPR_HID0, "HID0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_HID1, "HID1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_BBEAR, "BBEAR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_BBTAR, "BBTAR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_MCAR, "MCAR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_BOOKE_MCSR, "MCSR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_NPIDR, "NPIDR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_BUCSR, "BUCSR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_L1CFG0, "L1CFG0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 l1cfg0);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_L1CSR0, "L1CSR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_e500_l1csr0,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_Exxx_L1CSR1, "L1CSR1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_BOOKE_MCSRR0, "MCSRR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_BOOKE_MCSRR1, "MCSRR1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_MMUCSR0, "MMUCSR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_booke206_mmucsr0,

                 0x00000000);



#if !defined(CONFIG_USER_ONLY)

    env->nb_tlb = 0;

    env->tlb_type = TLB_MAS;

    for (i = 0; i < BOOKE206_MAX_TLBN; i++) {

        env->nb_tlb += booke206_tlb_size(env, i);

    }

#endif



    init_excp_e200(env);

    /* Allocate hardware IRQ controller */

    ppce500_irq_init(env);

}
