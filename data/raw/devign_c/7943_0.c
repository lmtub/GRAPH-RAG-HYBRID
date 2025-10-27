static void init_proc_970MP (CPUPPCState *env)

{

    gen_spr_ne_601(env);

    gen_spr_7xx(env);

    /* Time base */

    gen_tbl(env);

    /* Hardware implementation registers */

    /* XXX : not implemented */

    spr_register(env, SPR_HID0, "HID0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_clear,

                 0x60000000);

    /* XXX : not implemented */

    spr_register(env, SPR_HID1, "HID1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_970_HID5, "HID5",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 POWERPC970_HID5_INIT);

    /* XXX : not implemented */

    /* Memory management */

    /* XXX: not correct */

    gen_low_BATs(env);

    spr_register(env, SPR_HIOR, "SPR_HIOR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_hior, &spr_write_hior,

                 0x00000000);

    /* Logical partitionning */

    spr_register_kvm(env, SPR_LPCR, "LPCR",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     KVM_REG_PPC_LPCR, 0x00000000);

#if !defined(CONFIG_USER_ONLY)

    env->slb_nr = 32;

#endif

    init_excp_970(env);

    env->dcache_line_size = 128;

    env->icache_line_size = 128;

    /* Allocate hardware IRQ controller */

    ppc970_irq_init(env);

    /* Can't find information on what this should be on reset.  This

     * value is the one used by 74xx processors. */

    vscr_init(env, 0x00010000);

}
