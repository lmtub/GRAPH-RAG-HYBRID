static void init_proc_power5plus(CPUPPCState *env)

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

    spr_register(env, SPR_750FX_HID2, "HID2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_970_HID5, "HID5",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 POWERPC970_HID5_INIT);

    /* XXX : not implemented */

    spr_register(env, SPR_L2CR, "L2CR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, NULL,

                 0x00000000);

    /* Memory management */

    /* XXX: not correct */

    gen_low_BATs(env);

    /* XXX : not implemented */

    spr_register(env, SPR_MMUCFG, "MMUCFG",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, SPR_NOACCESS,

                 0x00000000); /* TOFIX */

    /* XXX : not implemented */

    spr_register(env, SPR_MMUCSR0, "MMUCSR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000); /* TOFIX */

    spr_register(env, SPR_HIOR, "SPR_HIOR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_hior, &spr_write_hior,

                 0x00000000);

    spr_register(env, SPR_CTRL, "SPR_CTRL",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_UCTRL, "SPR_UCTRL",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_VRSAVE, "SPR_VRSAVE",

                 &spr_read_generic, &spr_write_generic,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

#if !defined(CONFIG_USER_ONLY)

    env->slb_nr = 64;

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
