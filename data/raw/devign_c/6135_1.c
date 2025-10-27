static void init_proc_750cx (CPUPPCState *env)

{

    gen_spr_ne_601(env);

    gen_spr_7xx(env);

    /* XXX : not implemented */

    spr_register(env, SPR_L2CR, "L2CR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, NULL,

                 0x00000000);

    /* Time base */

    gen_tbl(env);

    /* Thermal management */

    gen_spr_thrm(env);

    /* This register is not implemented but is present for compatibility */

    spr_register(env, SPR_SDA, "SDA",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Hardware implementation registers */

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

    /* Memory management */

    gen_low_BATs(env);

    /* PowerPC 750cx has 8 DBATs and 8 IBATs */

    gen_high_BATs(env);

    init_excp_750cx(env);

    env->dcache_line_size = 32;

    env->icache_line_size = 32;

    /* Allocate hardware IRQ controller */

    ppc6xx_irq_init(env);

}
