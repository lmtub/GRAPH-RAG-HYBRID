static void query_facilities(void)

{

    struct sigaction sa_old, sa_new;

    register int r0 __asm__("0");

    register void *r1 __asm__("1");

    int fail;



    memset(&sa_new, 0, sizeof(sa_new));

    sa_new.sa_handler = sigill_handler;

    sigaction(SIGILL, &sa_new, &sa_old);



    /* First, try STORE FACILITY LIST EXTENDED.  If this is present, then

       we need not do any more probing.  Unfortunately, this itself is an

       extension and the original STORE FACILITY LIST instruction is

       kernel-only, storing its results at absolute address 200.  */

    /* stfle 0(%r1) */

    r1 = &facilities;

    asm volatile(".word 0xb2b0,0x1000"

                 : "=r"(r0) : "0"(0), "r"(r1) : "memory", "cc");



    if (got_sigill) {

        /* STORE FACILITY EXTENDED is not available.  Probe for one of each

           kind of instruction that we're interested in.  */

        /* ??? Possibly some of these are in practice never present unless

           the store-facility-extended facility is also present.  But since

           that isn't documented it's just better to probe for each.  */



        /* Test for z/Architecture.  Required even in 31-bit mode.  */

        got_sigill = 0;

        /* agr %r0,%r0 */

        asm volatile(".word 0xb908,0x0000" : "=r"(r0) : : "cc");

        if (!got_sigill) {

            facilities |= FACILITY_ZARCH_ACTIVE;

        }



        /* Test for long displacement.  */

        got_sigill = 0;

        /* ly %r0,0(%r1) */

        r1 = &facilities;

        asm volatile(".word 0xe300,0x1000,0x0058"

                     : "=r"(r0) : "r"(r1) : "cc");

        if (!got_sigill) {

            facilities |= FACILITY_LONG_DISP;

        }



        /* Test for extended immediates.  */

        got_sigill = 0;

        /* afi %r0,0 */

        asm volatile(".word 0xc209,0x0000,0x0000" : : : "cc");

        if (!got_sigill) {

            facilities |= FACILITY_EXT_IMM;

        }



        /* Test for general-instructions-extension.  */

        got_sigill = 0;

        /* msfi %r0,1 */

        asm volatile(".word 0xc201,0x0000,0x0001");

        if (!got_sigill) {

            facilities |= FACILITY_GEN_INST_EXT;

        }

    }



    sigaction(SIGILL, &sa_old, NULL);



    /* The translator currently uses these extensions unconditionally.

       Pruning this back to the base ESA/390 architecture doesn't seem

       worthwhile, since even the KVM target requires z/Arch.  */

    fail = 0;

    if ((facilities & FACILITY_ZARCH_ACTIVE) == 0) {

        fprintf(stderr, "TCG: z/Arch facility is required.\n");

        fprintf(stderr, "TCG: Boot with a 64-bit enabled kernel.\n");

        fail = 1;

    }

    if ((facilities & FACILITY_LONG_DISP) == 0) {

        fprintf(stderr, "TCG: long-displacement facility is required.\n");

        fail = 1;

    }



    /* So far there's just enough support for 31-bit mode to let the

       compile succeed.  This is good enough to run QEMU with KVM.  */

    if (sizeof(void *) != 8) {

        fprintf(stderr, "TCG: 31-bit mode is not supported.\n");

        fail = 1;

    }



    if (fail) {

        exit(-1);

    }

}
