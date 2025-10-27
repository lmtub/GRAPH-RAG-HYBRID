void HELPER(diag)(CPUS390XState *env, uint32_t r1, uint32_t r3, uint32_t num)

{

    uint64_t r;



    switch (num) {

    case 0x500:

        /* KVM hypercall */


        r = s390_virtio_hypercall(env);


        break;

    case 0x44:

        /* yield */

        r = 0;

        break;

    case 0x308:

        /* ipl */


        handle_diag_308(env, r1, r3);


        r = 0;

        break;

    case 0x288:

        /* time bomb (watchdog) */

        r = handle_diag_288(env, r1, r3);

        break;

    default:

        r = -1;

        break;

    }



    if (r) {

        program_interrupt(env, PGM_SPECIFICATION, ILEN_AUTO);

    }

}