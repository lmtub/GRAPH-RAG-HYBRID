uint64_t HELPER(diag)(CPUS390XState *env, uint32_t num, uint64_t mem,

                      uint64_t code)

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

        r = 0;

        break;

    default:

        r = -1;

        break;

    }



    if (r) {

        program_interrupt(env, PGM_OPERATION, ILEN_LATER_INC);

    }



    return r;

}
