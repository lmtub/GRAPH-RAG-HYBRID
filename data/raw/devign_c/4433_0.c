void HELPER(pka)(CPUS390XState *env, uint64_t dest, uint64_t src,

                 uint32_t srclen)

{

    uintptr_t ra = GETPC();

    int i;

    /* The destination operand is always 16 bytes long.  */

    const int destlen = 16;



    /* The operands are processed from right to left.  */

    src += srclen - 1;

    dest += destlen - 1;



    for (i = 0; i < destlen; i++) {

        uint8_t b = 0;



        /* Start with a positive sign */

        if (i == 0) {

            b = 0xc;

        } else if (srclen > 1) {

            b = cpu_ldub_data_ra(env, src, ra) & 0x0f;

            src--;

            srclen--;

        }



        if (srclen > 1) {

            b |= cpu_ldub_data_ra(env, src, ra) << 4;

            src--;

            srclen--;

        }



        cpu_stb_data_ra(env, dest, b, ra);

        dest--;

    }

}
