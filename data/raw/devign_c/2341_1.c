uint64_t helper_cmpbge(uint64_t op1, uint64_t op2)

{

#if defined(__SSE2__)

    uint64_t r;



    /* The cmpbge instruction is heavily used in the implementation of

       every string function on Alpha.  We can do much better than either

       the default loop below, or even an unrolled version by using the

       native vector support.  */

    {

        typedef uint64_t Q __attribute__((vector_size(16)));

        typedef uint8_t B __attribute__((vector_size(16)));



        Q q1 = (Q){ op1, 0 };

        Q q2 = (Q){ op2, 0 };



        q1 = (Q)((B)q1 >= (B)q2);



        r = q1[0];

    }



    /* Select only one bit from each byte.  */

    r &= 0x0101010101010101;



    /* Collect the bits into the bottom byte.  */

    /* .......A.......B.......C.......D.......E.......F.......G.......H */

    r |= r >> (8 - 1);



    /* .......A......AB......BC......CD......DE......EF......FG......GH */

    r |= r >> (16 - 2);



    /* .......A......AB.....ABC....ABCD....BCDE....CDEF....DEFG....EFGH */

    r |= r >> (32 - 4);



    /* .......A......AB.....ABC....ABCD...ABCDE..ABCDEF.ABCDEFGABCDEFGH */

    /* Return only the low 8 bits.  */

    return r & 0xff;

#else

    uint8_t opa, opb, res;

    int i;



    res = 0;

    for (i = 0; i < 8; i++) {

        opa = op1 >> (i * 8);

        opb = op2 >> (i * 8);

        if (opa >= opb) {

            res |= 1 << i;

        }

    }

    return res;

#endif

}
