void ff_ivi_inverse_haar_8x8(const int32_t *in, int16_t *out, ptrdiff_t pitch,

                             const uint8_t *flags)

{

    int     i, shift, sp1, sp2, sp3, sp4;

    const int32_t *src;

    int32_t *dst;

    int     tmp[64];

    int     t0, t1, t2, t3, t4, t5, t6, t7, t8;



    /* apply the InvHaar8 to all columns */

#define COMPENSATE(x) (x)

    src = in;

    dst = tmp;

    for (i = 0; i < 8; i++) {

        if (flags[i]) {

            /* pre-scaling */

            shift = !(i & 4);

            sp1 = src[ 0] << shift;

            sp2 = src[ 8] << shift;

            sp3 = src[16] << shift;

            sp4 = src[24] << shift;

            INV_HAAR8(    sp1,     sp2,     sp3,     sp4,

                      src[32], src[40], src[48], src[56],

                      dst[ 0], dst[ 8], dst[16], dst[24],

                      dst[32], dst[40], dst[48], dst[56],

                      t0, t1, t2, t3, t4, t5, t6, t7, t8);

        } else

            dst[ 0] = dst[ 8] = dst[16] = dst[24] =

            dst[32] = dst[40] = dst[48] = dst[56] = 0;



        src++;

        dst++;

    }

#undef  COMPENSATE



    /* apply the InvHaar8 to all rows */

#define COMPENSATE(x) (x)

    src = tmp;

    for (i = 0; i < 8; i++) {

        if (   !src[0] && !src[1] && !src[2] && !src[3]

            && !src[4] && !src[5] && !src[6] && !src[7]) {

            memset(out, 0, 8 * sizeof(out[0]));

        } else {

            INV_HAAR8(src[0], src[1], src[2], src[3],

                      src[4], src[5], src[6], src[7],

                      out[0], out[1], out[2], out[3],

                      out[4], out[5], out[6], out[7],

                      t0, t1, t2, t3, t4, t5, t6, t7, t8);

        }

        src += 8;

        out += pitch;

    }

#undef  COMPENSATE

}
