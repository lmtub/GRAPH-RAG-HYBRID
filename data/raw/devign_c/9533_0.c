static inline void h264_deblock_q1(register vector unsigned char p0,

                                                   register vector unsigned char p1,

                                                   register vector unsigned char p2,

                                                   register vector unsigned char q0,

                                                   register vector unsigned char tc0) {



    register vector unsigned char average = vec_avg(p0, q0);

    register vector unsigned char temp;

    register vector unsigned char uncliped;

    register vector unsigned char ones;

    register vector unsigned char max;

    register vector unsigned char min;



    temp = vec_xor(average, p2);

    average = vec_avg(average, p2);     /*avg(p2, avg(p0, q0)) */

    ones = vec_splat_u8(1);

    temp = vec_and(temp, ones);         /*(p2^avg(p0, q0)) & 1 */

    uncliped = vec_subs(average, temp); /*(p2+((p0+q0+1)>>1))>>1 */

    max = vec_adds(p1, tc0);

    min = vec_subs(p1, tc0);

    p1 = vec_max(min, uncliped);

    p1 = vec_min(max, p1);

}
