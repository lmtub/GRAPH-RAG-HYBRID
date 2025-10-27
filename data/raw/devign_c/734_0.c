void ff_vp3_v_loop_filter_c(uint8_t *first_pixel, int stride, int *bounding_values)

{

    unsigned char *end;

    int filter_value;

    const int nstride= -stride;



    for (end= first_pixel + 8; first_pixel < end; first_pixel++) {

        filter_value =

            (first_pixel[2 * nstride] - first_pixel[ stride])

         +3*(first_pixel[0          ] - first_pixel[nstride]);

        filter_value = bounding_values[(filter_value + 4) >> 3];

        first_pixel[nstride] = av_clip_uint8(first_pixel[nstride] + filter_value);

        first_pixel[0] = av_clip_uint8(first_pixel[0] - filter_value);

    }

}
