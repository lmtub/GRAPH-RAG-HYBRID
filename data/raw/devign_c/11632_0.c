static void vertical_filter(unsigned char *first_pixel, int stride,

    int *bounding_values)

{

    int i;

    int filter_value;



    for (i = 0; i < 8; i++, first_pixel++) {

        filter_value = 

            (first_pixel[-(2 * stride)] * 1) - 

            (first_pixel[-(1 * stride)] * 3) +

            (first_pixel[ (0         )] * 3) -

            (first_pixel[ (1 * stride)] * 1);

        filter_value = bounding_values[(filter_value + 4) >> 3];

        first_pixel[-(1 * stride)] = SATURATE_U8(first_pixel[-(1 * stride)] + filter_value);

        first_pixel[0] = SATURATE_U8(first_pixel[0] - filter_value);

    }

}
