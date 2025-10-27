static void horizontal_filter(unsigned char *first_pixel, int stride,

    int *bounding_values)

{

    unsigned char *end;

    int filter_value;



    for (end= first_pixel + 8*stride; first_pixel < end; first_pixel += stride) {

        filter_value =

            (first_pixel[-2] - first_pixel[ 1])

         +3*(first_pixel[ 0] - first_pixel[-1]);

        filter_value = bounding_values[(filter_value + 4) >> 3];

        first_pixel[-1] = clip_uint8(first_pixel[-1] + filter_value);

        first_pixel[ 0] = clip_uint8(first_pixel[ 0] - filter_value);

    }

}
