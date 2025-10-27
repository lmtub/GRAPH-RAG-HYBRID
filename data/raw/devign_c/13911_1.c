static inline void rgtc_block_internal(uint8_t *dst, ptrdiff_t stride,

                                       const uint8_t *block,

                                       const int *color_tab)

{

    uint8_t indices[16];

    int x, y;



    decompress_indices(indices, block + 2);



    /* Only one or two channels are stored at most, since it only used to

     * compress specular (black and white) or normal (red and green) maps.

     * Although the standard says to zero out unused components, many

     * implementations fill all of them with the same value. */

    for (y = 0; y < 4; y++) {

        for (x = 0; x < 4; x++) {

            int i = indices[x + y * 4];

            /* Interval expansion from [-1 1] or [0 1] to [0 255]. */

            int c = color_tab[i];

            uint32_t pixel = RGBA(c, c, c, 255);

            AV_WL32(dst + x * 4 + y * stride, pixel);

        }

    }

}
