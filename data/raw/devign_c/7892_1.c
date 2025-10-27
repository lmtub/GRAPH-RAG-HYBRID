static int minimum_frame_bits(VC2EncContext *s)

{

    int slice_x, slice_y, bits = 0;

    s->size_scaler = 64;

    for (slice_y = 0; slice_y < s->num_y; slice_y++) {

        for (slice_x = 0; slice_x < s->num_x; slice_x++) {

            bits += count_hq_slice(s, NULL, slice_x, slice_y, s->q_ceil);

        }

    }

    return bits;

}
