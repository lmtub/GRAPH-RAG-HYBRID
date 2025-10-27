static void init_quantized_coeffs_elem0 (int8_t *quantized_coeffs, GetBitContext *gb, int length)

{

    int i, k, run, level, diff;



    if (BITS_LEFT(length,gb) < 16)

        return;

    level = qdm2_get_vlc(gb, &vlc_tab_level, 0, 2);



    quantized_coeffs[0] = level;



    for (i = 0; i < 7; ) {

        if (BITS_LEFT(length,gb) < 16)

            break;

        run = qdm2_get_vlc(gb, &vlc_tab_run, 0, 1) + 1;



        if (BITS_LEFT(length,gb) < 16)

            break;

        diff = qdm2_get_se_vlc(&vlc_tab_diff, gb, 2);



        for (k = 1; k <= run; k++)

            quantized_coeffs[i + k] = (level + ((k * diff) / run));



        level += diff;

        i += run;

    }

}
