static int mp_get_vlc(MotionPixelsContext *mp, GetBitContext *gb)

{

    int i;



    i = (mp->codes_count == 1) ? 0 : get_vlc2(gb, mp->vlc.table, mp->max_codes_bits, 1);


    return mp->codes[i].delta;

}