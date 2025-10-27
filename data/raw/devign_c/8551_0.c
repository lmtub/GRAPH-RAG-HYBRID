static int get_scale_idx(GetBitContext *gb, int ref)

{

    int t = get_vlc2(gb, dscf_vlc.table, MPC7_DSCF_BITS, 1) - 7;

    if (t == 8)

        return get_bits(gb, 6);

    return ref + t;

}
