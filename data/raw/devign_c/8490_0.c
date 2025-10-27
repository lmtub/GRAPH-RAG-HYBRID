static int read_bfraction(VC1Context *v, GetBitContext* gb) {

    v->bfraction_lut_index = get_vlc2(gb, ff_vc1_bfraction_vlc.table, VC1_BFRACTION_VLC_BITS, 1);

    v->bfraction           = ff_vc1_bfraction_lut[v->bfraction_lut_index];

    return 0;

}
