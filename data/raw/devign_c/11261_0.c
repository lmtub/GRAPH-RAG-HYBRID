static always_inline int dv_rl2vlc(int run, int l, uint32_t* vlc)

{

    *vlc = dv_vlc_map[run][((uint16_t)l)&0x1ff].vlc;

    return dv_vlc_map[run][((uint16_t)l)&0x1ff].size;

}
