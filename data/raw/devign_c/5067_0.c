static int split_field_ref_list(Picture *dest, int dest_len,

                                Picture *src,  int src_len,

                                int parity,    int long_i){



    int i = split_field_half_ref_list(dest, dest_len, src, long_i, parity);

    dest += i;

    dest_len -= i;



    i += split_field_half_ref_list(dest, dest_len, src + long_i,

                                   src_len - long_i, parity);

    return i;

}
