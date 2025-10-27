int av_get_channel_layout_nb_channels(int64_t channel_layout)

{

    int count;

    uint64_t x = channel_layout;

    for (count = 0; x; count++)

        x &= x-1; // unset lowest set bit

    return count;

}
