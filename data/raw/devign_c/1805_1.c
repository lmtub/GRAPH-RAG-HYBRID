static int alloc_table(VLC *vlc, int size)

{

    int index;

    index = vlc->table_size;

    vlc->table_size += size;

    if (vlc->table_size > vlc->table_allocated) {

        vlc->table_allocated += (1 << vlc->bits);

        vlc->table = av_realloc(vlc->table,

                                sizeof(VLC_TYPE) * 2 * vlc->table_allocated);

        if (!vlc->table)

            return -1;

    }

    return index;

}
