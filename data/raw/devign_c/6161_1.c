static uint64_t get_channel_layout_single(const char *name, int name_len)

{

    int i;

    char *end;

    int64_t layout;



    for (i = 0; i < FF_ARRAY_ELEMS(channel_layout_map); i++) {

        if (strlen(channel_layout_map[i].name) == name_len &&

            !memcmp(channel_layout_map[i].name, name, name_len))

            return channel_layout_map[i].layout;

    }

    for (i = 0; i < FF_ARRAY_ELEMS(channel_names); i++)

        if (channel_names[i].name &&

            strlen(channel_names[i].name) == name_len &&

            !memcmp(channel_names[i].name, name, name_len))

            return (int64_t)1 << i;

    i = strtol(name, &end, 10);



    if ((end + 1 - name == name_len && *end  == 'c'))

        return av_get_default_channel_layout(i);



    layout = strtoll(name, &end, 0);

    if (end - name == name_len)

        return FFMAX(layout, 0);

    return 0;

}
