static int64_t *concat_channels_lists(const int64_t *layouts, const int *counts)

{

    int nb_layouts = 0, nb_counts = 0, i;

    int64_t *list;



    if (layouts)

        for (; layouts[nb_layouts] != -1; nb_layouts++);

    if (counts)

        for (; counts[nb_counts] != -1; nb_counts++);

    if (nb_counts > INT_MAX - 1 - nb_layouts)

        return NULL;

    if (!(list = av_calloc(nb_layouts + nb_counts + 1, sizeof(*list))))

        return NULL;

    for (i = 0; i < nb_layouts; i++)

        list[i] = layouts[i];

    for (i = 0; i < nb_counts; i++)

        list[nb_layouts + i] = FF_COUNT2LAYOUT(counts[i]);

    list[nb_layouts + nb_counts] = -1;

    return list;

}
