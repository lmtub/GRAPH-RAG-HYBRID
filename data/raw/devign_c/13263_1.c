static void init_parse_context(OptionParseContext *octx,

                               const OptionGroupDef *groups, int nb_groups)

{

    static const OptionGroupDef global_group = { "global" };

    int i;



    memset(octx, 0, sizeof(*octx));



    octx->nb_groups = nb_groups;

    octx->groups    = av_mallocz(sizeof(*octx->groups) * octx->nb_groups);

    if (!octx->groups)

        exit(1);



    for (i = 0; i < octx->nb_groups; i++)

        octx->groups[i].group_def = &groups[i];



    octx->global_opts.group_def = &global_group;

    octx->global_opts.arg       = "";



    init_opts();

}
