static int find_snapshot_by_id_or_name(BlockDriverState *bs, const char *name)

{

    BDRVQcowState *s = bs->opaque;

    int i, ret;



    ret = find_snapshot_by_id(bs, name);

    if (ret >= 0)

        return ret;

    for(i = 0; i < s->nb_snapshots; i++) {

        if (!strcmp(s->snapshots[i].name, name))

            return i;

    }

    return -1;

}
