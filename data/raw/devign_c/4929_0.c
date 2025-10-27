static void vmdk_refresh_limits(BlockDriverState *bs, Error **errp)

{

    BDRVVmdkState *s = bs->opaque;

    int i;



    for (i = 0; i < s->num_extents; i++) {

        if (!s->extents[i].flat) {

            bs->bl.write_zeroes_alignment =

                MAX(bs->bl.write_zeroes_alignment,

                    s->extents[i].cluster_sectors);

        }

    }

}
