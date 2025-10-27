static int raw_probe_geometry(BlockDriverState *bs, HDGeometry *geo)

{

    BDRVRawState *s = bs->opaque;

    if (s->offset || s->has_size) {

        return -ENOTSUP;

    }

    return bdrv_probe_geometry(bs->file->bs, geo);

}
