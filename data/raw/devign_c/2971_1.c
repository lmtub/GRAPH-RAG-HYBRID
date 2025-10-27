static int mxf_add_metadata_set(MXFContext *mxf, void *metadata_set)

{



    mxf->metadata_sets = av_realloc(mxf->metadata_sets, (mxf->metadata_sets_count + 1) * sizeof(*mxf->metadata_sets));

    if (!mxf->metadata_sets)

        return -1;

    mxf->metadata_sets[mxf->metadata_sets_count] = metadata_set;

    mxf->metadata_sets_count++;

    return 0;

}