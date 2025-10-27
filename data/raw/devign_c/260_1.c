static int mxf_read_close(AVFormatContext *s)

{

    MXFContext *mxf = s->priv_data;

    MXFIndexTableSegment *seg;

    int i;



    av_freep(&mxf->packages_refs);



    for (i = 0; i < s->nb_streams; i++)

        s->streams[i]->priv_data = NULL;



    for (i = 0; i < mxf->metadata_sets_count; i++) {

        switch (mxf->metadata_sets[i]->type) {

        case MultipleDescriptor:

            av_freep(&((MXFDescriptor *)mxf->metadata_sets[i])->sub_descriptors_refs);

            break;

        case Sequence:

            av_freep(&((MXFSequence *)mxf->metadata_sets[i])->structural_components_refs);

            break;

        case SourcePackage:

        case MaterialPackage:

            av_freep(&((MXFPackage *)mxf->metadata_sets[i])->tracks_refs);

            break;

        case IndexTableSegment:

            seg = (MXFIndexTableSegment *)mxf->metadata_sets[i];

            av_freep(&seg->temporal_offset_entries);

            av_freep(&seg->flag_entries);

            av_freep(&seg->stream_offset_entries);

            break;

        default:

            break;

        }

        av_freep(&mxf->metadata_sets[i]);

    }

    av_freep(&mxf->partitions);

    av_freep(&mxf->metadata_sets);

    av_freep(&mxf->aesc);

    av_freep(&mxf->local_tags);



    for (i = 0; i < mxf->nb_index_tables; i++) {

        av_freep(&mxf->index_tables[i].segments);

        av_freep(&mxf->index_tables[i].ptses);

        av_freep(&mxf->index_tables[i].fake_index);

    }

    av_freep(&mxf->index_tables);



    return 0;

}
