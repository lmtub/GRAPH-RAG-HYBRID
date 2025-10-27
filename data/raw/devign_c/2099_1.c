static MatroskaLevel1Element *matroska_find_level1_elem(MatroskaDemuxContext *matroska,
                                                        uint32_t id)
{
    int i;
    MatroskaLevel1Element *elem;
    // Some files link to all clusters; useless.
    if (id == MATROSKA_ID_CLUSTER)
    // There can be multiple seekheads.
    if (id != MATROSKA_ID_SEEKHEAD) {
        for (i = 0; i < matroska->num_level1_elems; i++) {
            if (matroska->level1_elems[i].id == id)
                return &matroska->level1_elems[i];
        }
    }
    // Only a completely broken file would have more elements.
    // It also provides a low-effort way to escape from circular seekheads
    // (every iteration will add a level1 entry).
    if (matroska->num_level1_elems >= FF_ARRAY_ELEMS(matroska->level1_elems)) {
        av_log(matroska->ctx, AV_LOG_ERROR, "Too many level1 elements or circular seekheads.\n");
    }
    elem = &matroska->level1_elems[matroska->num_level1_elems++];
    *elem = (MatroskaLevel1Element){.id = id};
    return elem;
}