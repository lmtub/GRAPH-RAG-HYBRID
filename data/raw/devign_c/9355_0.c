static int init_opaque_surf(QSVContext *qsv)

{

    AVQSVContext *hwctx_enc = qsv->ost->enc_ctx->hwaccel_context;

    mfxFrameSurface1 *surfaces;

    int i;



    qsv->nb_surfaces = hwctx_enc->nb_opaque_surfaces;



    qsv->opaque_surfaces_buf = av_buffer_ref(hwctx_enc->opaque_surfaces);

    qsv->surface_ptrs        = av_mallocz_array(qsv->nb_surfaces, sizeof(*qsv->surface_ptrs));

    qsv->surface_used        = av_mallocz_array(qsv->nb_surfaces, sizeof(*qsv->surface_used));

    if (!qsv->opaque_surfaces_buf || !qsv->surface_ptrs || !qsv->surface_used)

        return AVERROR(ENOMEM);



    surfaces = (mfxFrameSurface1*)qsv->opaque_surfaces_buf->data;

    for (i = 0; i < qsv->nb_surfaces; i++)

        qsv->surface_ptrs[i] = surfaces + i;



    qsv->opaque_alloc.Out.Surfaces   = qsv->surface_ptrs;

    qsv->opaque_alloc.Out.NumSurface = qsv->nb_surfaces;

    qsv->opaque_alloc.Out.Type       = hwctx_enc->opaque_alloc_type;



    qsv->opaque_alloc.Header.BufferId = MFX_EXTBUFF_OPAQUE_SURFACE_ALLOCATION;

    qsv->opaque_alloc.Header.BufferSz = sizeof(qsv->opaque_alloc);

    qsv->ext_buffers[0]               = (mfxExtBuffer*)&qsv->opaque_alloc;



    return 0;

}
