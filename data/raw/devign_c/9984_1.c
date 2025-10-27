static int set_sps(HEVCContext *s, const HEVCSPS *sps)

{

    #define HWACCEL_MAX (CONFIG_HEVC_DXVA2_HWACCEL + CONFIG_HEVC_D3D11VA_HWACCEL)

    enum AVPixelFormat pix_fmts[HWACCEL_MAX + 2], *fmt = pix_fmts;

    int ret;



    export_stream_params(s->avctx, &s->ps, sps);



    pic_arrays_free(s);

    ret = pic_arrays_init(s, sps);

    if (ret < 0)

        goto fail;



    if (sps->pix_fmt == AV_PIX_FMT_YUV420P || sps->pix_fmt == AV_PIX_FMT_YUVJ420P) {

#if CONFIG_HEVC_DXVA2_HWACCEL

        *fmt++ = AV_PIX_FMT_DXVA2_VLD;

#endif

#if CONFIG_HEVC_D3D11VA_HWACCEL

        *fmt++ = AV_PIX_FMT_D3D11VA_VLD;

#endif

    }



    *fmt++ = sps->pix_fmt;

    *fmt = AV_PIX_FMT_NONE;



    ret = ff_get_format(s->avctx, pix_fmts);

    if (ret < 0)

        goto fail;

    s->avctx->pix_fmt = ret;



    ff_hevc_pred_init(&s->hpc,     sps->bit_depth);

    ff_hevc_dsp_init (&s->hevcdsp, sps->bit_depth);

    ff_videodsp_init (&s->vdsp,    sps->bit_depth);



    if (sps->sao_enabled && !s->avctx->hwaccel) {

        av_frame_unref(s->tmp_frame);

        ret = ff_get_buffer(s->avctx, s->tmp_frame, AV_GET_BUFFER_FLAG_REF);

        if (ret < 0)

            goto fail;

        s->frame = s->tmp_frame;

    }



    s->ps.sps = sps;

    s->ps.vps = (HEVCVPS*) s->ps.vps_list[s->ps.sps->vps_id]->data;



    return 0;



fail:

    pic_arrays_free(s);

    s->ps.sps = NULL;

    return ret;

}
