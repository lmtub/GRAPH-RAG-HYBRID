static int commit_bitstream_and_slice_buffer(AVCodecContext *avctx,

                                             DECODER_BUFFER_DESC *bs,

                                             DECODER_BUFFER_DESC *sc)

{

    const struct MpegEncContext *s = avctx->priv_data;

    AVDXVAContext *ctx = avctx->hwaccel_context;

    struct dxva2_picture_context *ctx_pic =

        s->current_picture_ptr->hwaccel_picture_private;

    const int is_field = s->picture_structure != PICT_FRAME;

    const unsigned mb_count = s->mb_width * (s->mb_height >> is_field);

    void     *dxva_data_ptr;

    uint8_t  *dxva_data, *current, *end;

    unsigned dxva_size;

    unsigned i;

    unsigned type;



#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD) {

        type = D3D11_VIDEO_DECODER_BUFFER_BITSTREAM;

        if (FAILED(ID3D11VideoContext_GetDecoderBuffer(D3D11VA_CONTEXT(ctx)->video_context,

                                                       D3D11VA_CONTEXT(ctx)->decoder,

                                                       type,

                                                       &dxva_size, &dxva_data_ptr)))

            return -1;

    }

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        type = DXVA2_BitStreamDateBufferType;

        if (FAILED(IDirectXVideoDecoder_GetBuffer(DXVA2_CONTEXT(ctx)->decoder,

                                                  type,

                                                  &dxva_data_ptr, &dxva_size)))

            return -1;

    }

#endif



    dxva_data = dxva_data_ptr;

    current = dxva_data;

    end = dxva_data + dxva_size;



    for (i = 0; i < ctx_pic->slice_count; i++) {

        DXVA_SliceInfo *slice = &ctx_pic->slice[i];

        unsigned position = slice->dwSliceDataLocation;

        unsigned size     = slice->dwSliceBitsInBuffer / 8;

        if (size > end - current) {

            av_log(avctx, AV_LOG_ERROR, "Failed to build bitstream");

            break;

        }

        slice->dwSliceDataLocation = current - dxva_data;



        if (i < ctx_pic->slice_count - 1)

            slice->wNumberMBsInSlice =

                slice[1].wNumberMBsInSlice - slice[0].wNumberMBsInSlice;

        else

            slice->wNumberMBsInSlice =

                mb_count - slice[0].wNumberMBsInSlice;



        memcpy(current, &ctx_pic->bitstream[position], size);

        current += size;

    }

#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD)

        if (FAILED(ID3D11VideoContext_ReleaseDecoderBuffer(D3D11VA_CONTEXT(ctx)->video_context, D3D11VA_CONTEXT(ctx)->decoder, type)))

            return -1;

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD)

        if (FAILED(IDirectXVideoDecoder_ReleaseBuffer(DXVA2_CONTEXT(ctx)->decoder, type)))

            return -1;

#endif

    if (i < ctx_pic->slice_count)

        return -1;



#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD) {

        D3D11_VIDEO_DECODER_BUFFER_DESC *dsc11 = bs;

        memset(dsc11, 0, sizeof(*dsc11));

        dsc11->BufferType           = type;

        dsc11->DataSize             = current - dxva_data;

        dsc11->NumMBsInBuffer       = mb_count;



        type = D3D11_VIDEO_DECODER_BUFFER_SLICE_CONTROL;

    }

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        DXVA2_DecodeBufferDesc *dsc2 = bs;

        memset(dsc2, 0, sizeof(*dsc2));

        dsc2->CompressedBufferType = type;

        dsc2->DataSize             = current - dxva_data;

        dsc2->NumMBsInBuffer       = mb_count;



        type = DXVA2_SliceControlBufferType;

    }

#endif



    return ff_dxva2_commit_buffer(avctx, ctx, sc,

                                  type,

                                  ctx_pic->slice,

                                  ctx_pic->slice_count * sizeof(*ctx_pic->slice),

                                  mb_count);

}
