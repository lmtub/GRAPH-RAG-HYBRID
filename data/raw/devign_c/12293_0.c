static int vaapi_encode_mjpeg_init_picture_params(AVCodecContext *avctx,

                                                  VAAPIEncodePicture *pic)

{

    VAAPIEncodeContext               *ctx = avctx->priv_data;

    VAEncPictureParameterBufferJPEG *vpic = pic->codec_picture_params;

    VAAPIEncodeMJPEGContext         *priv = ctx->priv_data;



    vpic->reconstructed_picture = pic->recon_surface;

    vpic->coded_buf = pic->output_buffer;



    vpic->picture_width  = ctx->input_width;

    vpic->picture_height = ctx->input_height;



    vpic->pic_flags.bits.profile      = 0;

    vpic->pic_flags.bits.progressive  = 0;

    vpic->pic_flags.bits.huffman      = 1;

    vpic->pic_flags.bits.interleaved  = 0;

    vpic->pic_flags.bits.differential = 0;



    vpic->sample_bit_depth = 8;

    vpic->num_scan = 1;



    vpic->num_components = 3;



    vpic->component_id[0] = 1;

    vpic->component_id[1] = 2;

    vpic->component_id[2] = 3;



    priv->component_subsample_h[0] = 2;

    priv->component_subsample_v[0] = 2;

    priv->component_subsample_h[1] = 1;

    priv->component_subsample_v[1] = 1;

    priv->component_subsample_h[2] = 1;

    priv->component_subsample_v[2] = 1;



    vpic->quantiser_table_selector[0] = 0;

    vpic->quantiser_table_selector[1] = 1;

    vpic->quantiser_table_selector[2] = 1;



    vpic->quality = priv->quality;



    pic->nb_slices = 1;



    return 0;

}
