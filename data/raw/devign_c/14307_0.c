int ff_dxva2_is_d3d11(const AVCodecContext *avctx)

{

    if (CONFIG_D3D11VA)

        return avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD;

    else

        return 0;

}
